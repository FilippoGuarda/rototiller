#!/usr/bin/env python3
import math
import itertools
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set

import networkx as nx
from ortools.linear_solver import pywraplp
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import MarkerArray

from task_logger import TaskLogger, TaskLogRecord
from task_allocation_helpers import StationConfig, Task, RobotState, build_full_graph_with_stations


class TaskAllocationNode(Node):
    def __init__(self):
        super().__init__(
            'task_allocation_node',
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        self.num_robots = int(self.get_parameter("num_robots").value)
        self.global_frame = str(self.get_parameter("global_frame").value)
        self.robot_prefix = str(self.get_parameter("robot_base_frame_prefix").value)
        self.robot_suffix = str(self.get_parameter("robot_base_frame_suffix").value)
        self.node_match_threshold = float(
            self.get_parameter("node_match_threshold_m").value
        )

        if not self.has_parameter("k_nearest_graph_nodes"):
            self.declare_parameter("k_nearest_graph_nodes", 3)
        self.k_nearest = int(self.get_parameter("k_nearest_graph_nodes").value)

        self.alpha_d = float(self.get_parameter("alpha_distance").value)
        self.alpha_u = float(self.get_parameter("alpha_usage").value)
        self.alpha_b = float(self.get_parameter("alpha_battery").value)
        self.update_rate_hz = float(self.get_parameter("update_rate_hz").value)

        default_battery = float(self.get_parameter("robot_defaults.battery_soc").value)
        default_max_range = float(self.get_parameter("robot_defaults.max_range_m").value)
        default_usage = float(self.get_parameter("robot_defaults.usage_index").value)

        if not self.has_parameter("robot_defaults.footprint_radius"):
            self.declare_parameter("robot_defaults.footprint_radius", 0.5)
        self.robot_footprint_radius = float(
            self.get_parameter("robot_defaults.footprint_radius").value
        )

        if not self.has_parameter('task_batch_size'):
            self.declare_parameter('task_batch_size', 5)
        self.task_batch_size = int(self.get_parameter('task_batch_size').value)

        if not self.has_parameter("max_allocation_attempts"):
            self.declare_parameter("max_allocation_attempts", 40)
        self.max_allocation_attempts = int(self.get_parameter("max_allocation_attempts").value)

        if not self.has_parameter("retry_cooldown_sec"):
            self.declare_parameter("retry_cooldown_sec", 5.0)
        self.retry_cooldown_sec = float(self.get_parameter("retry_cooldown_sec").value)

        self.station_dwell_time = 1.5
        self.station_reservation_timeout = 300.0

        base_log_file_path = str(self.get_parameter("log_file_path").value)
        if not self.has_parameter("run_id"):
            self.declare_parameter("run_id", "default_run")
        self.run_id = str(self.get_parameter("run_id").value)

        if not self.has_parameter("algorithm_type"):
            self.declare_parameter("algorithm_type", "multi_chomp_scip")
        self.algorithm_type = str(self.get_parameter("algorithm_type").value)

        base, ext = os.path.splitext(base_log_file_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = (
            f"{base}_{self.algorithm_type}_{self.run_id}_{timestamp}{ext}"
        )

        self.logger = TaskLogger(self.log_file_path)

        self.reentrant_callback_group = ReentrantCallbackGroup()
        self.graph_callback_group = MutuallyExclusiveCallbackGroup()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.robot_states: Dict[int, RobotState] = {
            i: RobotState(
                robot_id=i,
                battery_soc=default_battery,
                max_range_m=default_max_range,
                usage_index=default_usage,
            )
            for i in range(1, self.num_robots + 1)
        }

        self.stations: Dict[str, StationConfig] = {}
        self.stations_by_type: Dict[str, List[StationConfig]] = {
            "a": [], "b": [], "c": [], "p": [],
        }

        self.load_station_config()

        parking_stations = sorted(
            s.name for s in self.stations_by_type["p"] if s.online
        )
        assert len(parking_stations) >= self.num_robots

        # Immutable fixed home parking per robot — never overridden by optimizer
        self.robot_parking_station: Dict[int, str] = {
            r: parking_stations[r - 1]
            for r in range(1, self.num_robots + 1)
        }

        self.task_queue: List[Task] = []
        self.robot_tasks: Dict[int, Optional[dict]] = {
            i: None for i in range(1, self.num_robots + 1)
        }

        self.occupied_stations: Set[str] = set()
        self.physical_occupancy: Dict[str, int] = {}

        self.graph = nx.Graph()
        self.full_graph = nx.Graph()
        self.graph_nodes_map_coords: Dict[int, Tuple[float, float]] = {}
        self.station_nodes: Dict[str, int] = {}

        qos_profile = QoSProfile(
            depth=10,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        self.graph_sub = self.create_subscription(
            String, 'skeleton_graph_json', self.graph_callback,
            qos_profile, callback_group=self.graph_callback_group,
        )
        self.task_sub = self.create_subscription(
            String, "/tasks", self.task_callback,
            10, callback_group=self.reentrant_callback_group,
        )

        self.goal_pubs: Dict[int, rclpy.publisher.Publisher] = {}
        for robot_id in range(1, self.num_robots + 1):
            topic = f"/{self.robot_prefix}{robot_id}/spades_goal"
            self.goal_pubs[robot_id] = self.create_publisher(PoseStamped, topic, 10)

        self.create_timer(
            1.0 / self.update_rate_hz,
            self.update_callback,
            callback_group=self.reentrant_callback_group,
        )

        self.get_logger().info(
            f"Task Allocation Node initialized. Robots: {self.num_robots}, "
            f"Update rate: {self.update_rate_hz} Hz, "
            f"Log file: {self.log_file_path}"
        )
        for r, ps in self.robot_parking_station.items():
            self.get_logger().info(f"  Robot {r} -> fixed home parking: {ps}")

    # --------------------------------------------------------------------- #
    # Station config
    # --------------------------------------------------------------------- #

    def load_station_config(self) -> None:
        for stype in ["a", "b", "c", "p"]:
            names_param = f"stations.{stype}.names"
            if not self.has_parameter(names_param):
                continue
            names = list(self.get_parameter(names_param).value)
            for sname in names:
                x = float(self.get_parameter(f"stations.{stype}.{sname}.x").value)
                y = float(self.get_parameter(f"stations.{stype}.{sname}.y").value)
                online = bool(self.get_parameter(f"stations.{stype}.{sname}.online").value)
                station = StationConfig(sname, stype, (x, y), online)
                self.stations[sname] = station
                self.stations_by_type[stype].append(station)

        self.get_logger().info(
            f"Loaded stations: A={len(self.stations_by_type['a'])}, "
            f"B={len(self.stations_by_type['b'])}, "
            f"C={len(self.stations_by_type['c'])}, "
            f"P={len(self.stations_by_type['p'])}"
        )

    # --------------------------------------------------------------------- #
    # Graph handling
    # --------------------------------------------------------------------- #

    def graph_callback(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            self.graph = nx.node_link_graph(data)
            self.graph_nodes_map_coords.clear()
            for n, d in self.graph.nodes(data=True):
                if 'pos' in d:
                    self.graph_nodes_map_coords[n] = (d['pos'][0], d['pos'][1])
            self.get_logger().debug(f"Graph JSON parsed: {self.graph.number_of_nodes()} nodes")
            self.inject_stations_to_graph()
        except Exception as e:
            self.get_logger().error(f"Failed to parse graph JSON: {e}")

    def inject_stations_to_graph(self) -> None:
        self.full_graph, self.station_nodes = build_full_graph_with_stations(
            self.graph, self.stations, self.k_nearest, self.graph_nodes_map_coords,
        )

    def find_closest_node(
        self, x: float, y: float, in_full_graph: bool = True,
        threshold: Optional[float] = None,
    ) -> Optional[int]:
        best_node: Optional[int] = None
        best_dist = float("inf")
        max_dist = threshold if threshold is not None else self.node_match_threshold
        nodes = self.full_graph.nodes(data=True) if in_full_graph else self.graph.nodes(data=True)

        if not nodes:
            for n_id, coords in self.graph_nodes_map_coords.items():
                d = math.hypot(x - coords[0], y - coords[1])
                if d < best_dist and d <= max_dist:
                    best_dist, best_node = d, n_id
            return best_node

        for n_id, data in nodes:
            pos = data.get("pos")
            if not pos:
                continue
            d = math.hypot(x - pos[0], y - pos[1])
            if d < best_dist and d <= max_dist:
                best_dist, best_node = d, n_id
        return best_node

    # --------------------------------------------------------------------- #
    # Robot pose
    # --------------------------------------------------------------------- #

    def get_robot_position(self, robot_id: int) -> Optional[Tuple[float, float]]:
        frame = f"{self.robot_prefix}{robot_id}{self.robot_suffix}"
        try:
            t = self.tf_buffer.lookup_transform(
                self.global_frame, frame, rclpy.time.Time()
            )
            return (t.transform.translation.x, t.transform.translation.y)
        except Exception:
            return None

    # --------------------------------------------------------------------- #
    # Status display
    # --------------------------------------------------------------------- #

    def print_status_table(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n" + "="*90)
        print(f" TASK ALLOCATION STATUS | PENDING: {len(self.task_queue)} | ROBOTS: {self.num_robots} ")
        print("="*90)
        print("\n [ ACTIVE ROBOT ASSIGNMENTS ]")
        print(f" {'ROBOT':<8} | {'TASK ID':<10} | {'LOCATION':<15} | {'COST':<8} | {'STATUS'}")
        print(" " + "-"*86)
        for r in range(1, self.num_robots + 1):
            task_info = self.robot_tasks[r]
            if task_info is None:
                print(f" Robot {r:<2} | {'-':<10} | {'-':<15} | {'-':<8} | IDLE / Waiting for task")
            else:
                task_id = task_info['task_id']
                path = task_info['path']
                idx = task_info['current_idx']
                cost = task_info.get('allocation_cost', 0.0) or 0.0
                is_parking = task_info.get('is_parking', False)
                loc = f"{path[idx]} ({idx+1}/{len(path)})" if idx < len(path) else "DONE"
                if is_parking:
                    status_str = f"Parking @ {self.robot_parking_station[r]}"
                else:
                    status_str = f"Executing sequence {'->'.join(path)}"
                print(f" Robot {r:<2} | {task_id:<10} | {loc:<15} | {float(cost):<8.1f} | {status_str}")

        print("\n [ PENDING TASK QUEUE ]")
        if not self.task_queue:
            print("  No tasks pending in queue.")
        else:
            print(f" {'PRIORITY':<8} | {'TASK ID':<10} | {'REQUESTED STATIONS':<30} | {'ATTEMPTS'}")
            print(" " + "-"*86)
            for t in sorted(self.task_queue, key=lambda x: x.priority, reverse=True):
                stations_str = "->".join(t.stations)
                print(f" {t.priority:<8.1f} | {t.task_id:<10} | {stations_str:<30} | {t.allocation_attempts}/{self.max_allocation_attempts}")
        print("="*90 + "\n")

    # --------------------------------------------------------------------- #
    # Task reception
    # NOTE: 'p' is NOT auto-appended here. Home parking is handled
    #       exclusively by _dispatch_home_parking().
    # --------------------------------------------------------------------- #

    def task_callback(self, msg: String) -> None:
        parts = [p.strip() for p in msg.data.split(",")]
        if len(parts) < 2:
            return

        task_id = parts[0]
        stations_str = parts[1].split("|")
        priority = float(parts[2]) if len(parts) > 2 else 1.0

        valid_tokens = set(self.stations.keys()) | set(self.stations_by_type.keys())
        if any(t not in valid_tokens for t in stations_str):
            self.get_logger().error("Task contains invalid station type")
            return

        task = Task(
            task_id=task_id,
            timestamp=self.get_clock().now().nanoseconds / 1e9,
            stations=stations_str,
            priority=priority,
        )
        self.task_queue.append(task)

    # --------------------------------------------------------------------- #
    # Home parking dispatch — bypasses optimizer entirely
    # --------------------------------------------------------------------- #

    def _dispatch_home_parking(self, robot_id: int) -> None:
        """Send robot to its fixed home parking station without using the optimizer."""
        if self.robot_tasks[robot_id] is not None:
            return

        park_station = self.robot_parking_station[robot_id]

        # Already physically there
        if self.physical_occupancy.get(park_station) == robot_id:
            return

        # Home parking is occupied by another robot — wait
        if park_station in self.occupied_stations:
            occupant = self.physical_occupancy.get(park_station)
            if occupant is not None and occupant != robot_id:
                self.get_logger().warn(
                    f"Robot {robot_id} home parking {park_station} physically "
                    f"occupied by robot {occupant}. Waiting."
                )
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9
        self.robot_tasks[robot_id] = {
            'task_id': f'park_{robot_id}',
            'path': [park_station],
            'current_idx': 0,
            'start_time': now_sec,
            'arrival_time': None,
            'station_reservation_time': now_sec,
            'allocation_cost': 0.0,
            'had_collision': False,
            # is_parking=True immediately so robot is preemptible during transit
            'is_parking': True,
        }
        self.occupied_stations.add(park_station)
        self.send_to_station(robot_id, park_station, log_dispatch=True)
        self.get_logger().info(
            f"Robot {robot_id} dispatched to fixed home parking: {park_station}"
        )

    # --------------------------------------------------------------------- #
    # OR-Tools Batch Allocation
    # --------------------------------------------------------------------- #

    def allocate_task_batch(self, tasks: List[Task]) -> set:
        assigned_task_ids = set()
        sorted_tasks = sorted(tasks, key=lambda x: x.priority, reverse=True)
        now_sec = self.get_clock().now().nanoseconds / 1e9

        eligible_robots = [
            r for r in range(1, self.num_robots + 1)
            if self.robot_tasks[r] is None or self.robot_tasks[r].get('is_parking', False)
        ]

        for task in sorted_tasks:
            solver = pywraplp.Solver.CreateSolver('SCIP')
            if not solver:
                self.get_logger().error("OR-Tools SCIP solver unavailable.")
                return assigned_task_ids

            # ---- 1. Determine tails ----------------------------------------
            robot_tails: Dict[int, Optional[int]] = {}
            for r in eligible_robots:
                task_info = self.robot_tasks[r]
                if task_info is not None and not task_info.get('is_parking', False):
                    last_station = task_info['path'][-1]
                    robot_tails[r] = self.station_nodes.get(last_station)
                else:
                    pos = self.get_robot_position(r)
                    if pos:
                        best, best_dist = None, float('inf')
                        for nid, coords in self.graph_nodes_map_coords.items():
                            d = math.hypot(pos[0] - coords[0], pos[1] - coords[1])
                            if d < best_dist:
                                best_dist, best = d, nid
                        robot_tails[r] = best
                    else:
                        robot_tails[r] = None
                        self.get_logger().warn(
                            f'Robot {r}: no TF position available, excluded from allocation'
                        )

            # ---- 2. Build decision variables (per-robot combinations) -------
            x: Dict[Tuple[int, int], any] = {}
            costs: Dict[Tuple[int, int], float] = {}
            # Mutual exclusion: only path[0] (the immediate target).
            # Intermediate stations are NOT pre-reserved to avoid deadlocks —
            # the cost minimizer naturally separates robots across stations.
            station_usage_vars: Dict[str, list] = {}
            # Store per-robot combination lists for the assignment block
            robot_combinations: Dict[int, list] = {}

            for r in eligible_robots:
                tail_node = robot_tails.get(r)
                if tail_node is None:
                    continue

                # Expand station tokens into concrete names for THIS robot.
                groups = []
                for item in task.stations:
                    if (
                        item == "p"
                        or (item in self.stations and self.stations[item].station_type == "p")
                    ):
                        # Always resolve 'p' to this robot's fixed home parking
                        groups.append([self.robot_parking_station[r]])
                    elif item in self.stations:
                        groups.append([item])
                    elif item in self.stations_by_type:
                        # Pre-filter: only offer stations that are currently free
                        available = [
                            s.name for s in self.stations_by_type[item]
                            if s.online
                            and s.name not in self.occupied_stations
                            and self.physical_occupancy.get(s.name) in (None, r)
                        ]
                        groups.append(available)

                if any(not g for g in groups):
                    continue

                combinations = list(itertools.product(*groups))
                robot_combinations[r] = combinations

                robot = self.robot_states[r]
                remaining = robot.battery_soc * robot.max_range_m

                for c_idx, path in enumerate(combinations):
                    # Reject combinations that assign this robot the wrong parking station
                    parking_violation = any(
                        self.stations[sname].station_type == "p"
                        and sname != self.robot_parking_station[r]
                        for sname in path
                    )
                    if parking_violation:
                        continue

                    # Reject combinations where path[0] is already hard-reserved.
                    # Intermediate stations are NOT blocked here — they are
                    # momentary occupancies that will be free when needed.
                    first_station = path[0]
                    if (
                        first_station in self.occupied_stations
                        or (
                            self.physical_occupancy.get(first_station) not in (None, r)
                        )
                    ):
                        continue

                    try:
                        target_node = self.station_nodes.get(first_station)
                        if tail_node not in self.full_graph or target_node not in self.full_graph:
                            continue

                        dist = nx.shortest_path_length(
                            self.full_graph, tail_node, target_node, weight='weight'
                        )
                        d_cost = self.alpha_d * dist
                        b_cost = float('inf') if remaining <= 0.0 else self.alpha_b * (dist / remaining)
                        u_cost = self.alpha_u * robot.usage_index
                        cost = d_cost + b_cost + u_cost

                        for i in range(len(path) - 1):
                            n1 = self.station_nodes[path[i]]
                            n2 = self.station_nodes[path[i + 1]]
                            cost += self.alpha_d * nx.shortest_path_length(
                                self.full_graph, n1, n2, weight='weight'
                            )

                        var = solver.IntVar(0, 1, f'x_{r}_{c_idx}')
                        x[(r, c_idx)] = var
                        costs[(r, c_idx)] = cost

                        # Mutual exclusion only on path[0] (immediate target).
                        # This prevents two robots from racing to the same first
                        # station without causing deadlocks on intermediate ones.
                        if first_station not in station_usage_vars:
                            station_usage_vars[first_station] = []
                        station_usage_vars[first_station].append(var)

                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue

            if not x:
                continue

            # ---- 3. Constraints --------------------------------------------
            # Exactly one (robot, combination) pair is selected per task
            solver.Add(solver.Sum(x.values()) == 1)

            # No two robots may target the same first station simultaneously
            for sname, vars_using_station in station_usage_vars.items():
                if len(vars_using_station) > 1:
                    solver.Add(solver.Sum(vars_using_station) <= 1)

            solver.Minimize(
                solver.Sum(var * costs[key] for key, var in x.items())
            )

            # ---- 4. Solve and assign ----------------------------------------
            status = solver.Solve()

            if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
                for (r, c_idx), var in x.items():
                    if var.solution_value() > 0.5:
                        combinations = robot_combinations[r]
                        selected_path = list(combinations[c_idx])

                        is_parking = (
                            len(selected_path) == 1
                            and self.stations[selected_path[0]].station_type == "p"
                        )

                        assigned_task_ids.add(task.task_id)

                        # Preempt parking if robot is currently heading home
                        if self.robot_tasks[r] is not None and self.robot_tasks[r].get("is_parking", False):
                            old_task = self.robot_tasks[r]
                            self.occupied_stations.discard(old_task["path"][0])
                            self.robot_states[r].usage_index = max(0.0, self.robot_states[r].usage_index - 1.0)
                            # Uncomment for logging when parking gets preempted
                            # self.logger.log(
                            #     TaskLogRecord(
                            #         timestamp=now_sec,
                            #         run_id=self.run_id,
                            #         task_id=old_task["task_id"],
                            #         robot_id=f"robot_{r}",
                            #         event="CANCELLED",
                            #         status="PREEMPTED",
                            #         allocation_cost=old_task.get("allocation_cost"),
                            #         duration=now_sec - old_task["start_time"],
                            #         path="->".join(old_task["path"]),
                            #         collision_flag=int(old_task.get("had_collision", False)),
                            #         message=(
                            #             f"Parking at {old_task['path'][0]} preempted by new task. "
                            #             f"Robot {r} will return to {self.robot_parking_station[r]} on completion."
                            #         ),
                            #     )
                            # )
                            self.robot_tasks[r] = None 

                        if self.robot_tasks[r] is None:
                            self.robot_tasks[r] = {
                                'task_id': task.task_id,
                                'path': selected_path,
                                'current_idx': 0,
                                'start_time': now_sec,
                                'arrival_time': None,
                                'station_reservation_time': now_sec,
                                'allocation_cost': costs[(r, c_idx)],
                                'had_collision': False,
                                'is_parking': is_parking,
                            }
                            self.occupied_stations.add(selected_path[0])
                            self.send_to_station(r, selected_path[0], log_dispatch=True)
                        else:
                            continue

                        self.robot_states[r].usage_index += 1.0


                        # Do not log parking tasks
                        if not is_parking:
                            self.logger.log(
                                TaskLogRecord(
                                    timestamp=now_sec,
                                    run_id=self.run_id,
                                    task_id=task.task_id,
                                    robot_id=f"robot_{r}",
                                    event="ASSIGNED",
                                    status="OK",
                                    allocation_cost=costs[(r, c_idx)],
                                    duration=None,
                                    path="->".join(selected_path),
                                    collision_flag=0,
                                    message="Task assigned to robot",
                                )
                            )
                        break

        return assigned_task_ids

    # --------------------------------------------------------------------- #
    # Navigation
    # --------------------------------------------------------------------- #

    def send_to_station(self, robot_id: int, station_name: str, log_dispatch: bool = True) -> None:
        station = self.stations[station_name]
        msg = PoseStamped()
        msg.header.frame_id = self.global_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(station.position[0])
        msg.pose.position.y = float(station.position[1])
        msg.pose.orientation.w = 1.0
        self.goal_pubs[robot_id].publish(msg)

    # --------------------------------------------------------------------- #
    # Collision and occupancy tracking
    # --------------------------------------------------------------------- #

    def update_collision_flags(self) -> None:
        positions: Dict[int, Tuple[float, float]] = {}
        for r in range(1, self.num_robots + 1):
            pos = self.get_robot_position(r)
            if pos is not None:
                positions[r] = pos

        for r_i, r_j in itertools.combinations(positions.keys(), 2):
            xi, yi = positions[r_i]
            xj, yj = positions[r_j]
            d = math.hypot(xi - xj, yi - yj)
            if d < 1.8 * self.robot_footprint_radius:
                if self.robot_tasks[r_i] is not None:
                    self.robot_tasks[r_i]["had_collision"] = True
                if self.robot_tasks[r_j] is not None:
                    self.robot_tasks[r_j]["had_collision"] = True

    def update_physical_occupancy(self) -> None:
        self.physical_occupancy.clear()
        for r in range(1, self.num_robots + 1):
            pos = self.get_robot_position(r)
            if pos is None:
                continue
            x, y = pos
            for s_name, station in self.stations.items():
                if not station.online:
                    continue
                d = math.hypot(x - station.position[0], y - station.position[1])
                if d < self.robot_footprint_radius:
                    self.physical_occupancy[s_name] = r

    # --------------------------------------------------------------------- #
    # Main update loop
    # --------------------------------------------------------------------- #

    def update_callback(self) -> None:
        now_sec = self.get_clock().now().nanoseconds / 1e9

        # Reservation timeout housekeeping
        for r, task_info in self.robot_tasks.items():
            if not task_info:
                continue
            if task_info["current_idx"] < len(task_info["path"]):
                reserved_station = task_info["path"][task_info["current_idx"]]
                t_res = task_info.get("station_reservation_time", None)
                if (
                    t_res is not None
                    and task_info["arrival_time"] is not None
                    and now_sec - t_res > self.station_reservation_timeout
                ):
                    self.occupied_stations.discard(reserved_station)
                    task_info["station_reservation_time"] = now_sec

        self.update_physical_occupancy()
        self.update_collision_flags()

        # Progress active tasks
        for r in range(1, self.num_robots + 1):
            task_info = self.robot_tasks[r]
            if not task_info:
                continue

            path: List[str] = task_info["path"]
            curr_idx: int = task_info["current_idx"]

            # Task fully completed
            if curr_idx >= len(path):
                now_sec = self.get_clock().now().nanoseconds / 1e9
                duration = now_sec - task_info["start_time"]
                self.occupied_stations.discard(path[-1])
                collision_flag = 1 if task_info.get("had_collision", False) else 0

                self.logger.log(
                    TaskLogRecord(
                        timestamp=now_sec,
                        run_id=self.run_id,
                        task_id=task_info["task_id"],
                        robot_id=f"robot_{r}",
                        event="COMPLETED",
                        status="OK",
                        allocation_cost=task_info.get("allocation_cost", None),
                        duration=duration,
                        path="->".join(path),
                        collision_flag=collision_flag,
                        message="Task executed successfully",
                    )
                )
                self.robot_states[r].usage_index = max(0.0, self.robot_states[r].usage_index - 1.0)
                self.robot_tasks[r] = None
                # Immediately send home — no optimizer involvement
                self._dispatch_home_parking(r)
                continue

            curr_station_name = path[curr_idx]
            curr_station = self.stations[curr_station_name]

            r_pos = self.get_robot_position(r)
            if r_pos:
                dist = math.hypot(
                    r_pos[0] - curr_station.position[0],
                    r_pos[1] - curr_station.position[1],
                )

                if dist < self.robot_footprint_radius:
                    task_info["station_reservation_time"] = now_sec

                    if task_info["arrival_time"] is None:
                        task_info["arrival_time"] = now_sec
                        continue

                    # Last station in path — mark complete on next tick
                    if curr_idx == len(path) - 1:
                        task_info["current_idx"] += 1
                        task_info["arrival_time"] = None
                        if self.stations[curr_station_name].station_type == "p":
                            self.occupied_stations.discard(curr_station_name)
                            task_info["is_parking"] = True
                        continue

                    next_station = path[curr_idx + 1]
                    self.occupied_stations.add(next_station)
                    self.occupied_stations.discard(curr_station_name)
                    self.send_to_station(r, next_station, log_dispatch=True)
                    task_info["current_idx"] += 1
                    task_info["arrival_time"] = None

                    if (
                        task_info["current_idx"] == len(path) - 1
                        and self.stations[next_station].station_type == "p"
                    ):
                        task_info["is_parking"] = True

        # Send idle robots home before running the optimizer
        for r in range(1, self.num_robots + 1):
            if self.robot_tasks[r] is None:
                self._dispatch_home_parking(r)

        # Batch optimizer — only for real work tasks
        idle_robots = [
            r for r in range(1, self.num_robots + 1)
            if self.robot_tasks[r] is None or self.robot_tasks[r].get('is_parking', False)
        ]

        if self.task_queue and idle_robots:
            batch_size = min(len(self.task_queue), self.task_batch_size)
            batch = self.task_queue[:batch_size]
            self.task_queue = self.task_queue[batch_size:]

            successfully_assigned_ids = self.allocate_task_batch(batch)

            unassigned_tasks = [t for t in batch if t.task_id not in successfully_assigned_ids]
            for t in unassigned_tasks:
                t.allocation_attempts += 1
                if t.allocation_attempts < self.max_allocation_attempts:
                    self.task_queue.append(t)
                else:
                    self.get_logger().warn(
                        f"Dropping task {t.task_id} after {self.max_allocation_attempts} attempts."
                    )

        self.print_status_table()


def main(args=None):
    rclpy.init(args=args)
    node = TaskAllocationNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()