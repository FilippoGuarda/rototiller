#!/usr/bin/env python3
import math
import itertools
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set

import networkx as nx
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

        if not self.has_parameter("max_allocation_attempts"):
            self.declare_parameter("max_allocation_attempts", 40)
        self.max_allocation_attempts = int(self.get_parameter("max_allocation_attempts").value)

        if not self.has_parameter("retry_cooldown_sec"):
            self.declare_parameter("retry_cooldown_sec", 5.0)
        self.retry_cooldown_sec = float(self.get_parameter("retry_cooldown_sec").value)
        
        self.station_reach_radius = 0.9
        self.station_dwell_time = 1.5
        self.station_reservation_timeout = 300.0

        # Logging / run metadata
        base_log_file_path = str(self.get_parameter("log_file_path").value)
        if not self.has_parameter("run_id"):
            self.declare_parameter("run_id", "default_run")
        self.run_id = str(self.get_parameter("run_id").value)

        if not self.has_parameter("algorithm_type"):
            self.declare_parameter("algorithm_type", "multi_chomp_original")
        self.algorithm_type = str(self.get_parameter("algorithm_type").value)

        base, ext = os.path.splitext(base_log_file_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = (
            f"{base}_{self.algorithm_type}_{self.run_id}_{timestamp}{ext}"
        )
        self.logger = TaskLogger(self.log_file_path)

        # ROS infra
        self.reentrant_callback_group = ReentrantCallbackGroup()
        self.graph_callback_group = MutuallyExclusiveCallbackGroup()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Robot states
        self.robot_states: Dict[int, RobotState] = {
            i: RobotState(
                robot_id=i,
                battery_soc=default_battery,
                max_range_m=default_max_range,
                usage_index=default_usage,
            )
            for i in range(1, self.num_robots + 1)
        }

        # Station configuration
        self.stations: Dict[str, StationConfig] = {}
        self.stations_by_type: Dict[str, List[StationConfig]] = {
            "a": [],
            "b": [],
            "c": [],
            "p": [],  # Added parking type
        }
        self.load_station_config()

        # Task state
        self.task_queue: List[Task] = []
        self.robot_tasks: Dict[int, Optional[dict]] = {
            i: None for i in range(1, self.num_robots + 1)
        }
        self.occupied_stations: Set[str] = set()
        self.physical_occupancy: Set[str] = set()

        # Graphs
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
            String,
            'skeleton_graph_json',
            self.graph_callback,
            qos_profile,
            callback_group=self.graph_callback_group,
        )

        self.task_sub = self.create_subscription(
            String,
            "/tasks",
            self.task_callback,
            10,
            callback_group=self.reentrant_callback_group,
        )

        self.goal_pubs: Dict[int, rclpy.publisher.Publisher] = {}
        for robot_id in range(1, self.num_robots + 1):
            topic = f"/{self.robot_prefix}{robot_id}/spades_goal"
            self.goal_pubs[robot_id] = self.create_publisher(
                PoseStamped, topic, 10
            )

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

    # --------------------------------------------------------------------- #
    # Station configuration
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
            "Loaded stations: "
            f"A={len(self.stations_by_type['a'])}, "
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

            self.get_logger().debug(f"Graph JSON parsed: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
            self.inject_stations_to_graph()
        except Exception as e:
            self.get_logger().error(f"Failed to parse graph JSON: {e}")

    def inject_stations_to_graph(self) -> None:
        self.full_graph, self.station_nodes = build_full_graph_with_stations(
            self.graph,
            self.stations,
            self.k_nearest,
            self.graph_nodes_map_coords,
        )

    def find_closest_node(
        self,
        x: float,
        y: float,
        in_full_graph: bool = True,
        threshold: Optional[float] = None,
    ) -> Optional[int]:
        best_node: Optional[int] = None
        best_dist = float("inf")
        max_dist = threshold if threshold is not None else self.node_match_threshold

        nodes = (
            self.full_graph.nodes(data=True)
            if in_full_graph
            else self.graph.nodes(data=True)
        )

        if not nodes:
            for n_id, coords in self.graph_nodes_map_coords.items():
                d = math.hypot(x - coords[0], y - coords[1])
                if d < best_dist and d <= max_dist:
                    best_dist = d
                    best_node = n_id
            return best_node

        for n_id, data in nodes:
            pos = data.get("pos")
            if not pos:
                continue
            d = math.hypot(x - pos[0], y - pos[1])
            if d < best_dist and d <= max_dist:
                best_dist = d
                best_node = n_id

        return best_node

    # --------------------------------------------------------------------- #
    # Robot pose and task reception
    # --------------------------------------------------------------------- #

    def get_robot_position(
        self, robot_id: int
    ) -> Optional[Tuple[float, float]]:
        frame = f"{self.robot_prefix}{robot_id}{self.robot_suffix}"
        try:
            t = self.tf_buffer.lookup_transform(
                self.global_frame, frame, rclpy.time.Time()
            )
            return (t.transform.translation.x, t.transform.translation.y)
        except Exception as e:
            self.get_logger().debug(
                f"TF lookup failed for {frame}: {e}",
                throttle_duration_sec=5.0,
            )
            return None

    def task_callback(self, msg: String) -> None:
        parts = [p.strip() for p in msg.data.split(",")]
        if len(parts) < 2:
            self.get_logger().debug(f"Bad task format: {msg.data}")
            return

        task_id = parts[0]
        stations_str = parts[1].split("|")
        priority = float(parts[2]) if len(parts) > 2 else 1.0
        
        # Check if parking stations exist and append "p" to the sequence if not already present
        has_p = "p" in self.stations_by_type and any(s.online for s in self.stations_by_type["p"])
        if has_p and stations_str[-1] != "p":
            stations_str.append("p")

        valid_tokens = set(self.stations.keys()) | set(
            self.stations_by_type.keys()
        )
        if any(t not in valid_tokens for t in stations_str):
            self.get_logger().debug(
                f"Task {task_id} references unknown station/type(s): {stations_str}"
            )
            return

        task = Task(
            task_id=task_id,
            timestamp=self.get_clock().now().nanoseconds / 1e9,
            stations=stations_str,
            priority=priority,
        )

        self.task_queue.append(task)
        self.get_logger().debug(
            f"Queued task {task_id} for sequence: {' -> '.join(stations_str)}"
        )

    # --------------------------------------------------------------------- #
    # Allocation
    # --------------------------------------------------------------------- #

    def allocate_task(
        self, task: Task
    ) -> Tuple[Optional[int], float, List[str]]:
        if not self.full_graph.nodes:
            self.get_logger().debug(
                "Skeleton graph is currently empty or uninitialized!",
                throttle_duration_sec=2.0,
            )
            return None, float("inf"), []

        groups: List[List[str]] = []
        for item in task.stations:
            if item in self.stations:
                groups.append([item])
            elif item in self.stations_by_type:
                groups.append(
                    [
                        s.name
                        for s in self.stations_by_type[item]
                        if s.online
                        and s.name not in self.occupied_stations
                        and s.name not in self.physical_occupancy
                    ]
                )
            else:
                self.get_logger().debug(f"Unknown station or type: {item}")
                return None, float("inf"), []

        if any(not g for g in groups):
            self.get_logger().debug(
                f"Task {task.task_id}: some station groups empty/offline "
                f"(tokens: {task.stations})"
            )
            return None, float("inf"), []

        max_combinations = 10000
        total_combinations = 1
        for g in groups:
            total_combinations *= len(g)
            if total_combinations > max_combinations:
                self.get_logger().debug(
                    f"Task {task.task_id}: too many station combinations "
                    f"({total_combinations}), skipping allocation."
                )
                return None, float("inf"), []

        all_stations = set(itertools.chain(*groups))

        W_R_S: Dict[int, Dict[str, float]] = {
            r: {} for r in range(1, self.num_robots + 1)
        }
        for r in range(1, self.num_robots + 1):
            task_info = self.robot_tasks[r]
            # Skip if robot is busy with a non-parking task
            if task_info is not None and not task_info.get("is_parking", False):
                continue

            pos = self.get_robot_position(r)
            if pos is None:
                self.get_logger().debug(f"No TF for robot_{r}; excluding it from allocation.")
                rn = None
            else:
                rn = self.find_closest_node(
                    pos[0], pos[1], threshold=float("inf")
                )
                if rn is None:
                    self.get_logger().debug(
                        f"No graph node near robot_{r} at {pos}; "
                        f"excluding from allocation."
                    )

            robot = self.robot_states[r]
            remaining = robot.battery_soc * robot.max_range_m

            for s_name in all_stations:
                if (
                    rn is None
                    or s_name in self.occupied_stations
                    or s_name in self.physical_occupancy
                ):
                    W_R_S[r][s_name] = float("inf")
                    continue

                s_node = self.station_nodes.get(s_name)
                if s_node is None:
                    W_R_S[r][s_name] = float("inf")
                    continue

                try:
                    dist = nx.shortest_path_length(
                        self.full_graph, rn, s_node, weight="weight"
                    )
                    d_cost = self.alpha_d * dist
                    b_cost = (
                        float("inf")
                        if remaining <= 0.0
                        else self.alpha_b * (dist / remaining)
                    )
                    u_cost = self.alpha_u * robot.usage_index
                    W_R_S[r][s_name] = d_cost + b_cost + u_cost
                except nx.NetworkXNoPath:
                    W_R_S[r][s_name] = float("inf")

        W_S_S: Dict[str, Dict[str, float]] = {}
        for s1 in all_stations:
            W_S_S[s1] = {}
            n1 = self.station_nodes.get(s1)
            for s2 in all_stations:
                n2 = self.station_nodes.get(s2)
                if s1 == s2:
                    W_S_S[s1][s2] = 0.0
                elif (
                    n1 is None
                    or n2 is None
                    or s2 in self.occupied_stations
                    or s2 in self.physical_occupancy
                ):
                    W_S_S[s1][s2] = float("inf")
                else:
                    try:
                        W_S_S[s1][s2] = nx.shortest_path_length(
                            self.full_graph, n1, n2, weight="weight"
                        )
                    except nx.NetworkXNoPath:
                        W_S_S[s1][s2] = float("inf")

        best_cost = float("inf")
        best_robot: Optional[int] = None
        best_path: List[str] = []

        for r in range(1, self.num_robots + 1):
            task_info = self.robot_tasks[r]
            if task_info is not None and not task_info.get("is_parking", False):
                continue
                
            if not W_R_S[r]:
                continue

            for path in itertools.product(*groups):
                if len(set(path)) != len(path):
                    continue

                cost = W_R_S[r][path[0]]
                for i in range(len(path) - 1):
                    cost += W_S_S[path[i]][path[i + 1]] * self.alpha_d

                if cost < best_cost:
                    best_cost = cost
                    best_robot = r
                    best_path = list(path)

        return best_robot, best_cost, best_path

    # --------------------------------------------------------------------- #
    # Execution and occupancy
    # --------------------------------------------------------------------- #

    def send_to_station(
        self, robot_id: int, station_name: str, log_dispatch: bool = True
    ) -> None:
        station = self.stations[station_name]
        msg = PoseStamped()
        msg.header.frame_id = self.global_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(station.position[0])
        msg.pose.position.y = float(station.position[1])
        msg.pose.orientation.w = 1.0

        self.goal_pubs[robot_id].publish(msg)
        if log_dispatch:
            self.get_logger().debug(
                f"Dispatched robot_{robot_id} to {station_name} via multi_chomp"
            )

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
            if d < 2.0 * self.robot_footprint_radius:
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
                d = math.hypot(
                    x - station.position[0], y - station.position[1]
                )
                if d < self.robot_footprint_radius:
                    self.physical_occupancy.add(s_name)

    def update_callback(self) -> None:
        now_sec = self.get_clock().now().nanoseconds / 1e9

        for r, task_info in self.robot_tasks.items():
            if not task_info:
                continue

            reserved_station = task_info["path"][task_info["current_idx"]]
            t_res = task_info.get("station_reservation_time", None)
            if t_res is None:
                continue

            if now_sec - t_res > self.station_reservation_timeout:
                self.get_logger().debug(
                    f"Soft-releasing station {reserved_station} "
                    f"for robot_{r} due to timeout"
                )
                self.occupied_stations.discard(reserved_station)
                task_info["station_reservation_time"] = now_sec

        self.update_physical_occupancy()
        self.update_collision_flags()

        for r in range(1, self.num_robots + 1):
            task_info = self.robot_tasks[r]
            if not task_info:
                continue

            path: List[str] = task_info["path"]
            curr_idx: int = task_info["current_idx"]
            if curr_idx >= len(path):
                continue

            curr_station_name = path[curr_idx]
            curr_station = self.stations[curr_station_name]

            r_pos = self.get_robot_position(r)
            if r_pos:
                dist = math.hypot(
                    r_pos[0] - curr_station.position[0],
                    r_pos[1] - curr_station.position[1],
                )
                if dist < self.station_reach_radius:
                    task_info["station_reservation_time"] = now_sec
                    if task_info["arrival_time"] is None:
                        task_info["arrival_time"] = now_sec
                    elif (
                        now_sec - task_info["arrival_time"]
                        >= self.station_dwell_time
                    ):
                        self.occupied_stations.discard(curr_station_name)
                        task_info["current_idx"] += 1
                        task_info["arrival_time"] = None

                        if task_info["current_idx"] < len(path):
                            next_station = path[task_info["current_idx"]]
                            self.occupied_stations.add(next_station)
                            self.send_to_station(
                                r, next_station, log_dispatch=True
                            )
                            
                            # Mark as parking if this is the last leg and it's a 'p' station
                            if task_info["current_idx"] == len(path) - 1 and self.stations[next_station].station_type == "p":
                                task_info["is_parking"] = True
                        else:
                            now_sec = self.get_clock().now().nanoseconds / 1e9
                            duration = now_sec - task_info["start_time"]
                            
                            self.get_logger().info(
                                f"Robot_{r} completed task "
                                f"{task_info['task_id']} "
                                f"in {duration:.2f}s"
                            )
                            
                            # CRITICAL FIX: Safely discard all stations in the path to prevent starvation
                            for s in path:
                                self.occupied_stations.discard(s)

                            collision_flag = 1 if task_info.get(
                                "had_collision", False
                            ) else 0
                            allocation_cost = task_info.get(
                                "allocation_cost", None
                            )

                            self.logger.log(
                                TaskLogRecord(
                                    timestamp=now_sec,
                                    run_id=self.run_id,
                                    task_id=task_info["task_id"],
                                    robot_id=f"robot_{r}",
                                    event="COMPLETED",
                                    status="OK",
                                    allocation_cost=allocation_cost,
                                    duration=duration,
                                    path="->".join(path),
                                    collision_flag=collision_flag,
                                    message="Task executed successfully",
                                )
                            )
                            self.robot_tasks[r] = None

        while self.task_queue:
            # Idle robots include those doing nothing AND those currently executing a preemptable parking leg
            idle_robots = [
                r for r in range(1, self.num_robots + 1)
                if self.robot_tasks[r] is None or self.robot_tasks[r].get("is_parking", False)
            ]
            
            if not idle_robots:
                break

            self.task_queue.sort(key=lambda x: x.priority, reverse=True)

            assigned_any = False
            now_sec = self.get_clock().now().nanoseconds / 1e9

            for idx, task in enumerate(self.task_queue):
                if now_sec - task.last_attempt_time < self.retry_cooldown_sec:
                    continue

                is_impossible = False
                valid_groups = []
                
                for item in task.stations:
                    if item in self.stations:
                        valid_groups.append([item])
                    elif item in self.stations_by_type:
                        online_stations = [s.name for s in self.stations_by_type[item] if s.online]
                        if not online_stations:
                            is_impossible = True
                            break
                        valid_groups.append(online_stations)
                    else:
                        is_impossible = True
                        break

                if is_impossible:
                    self.get_logger().warn(f"Dropping task {task.task_id}: Invalid or offline stations.")
                    self.task_queue.pop(idx)
                    break

                is_blocked = False
                for g in valid_groups:
                    if all(s in self.occupied_stations or s in self.physical_occupancy for s in g):
                        is_blocked = True
                        break


                if not is_blocked:
                    task.allocation_attempts += 1
                    task.last_attempt_time = now_sec

                if task.allocation_attempts > self.max_allocation_attempts:
                    self.get_logger().warn(f"Dropping task {task.task_id} after {task.allocation_attempts} failed route attempts.")
                    self.task_queue.pop(idx)
                    break

                robot_id, cost, path = self.allocate_task(task)
                if robot_id is None or math.isinf(cost) or not path:
                    continue

                self.task_queue.pop(idx)
                
                # Handle preemption if robot was parking
                if self.robot_tasks[robot_id] is not None:
                    old_task = self.robot_tasks[robot_id]
                    # Free remaining stations for the cancelled task
                    for s in old_task["path"][old_task["current_idx"]:]:
                        self.occupied_stations.discard(s)
                        
                    # Slightly adjust usage index down since it was preempted
                    self.robot_states[robot_id].usage_index = max(0.0, self.robot_states[robot_id].usage_index - 1.0)
                    
                    self.logger.log(
                        TaskLogRecord(
                            timestamp=now_sec,
                            run_id=self.run_id,
                            task_id=old_task["task_id"],
                            robot_id=f"robot_{robot_id}",
                            event="CANCELLED",
                            status="PREEMPTED",
                            allocation_cost=old_task.get("allocation_cost"),
                            duration=now_sec - old_task["start_time"],
                            path="->".join(old_task["path"]),
                            collision_flag=int(old_task.get("had_collision", False)),
                            message="Parking preempted by new task",
                        )
                    )
                
                self.get_logger().info(
                    f"Assigned task {task.task_id} to robot_{robot_id} "
                    f"(Cost: {cost:.2f}, Path: {' -> '.join(path)})"
                )

                now_sec = self.get_clock().now().nanoseconds / 1e9
                is_parking = (len(path) == 1 and self.stations[path[0]].station_type == "p")

                self.robot_tasks[robot_id] = {
                    "task_id": task.task_id,
                    "path": path,
                    "current_idx": 0,
                    "start_time": now_sec,
                    "arrival_time": None,
                    "station_reservation_time": now_sec,
                    "allocation_cost": cost,
                    "had_collision": False,
                    "is_parking": is_parking,
                }

                for s in path:
                    self.occupied_stations.add(s)

                self.robot_states[robot_id].usage_index += 1.0

                self.logger.log(
                    TaskLogRecord(
                        timestamp=now_sec,
                        run_id=self.run_id,
                        task_id=task.task_id,
                        robot_id=f"robot_{robot_id}",
                        event="ASSIGNED",
                        status="OK",
                        allocation_cost=cost,
                        duration=None,
                        path="->".join(path),
                        collision_flag=0,
                        message="Task assigned to robot",
                    )
                )

                self.send_to_station(robot_id, path[0], log_dispatch=True)
                assigned_any = True
                break

            if not assigned_any:
                self.get_logger().debug(
                    "No feasible assignment for any pending task with current "
                    "Graph/TF/occupancies.",
                    throttle_duration_sec=3.0,
                )
                break


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