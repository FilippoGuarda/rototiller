#!/usr/bin/env python3
"""
Extended Task Allocation Node with parallel BFS-MAS graph vs. Nav2 SMAC metric comparison.

This node runs your existing BFS-MAS graph-based allocator alongside a Nav2 SMAC metric-map
baseline, logging allocation time, robot–task pairs, and Hausdorff distance for every task.
Logs are written to /logs/<run_id>/ for later analysis.

Usage (example):
    ros2 run rototiller task_allocation_comparison_node.py \
        --ros-args --num_robots:=6 \
        --ros-args --map:=/path/to/map.yaml \
        --ros-args --graph:=/path/to/skeleton_graph.json \
        --ros-args --stations:=/path/to/stations.yaml \
        --ros-args --run_id:=comparison_run_01
"""

import csv
import heapq
import numpy as np

from nav_msgs.msg import OccupancyGrid

import math
import itertools
import os
import json
import time
from datetime import datetime
from pathlib import Path
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


    

class MetricPlannerNode:
    """
    Read-only metric planner based on robot1's global costmap.

    It does not command Nav2 or move any robot. It subscribes to the supplied
    OccupancyGrid topic and evaluates A* paths for allocation-cost comparison.
    """

    def __init__(
        self,
        node: Node,
        costmap_topic: str = "/robot1/global_costmap/costmap",
        global_frame: str = "map",
        obstacle_cost_threshold: int = 253,
        allow_unknown: bool = False,
    ):
        self.node = node
        self.global_frame = global_frame
        self.costmap_topic = costmap_topic
        self.obstacle_cost_threshold = obstacle_cost_threshold
        self.allow_unknown = allow_unknown

        self.grid: Optional[np.ndarray] = None
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.resolution = 0.0
        self.width = 0
        self.height = 0
        self.frame_id = ""
        self.last_update_time = 0.0
        self._last_allocation_status: Dict[str, Tuple[Optional[int], Optional[int]]] = {}

        qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        self.costmap_sub = node.create_subscription(
            OccupancyGrid,
            costmap_topic,
            self._costmap_callback,
            qos,
        )

        node.get_logger().info(
            f"Metric planner subscribed to costmap: {costmap_topic}"
        )

    @property
    def ready(self) -> bool:
        return (
            self.grid is not None
            and self.width > 0
            and self.height > 0
            and self.resolution > 0.0
        )

    def _costmap_callback(self, msg: OccupancyGrid) -> None:
        width = int(msg.info.width)
        height = int(msg.info.height)

        if width <= 0 or height <= 0 or len(msg.data) != width * height:
            self.node.get_logger().warning(
                f"Ignoring invalid costmap: {width}x{height}, "
                f"data length={len(msg.data)}"
            )
            return

        raw_grid = np.asarray(msg.data, dtype=np.int16).reshape((height, width))

        blocked = raw_grid >= self.obstacle_cost_threshold
        if not self.allow_unknown:
            blocked |= raw_grid < 0

        self.grid = blocked
        self.width = width
        self.height = height
        self.resolution = float(msg.info.resolution)
        self.origin_x = float(msg.info.origin.position.x)
        self.origin_y = float(msg.info.origin.position.y)
        self.frame_id = msg.header.frame_id or self.global_frame
        self.last_update_time = time.monotonic()

    def _in_bounds(self, cell: Tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def _world_to_grid(self, point: Tuple[float, float]) -> Tuple[int, int]:
        x, y = point
        gx = math.floor((x - self.origin_x) / self.resolution)
        gy = math.floor((y - self.origin_y) / self.resolution)
        return int(gx), int(gy)

    def _grid_to_world(self, cell: Tuple[int, int]) -> Tuple[float, float]:
        gx, gy = cell
        return (
            self.origin_x + (gx + 0.5) * self.resolution,
            self.origin_y + (gy + 0.5) * self.resolution,
        )

    def _nearest_free_cell(
        self,
        requested: Tuple[int, int],
        max_radius_cells: int = 20,
    ) -> Optional[Tuple[int, int]]:
        if self.grid is None:
            return None

        if self._in_bounds(requested) and not self.grid[requested[1], requested[0]]:
            return requested

        cx, cy = requested
        for radius in range(1, max_radius_cells + 1):
            min_x = max(0, cx - radius)
            max_x = min(self.width - 1, cx + radius)
            min_y = max(0, cy - radius)
            max_y = min(self.height - 1, cy + radius)

            for x in range(min_x, max_x + 1):
                for y in (min_y, max_y):
                    if not self.grid[y, x]:
                        return x, y

            for y in range(min_y + 1, max_y):
                for x in (min_x, max_x):
                    if not self.grid[y, x]:
                        return x, y

        return None

    @staticmethod
    def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def _neighbors(
        self,
        cell: Tuple[int, int],
    ) -> List[Tuple[Tuple[int, int], float]]:
        if self.grid is None:
            return []

        x, y = cell
        result = []

        for dx, dy in (
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ):
            nx, ny = x + dx, y + dy
            candidate = (nx, ny)

            if not self._in_bounds(candidate):
                continue
            if self.grid[ny, nx]:
                continue

            # Disallow diagonal corner-cutting through obstacles.
            if dx != 0 and dy != 0:
                if self.grid[y, nx] or self.grid[ny, x]:
                    continue
                cost = math.sqrt(2.0)
            else:
                cost = 1.0

            result.append((candidate, cost))

        return result

    def _astar(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        open_heap = []
        heapq.heappush(open_heap, (self._heuristic(start, goal), 0.0, start))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score = {start: 0.0}
        closed: Set[Tuple[int, int]] = set()

        while open_heap:
            _, current_cost, current = heapq.heappop(open_heap)

            if current in closed:
                continue
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            closed.add(current)

            for neighbor, step_cost in self._neighbors(current):
                if neighbor in closed:
                    continue

                tentative_cost = current_cost + step_cost
                if tentative_cost >= g_score.get(neighbor, float("inf")):
                    continue

                came_from[neighbor] = current
                g_score[neighbor] = tentative_cost
                priority = tentative_cost + self._heuristic(neighbor, goal)
                heapq.heappush(open_heap, (priority, tentative_cost, neighbor))

        return None

    def plan_path(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> Tuple[Optional[List[Tuple[float, float]]], float]:
        t0 = time.perf_counter()

        if not self.ready:
            return None, (time.perf_counter() - t0) * 1000.0

        requested_start = self._world_to_grid(start)
        requested_goal = self._world_to_grid(goal)

        max_snap_radius = max(1, int(1.0 / self.resolution))
        start_cell = self._nearest_free_cell(requested_start, max_snap_radius)
        goal_cell = self._nearest_free_cell(requested_goal, max_snap_radius)

        if start_cell is None or goal_cell is None:
            return None, (time.perf_counter() - t0) * 1000.0

        grid_path = self._astar(start_cell, goal_cell)
        if not grid_path:
            return None, (time.perf_counter() - t0) * 1000.0

        world_path = [self._grid_to_world(cell) for cell in grid_path]
        return world_path, (time.perf_counter() - t0) * 1000.0

    def shutdown(self) -> None:
        # The subscription is owned by the parent ROS node.
        pass

class TaskAllocationComparisonNode(Node):
    def __init__(self):
        super().__init__(
            "task_allocation_comparison_node",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        self.num_robots = int(self.get_parameter("num_robots").value)
        self.global_frame = str(self.get_parameter("global_frame").value)
        self.robot_prefix = str(self.get_parameter("robot_base_frame_prefix").value)
        self.robot_suffix = str(self.get_parameter("robot_base_frame_suffix").value)
        self.node_match_threshold = float(self.get_parameter("node_match_threshold_m").value)

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
        self.robot_footprint_radius = float(self.get_parameter("robot_defaults.footprint_radius").value)

        if not self.has_parameter("task_batch_size"):
            self.declare_parameter("task_batch_size", 6)
        self.task_batch_size = int(self.get_parameter("task_batch_size").value)

        # Don't attempt an allocation round until at least this many tasks are
        # queued (defaults to the fleet size, so a full batch can be matched
        # against the fleet at once instead of dispatching piecemeal).
        if not self.has_parameter("min_batch_size"):
            self.declare_parameter("min_batch_size", self.num_robots)
        self.min_batch_size = int(self.get_parameter("min_batch_size").value)

        # ...unless the oldest queued task has been waiting this long already,
        # in which case process whatever is queued rather than starving
        # stragglers (e.g. the last few tasks of a run) forever.
        if not self.has_parameter("max_batch_wait_sec"):
            self.declare_parameter("max_batch_wait_sec", 8.0)
        self.max_batch_wait_sec = float(self.get_parameter("max_batch_wait_sec").value)

        if not self.has_parameter("max_allocation_attempts"):
            self.declare_parameter("max_allocation_attempts", 40)
        self.max_allocation_attempts = int(self.get_parameter("max_allocation_attempts").value)

        if not self.has_parameter("retry_cooldown_sec"):
            self.declare_parameter("retry_cooldown_sec", 5.0)
        self.retry_cooldown_sec = float(self.get_parameter("retry_cooldown_sec").value)

        if not self.has_parameter("fleet_tf_ready_timeout_sec"):
            self.declare_parameter("fleet_tf_ready_timeout_sec", 10.0)
        self.fleet_tf_ready_timeout_sec = float(self.get_parameter("fleet_tf_ready_timeout_sec").value)

        self.station_dwell_time = 1.5
        self.station_reservation_timeout = 60.0

        # Logging setup
        base_log_dir = str(self.get_parameter("log_file_path").value)
        if not self.has_parameter("run_id"):
            self.declare_parameter("run_id", "default_run")
        self.run_id = str(self.get_parameter("run_id").value)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(base_log_dir) / f"{self.run_id}_{timestamp}"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Separate log files for graph vs. metric
        self.graph_log_path = self.log_dir / "allocation_graph.csv"
        self.metric_log_path = self.log_dir / "allocation_metric.csv"
        self.comparison_log_path = self.log_dir / "allocation_comparison.csv"

        self.graph_logger = TaskLogger(str(self.graph_log_path))
        self.metric_logger = TaskLogger(str(self.metric_log_path))

        # Comparison CSV header
        with open(self.comparison_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "run_id",
                "task_id",

                "graph_robot_id",
                "graph_station",
                "graph_allocation_time_ms",
                "graph_predicted_cost",

                "metric_robot_id",
                "metric_station",
                "metric_allocation_time_ms",
                "metric_predicted_cost",

                "cost_difference_metric_minus_graph",
                "hausdorff_distance_m",
                "graph_path",
                "metric_path",
                "status",
            ])

        self.reentrant_callback_group = ReentrantCallbackGroup()
        self.graph_callback_group = MutuallyExclusiveCallbackGroup()
        # Task ingestion and allocation both touch self.task_queue / self.robot_tasks /
        # self._last_allocation_status without locks, so they must never run concurrently
        # with each other or with themselves.
        self.allocation_callback_group = MutuallyExclusiveCallbackGroup()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.robot_states: Dict[int, RobotState] = {
            i: RobotState(robot_id=i, battery_soc=default_battery, max_range_m=default_max_range, usage_index=default_usage)
            for i in range(1, self.num_robots + 1)
        }

        self.stations: Dict[str, StationConfig] = {}
        self.stations_by_type: Dict[str, List[StationConfig]] = {"a": [], "b": [], "c": [], "p": []}
        self.load_station_config()

        parking_stations = sorted(s.name for s in self.stations_by_type["p"] if s.online)
        assert len(parking_stations) >= self.num_robots

        self.robot_parking_station: Dict[int, str] = {r: parking_stations[r - 1] for r in range(1, self.num_robots + 1)}

        self.task_queue: List[Task] = []
        self.robot_tasks: Dict[int, Optional[dict]] = {i: None for i in range(1, self.num_robots + 1)}
        self._last_allocation_status: Dict[str, Tuple[Optional[int], Optional[int]]] = {}

        self.occupied_stations: Set[str] = set()
        self.physical_occupancy: Dict[str, int] = {}

        self.graph = nx.Graph()
        self.full_graph = nx.Graph()
        self.graph_nodes_map_coords: Dict[int, Tuple[float, float]] = {}
        self.station_nodes: Dict[str, int] = {}

        self._fleet_tf_deadline = None
        self._tf_subset_robots = None

        qos_profile = QoSProfile(
            depth=10,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        self.graph_sub = self.create_subscription(
            String, "skeleton_graph_json", self.graph_callback, qos_profile, callback_group=self.graph_callback_group
        )

        self.task_sub = self.create_subscription(
            String, "/tasks", self.task_callback, 10, callback_group=self.allocation_callback_group
        )

        self.goal_pubs: Dict[int, rclpy.publisher.Publisher] = {}
        for robot_id in range(1, self.num_robots + 1):
            topic = f"/{self.robot_prefix}{robot_id}/spades_goal"
            self.goal_pubs[robot_id] = self.create_publisher(PoseStamped, topic, 10)

        self.create_timer(
            1.0 / self.update_rate_hz,
            self.update_callback,
            callback_group=self.allocation_callback_group,
        )

        if not self.has_parameter("metric_costmap_topic"):
            self.declare_parameter(
                "metric_costmap_topic",
                "/robot1/global_costmap/costmap",
            )

        if not self.has_parameter("metric_obstacle_cost_threshold"):
            self.declare_parameter("metric_obstacle_cost_threshold", 253)

        if not self.has_parameter("metric_allow_unknown"):
            self.declare_parameter("metric_allow_unknown", False)

        self.metric_planner = MetricPlannerNode(
            node=self,
            costmap_topic=str(self.get_parameter("metric_costmap_topic").value),
            global_frame=self.global_frame,
            obstacle_cost_threshold=int(
                self.get_parameter("metric_obstacle_cost_threshold").value
            ),
            allow_unknown=bool(self.get_parameter("metric_allow_unknown").value),
        )

        self.get_logger().info(
            f"Task Allocation Comparison Node initialized. Robots: {self.num_robots}, "
            f"Update rate: {self.update_rate_hz} Hz, Log dir: {self.log_dir}"
        )

        for r, ps in self.robot_parking_station.items():
            self.get_logger().info(f" Robot {r} -> fixed home parking: {ps}")

    def _graph_astar_distance(self, start_node: int, goal_node: int) -> Optional[float]:

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
            f"B={len(self.stations_by_type['b'])}, C={len(self.stations_by_type['c'])}, P={len(self.stations_by_type['p'])}"
        )

    def graph_callback(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            self.graph = nx.node_link_graph(data, edges="links")
            self.graph_nodes_map_coords.clear()
            for n, d in self.graph.nodes(data=True):
                if "pos" in d:
                    self.graph_nodes_map_coords[n] = (d["pos"][0], d["pos"][1])
            self.get_logger().debug(f"Graph JSON parsed: {self.graph.number_of_nodes()} nodes")
            self.inject_stations_to_graph()
        except Exception as e:
            self.get_logger().error(f"Failed to parse graph JSON: {e}")

    def inject_stations_to_graph(self) -> None:
        self.full_graph, self.station_nodes = build_full_graph_with_stations(
            self.graph, self.stations, self.k_nearest, self.graph_nodes_map_coords
        )

    def find_closest_node(
        self, x: float, y: float, in_full_graph: bool = True, threshold: Optional[float] = None
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

    def get_robot_position(self, robot_id: int) -> Optional[Tuple[float, float]]:
        frame = f"{self.robot_prefix}{robot_id}{self.robot_suffix}"
        try:
            t = self.tf_buffer.lookup_transform(self.global_frame, frame, rclpy.time.Time())
            return (t.transform.translation.x, t.transform.translation.y)
        except Exception:
            return None

    def _allocation_inputs_ready(self) -> bool:
        if self.graph.number_of_nodes() == 0 or not self.station_nodes:
            self._fleet_tf_deadline = None
            self._tf_subset_robots = None
            self.get_logger().warning("Waiting for skeleton graph before allocating tasks...", throttle_duration_sec=10.0)
            return False

        robots_without_tf = [r for r in range(1, self.num_robots + 1) if self.get_robot_position(r) is None]

        if not robots_without_tf:
            self._fleet_tf_deadline = None
            self._tf_subset_robots = None
            return True

        if len(robots_without_tf) == self.num_robots:
            self._fleet_tf_deadline = None
            self._tf_subset_robots = None
            self.get_logger().warning("Waiting for TF poses (no robot localized yet)...", throttle_duration_sec=10.0)
            return False

        now = self.get_clock().now().nanoseconds / 1e9
        missing = ", ".join(str(r) for r in robots_without_tf)

        if self._fleet_tf_deadline is None:
            self._fleet_tf_deadline = now + self.fleet_tf_ready_timeout_sec
            self.get_logger().warn(
                f"Robots without TF: {missing}. Proceeding with allocation for the localized fleet in "
                f"{self.fleet_tf_ready_timeout_sec:.1f}s unless their TF appears."
            )
            return False

        if now < self._fleet_tf_deadline:
            self.get_logger().warning(
                f"Waiting for TF poses of robots: {missing} ({self._fleet_tf_deadline - now:.1f}s of grace period left)...",
                throttle_duration_sec=5.0,
            )
            return False

        if self._tf_subset_robots is None:
            self._tf_subset_robots = set(robots_without_tf)
            self.get_logger().warn(
                f"Proceeding WITHOUT robots {missing} (no TF after {self.fleet_tf_ready_timeout_sec:.1f}s grace period). "
                f"They will join allocation as soon as their TF appears."
            )

        return True

    def allocate_task_batch_comparison(self, tasks: List[Task]) -> Set[str]:
        """Allocate tasks using both graph and metric methods, logging comparison data."""
        assigned_task_ids = set()
        sorted_tasks = sorted(tasks, key=lambda x: x.priority, reverse=True)
        now_sec = self.get_clock().now().nanoseconds / 1e9

        eligible_robots = [
            r for r in range(1, self.num_robots + 1)
            if self.robot_tasks[r] is None or self.robot_tasks[r].get("is_parking", False)
        ]

        if not eligible_robots:
            # No robot can take on new work right now; every task in this batch
            # is guaranteed PARTIAL, so don't burn a solve (or a CSV row) per
            # task on a foregone conclusion. Just note it once per task and
            # leave the whole batch queued for the next attempt.
            status = (None, None)
            for task in sorted_tasks:
                if self._last_allocation_status.get(task.task_id) != status:
                    self.get_logger().warning(
                        f"Withholding {task.task_id}: no robots available for allocation."
                    )
                    self._last_allocation_status[task.task_id] = status
            return assigned_task_ids

        for task in sorted_tasks:
            # --- Graph-based allocation ---
            t0_graph = time.perf_counter()
            graph_result = self._allocate_single_task_graph(task, eligible_robots)
            t1_graph = time.perf_counter()
            graph_time_ms = (t1_graph - t0_graph) * 1000.0

            # --- Metric-based allocation ---
            t0_metric = time.perf_counter()
            metric_result = self._allocate_single_task_metric(task, eligible_robots)
            t1_metric = time.perf_counter()
            metric_time_ms = (t1_metric - t0_metric) * 1000.0

            # --- Hausdorff distance between the two paths ---
            hausdorff = 0.0
            if graph_result["path"] and metric_result["path"]:
                path_graph = [self.stations[s].position for s in graph_result["path"]]
                path_metric = [self.stations[s].position for s in metric_result["path"]]
                hausdorff = self._hausdorff_distance(path_graph, path_metric)

            graph_ok = graph_result["robot_id"] is not None
            metric_ok = metric_result["robot_id"] is not None

            graph_cost = graph_result["cost"] if graph_ok else None
            metric_cost = metric_result["cost"] if metric_ok else None
            cost_difference = (
                metric_cost - graph_cost
                if graph_cost is not None and metric_cost is not None
                else None
            )

            graph_station = graph_result["path"][0] if graph_result["path"] else ""
            metric_station = metric_result["path"][0] if metric_result["path"] else ""

            with open(self.comparison_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    f"{now_sec:.3f}",
                    self.run_id,
                    task.task_id,
                    graph_result["robot_id"],
                    graph_station,
                    f"{graph_cost:.6f}" if graph_cost is not None else "",
                    f"{graph_time_ms:.3f}",
                    metric_result["robot_id"],
                    metric_station,
                    f"{metric_cost:.6f}" if metric_cost is not None else "",
                    f"{metric_time_ms:.3f}",
                    f"{cost_difference:.6f}" if cost_difference is not None else "",
                    f"{hausdorff:.3f}",
                    "->".join(graph_result["path"]),
                    "->".join(metric_result["path"]),
                    "OK" if graph_ok and metric_ok else "PARTIAL",
                ])


            status = (
                graph_result["robot_id"],
                metric_result["robot_id"],
            )

            if graph_result["robot_id"] is None or metric_result["robot_id"] is None:
                if self._last_allocation_status.get(task.task_id) != status:
                    self.get_logger().warning(
                        f"Withholding {task.task_id}: "
                        f"graph_robot={graph_result['robot_id']}, "
                        f"metric_robot={metric_result['robot_id']}. "
                        "No goal will be published."
                    )
                    self._last_allocation_status[task.task_id] = status
                continue

            self._last_allocation_status.pop(task.task_id, None)

            # Use graph result for actual dispatch (your production logic)
            r = graph_result["robot_id"]
            selected_path = graph_result["path"]
            cost = graph_result["cost"]

            is_parking = len(selected_path) == 1 and self.stations[selected_path[0]].station_type == "p"
            assigned_task_ids.add(task.task_id)

            if self.robot_tasks[r] is not None and self.robot_tasks[r].get("is_parking", False):
                old_task = self.robot_tasks[r]
                self.occupied_stations.discard(old_task["path"][0])
                self.robot_states[r].usage_index = max(0.0, self.robot_states[r].usage_index - 1.0)
                self.robot_tasks[r] = None

            if self.robot_tasks[r] is None:
                self.robot_tasks[r] = {
                    "task_id": task.task_id,
                    "path": selected_path,
                    "current_idx": 0,
                    "start_time": now_sec,
                    "arrival_time": None,
                    "station_reservation_time": now_sec,
                    "allocation_cost": cost,
                    "had_collision": False,
                    "is_parking": is_parking,
                }
                self.occupied_stations.add(selected_path[0])
                self.send_to_station(r, selected_path[0], log_dispatch=True)
                self.robot_states[r].usage_index += 1.0

                if not is_parking:
                    self.graph_logger.log(
                        TaskLogRecord(
                            timestamp=now_sec,
                            run_id=self.run_id,
                            task_id=task.task_id,
                            robot_id=f"robot_{r}",
                            event="ASSIGNED",
                            status="OK",
                            allocation_cost=cost,
                            duration=None,
                            path="->".join(selected_path),
                            collision_flag=0,
                            message="Task assigned (graph-based)",
                        )
                    )
                    self.metric_logger.log(
                        TaskLogRecord(
                            timestamp=now_sec,
                            run_id=self.run_id,
                            task_id=task.task_id,
                            robot_id=f"robot_{metric_result['robot_id']}",
                            event="ASSIGNED",
                            status="OK",
                            allocation_cost=metric_result["cost"],
                            duration=None,
                            path="->".join(metric_result["path"]),
                            collision_flag=0,
                            message="Task assigned (metric-based baseline)",
                        )
                    )

        return assigned_task_ids

    def _allocate_single_task_graph(self, task: Task, eligible_robots: List[int]) -> dict:
        """Graph-based single-task allocation (your existing logic, simplified for comparison)."""
        solver = pywraplp.Solver.CreateSolver("SCIP")
        if not solver:
            return {"robot_id": None, "path": [], "cost": 0.0}

        x = {}
        costs = {}
        station_usage_vars = {}
        robot_combinations = {}

        for r in eligible_robots:
            task_info = self.robot_tasks[r]
            if task_info is not None and not task_info.get("is_parking", False):
                last_station = task_info["path"][-1]
                tail_node = self.station_nodes.get(last_station)
            else:
                pos = self.get_robot_position(r)
                if pos:
                    best, best_dist = None, float("inf")
                    for nid, coords in self.graph_nodes_map_coords.items():
                        d = math.hypot(pos[0] - coords[0], pos[1] - coords[1])
                        if d < best_dist:
                            best_dist, best = d, nid
                    tail_node = best
                else:
                    tail_node = None

            if tail_node is None:
                continue

            groups = []
            for item in task.stations:
                if item == "p" or (item in self.stations and self.stations[item].station_type == "p"):
                    groups.append([self.robot_parking_station[r]])
                elif item in self.stations:
                    groups.append([item])
                elif item in self.stations_by_type:
                    available = [
                        s.name for s in self.stations_by_type[item]
                        if s.online and s.name not in self.occupied_stations and self.physical_occupancy.get(s.name) in (None, r)
                    ]
                    groups.append(available)

            if any(not g for g in groups):
                continue

            combinations = list(itertools.product(*groups))
            robot_combinations[r] = combinations

            robot = self.robot_states[r]
            remaining = robot.battery_soc * robot.max_range_m

            for c_idx, path in enumerate(combinations):
                parking_violation = any(
                    self.stations[sname].station_type == "p" and sname != self.robot_parking_station[r]
                    for sname in path
                )
                if parking_violation:
                    continue

                first_station = path[0]
                if first_station in self.occupied_stations or (
                    self.physical_occupancy.get(first_station) not in (None, r)
                ):
                    continue
                
                target_node = self.station_nodes.get(first_station)
                if tail_node not in self.full_graph or target_node not in self.full_graph:
                    continue

                dist = nx.shortest_path_length(self.full_graph, tail_node, target_node, weight="weight")
                d_cost = self.alpha_d * dist
                b_cost = float("inf") if remaining <= 0.0 else self.alpha_b * (dist / remaining)
                u_cost = self.alpha_u * robot.usage_index
                cost = d_cost + b_cost + u_cost

                dist = self._graph_astar_distance(tail_node, target_node)
                if dist is None:
                    continue  # no path on graph
                d_cost = self.alpha_d * dist
                b_cost = float("inf") if remaining <= 0.0 else self.alpha_b * (dist / remaining)
                u_cost = self.alpha_u * robot.usage_index
                cost = d_cost + b_cost + u_cost

                for i in range(len(path) - 1):
                    n1 = self.station_nodes[path[i]]
                    n2 = self.station_nodes[path[i + 1]]
                    seg = self._graph_astar_distance(n1, n2)
                    if seg is None:
                        break  # invalidate this candidate
                    cost += self.alpha_d * seg

                # If the inner loop breaks early due to a missing segment, skip this candidate:
                if seg is None:
                    continue

                var = solver.IntVar(0, 1, f"x_{r}_{c_idx}")
                x[(r, c_idx)] = var
                costs[(r, c_idx)] = cost

                if first_station not in station_usage_vars:
                    station_usage_vars[first_station] = []
                station_usage_vars[first_station].append(var)

        if not x:
            return {"robot_id": None, "path": [], "cost": 0.0}

        solver.Add(solver.Sum(x.values()) == 1)
        for sname, vars_using_station in station_usage_vars.items():
            if len(vars_using_station) > 1:
                solver.Add(solver.Sum(vars_using_station) <= 1)

        solver.Minimize(solver.Sum(var * costs[key] for key, var in x.items()))
        status = solver.Solve()

        if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            for (r, c_idx), var in x.items():
                if var.solution_value() > 0.5:
                    combinations = robot_combinations[r]
                    selected_path = list(combinations[c_idx])
                    return {"robot_id": r, "path": selected_path, "cost": costs[(r, c_idx)]}

        return {"robot_id": None, "path": [], "cost": 0.0}

    def _allocate_single_task_metric(
        self,
        task: Task,
        eligible_robots: List[int],
    ) -> dict:
        """Allocate one task using A* distances on robot1's current global costmap."""

        if self.metric_planner is None:
            self.get_logger().warning(
                "Metric allocation skipped: metric planner is not configured.",
                throttle_duration_sec=5.0,
            )
            return {"robot_id": None, "path": [], "cost": 0.0}

        if not self.metric_planner.ready:
            self.get_logger().warning(
                "Metric allocation skipped: no usable costmap has arrived on "
                f"{self.metric_planner.costmap_topic}.",
                throttle_duration_sec=5.0,
            )
            return {"robot_id": None, "path": [], "cost": 0.0}

        best_robot = None
        best_cost = float("inf")
        best_path = []

        for r in eligible_robots:
            task_info = self.robot_tasks[r]

            if task_info is not None and not task_info.get("is_parking", False):
                last_station = task_info["path"][-1]
                tail_pos = self.stations[last_station].position
            else:
                tail_pos = self.get_robot_position(r)

            if tail_pos is None:
                self.get_logger().debug(
                    f"Metric allocation: robot {r} skipped; no TF pose."
                )
                continue

            groups = []
            for item in task.stations:
                if item == "p" or (
                    item in self.stations
                    and self.stations[item].station_type == "p"
                ):
                    groups.append([self.robot_parking_station[r]])

                elif item in self.stations:
                    groups.append([item])

                elif item in self.stations_by_type:
                    available = [
                        station.name
                        for station in self.stations_by_type[item]
                        if station.online
                        and station.name not in self.occupied_stations
                        and self.physical_occupancy.get(station.name) in (None, r)
                    ]
                    groups.append(available)

                else:
                    self.get_logger().error(
                        f"Metric allocation: unknown station token '{item}'."
                    )
                    groups = []
                    break

            if not groups or any(not group for group in groups):
                continue

            robot = self.robot_states[r]
            remaining_range = robot.battery_soc * robot.max_range_m

            for candidate_path in itertools.product(*groups):
                candidate_path = list(candidate_path)

                if any(
                    self.stations[station_name].station_type == "p"
                    and station_name != self.robot_parking_station[r]
                    for station_name in candidate_path
                ):
                    continue

                first_station = candidate_path[0]
                if first_station in self.occupied_stations:
                    continue
                if self.physical_occupancy.get(first_station) not in (None, r):
                    continue

                total_distance = 0.0
                previous_position = tail_pos
                valid_candidate = True

                for station_name in candidate_path:
                    goal_position = self.stations[station_name].position
                    waypoints, planning_ms = self.metric_planner.plan_path(
                        previous_position,
                        goal_position,
                    )

                    if not waypoints:
                        valid_candidate = False
                        self.get_logger().debug(
                            f"Metric no-path: robot={r}, task={task.task_id}, "
                            f"from={previous_position}, to={goal_position}, "
                            f"station={station_name}, time={planning_ms:.2f} ms"
                        )
                        break

                    segment_distance = sum(
                        math.hypot(
                            waypoints[i][0] - waypoints[i - 1][0],
                            waypoints[i][1] - waypoints[i - 1][1],
                        )
                        for i in range(1, len(waypoints))
                    )
                    total_distance += segment_distance
                    previous_position = goal_position

                if not valid_candidate:
                    continue

                battery_cost = (
                    float("inf")
                    if remaining_range <= 0.0
                    else self.alpha_b * (total_distance / remaining_range)
                )
                cost = (
                    self.alpha_d * total_distance
                    + battery_cost
                    + self.alpha_u * robot.usage_index
                )

                if cost < best_cost:
                    best_cost = cost
                    best_robot = r
                    best_path = candidate_path

        return {
            "robot_id": best_robot,
            "path": best_path,
            "cost": best_cost if best_robot is not None else 0.0,
        }

    def _hausdorff_distance(self, path_a: List[Tuple[float, float]], path_b: List[Tuple[float, float]]) -> float:
        if not path_a or not path_b:
            return 0.0

        a_arr = np.array(path_a)
        b_arr = np.array(path_b)

        dists_a_to_b = np.min(np.linalg.norm(a_arr[:, None, :] - b_arr[None, :, :], axis=2), axis=1)
        dists_b_to_a = np.min(np.linalg.norm(b_arr[:, None, :] - a_arr[None, :, :], axis=2), axis=1)

        return float(max(np.max(dists_a_to_b), np.max(dists_b_to_a)))

    def send_to_station(self, robot_id: int, station_name: str, log_dispatch: bool = True) -> None:
        station = self.stations[station_name]
        msg = PoseStamped()
        msg.header.frame_id = self.global_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(station.position[0])
        msg.pose.position.y = float(station.position[1])
        msg.pose.orientation.w = 1.0
        self.goal_pubs[robot_id].publish(msg)

    def update_callback(self) -> None:
        if not self._allocation_inputs_ready():
            return

        if not self.task_queue:
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9

        # Wait for the queue to fill up to min_batch_size before spending a
        # solve on it, unless the oldest queued task has already been
        # waiting too long (so tail-end stragglers still get processed).
        oldest_pending_sec = min(t.timestamp for t in self.task_queue)
        queue_full_enough = len(self.task_queue) >= self.min_batch_size
        waited_too_long = (now_sec - oldest_pending_sec) >= self.max_batch_wait_sec
        if not queue_full_enough and not waited_too_long:
            return

        # Respect retry cooldown if last attempt was too recent
        if hasattr(self, "_last_alloc_time_sec"):
            if now_sec - self._last_alloc_time_sec < self.retry_cooldown_sec:
                return

        # Batch up to task_batch_size tasks
        batch = self.task_queue[: self.task_batch_size]
        remaining = self.task_queue[self.task_batch_size :]

        assigned = self.allocate_task_batch_comparison(batch)

        # Remove assigned tasks from queue
        self.task_queue = [t for t in remaining if t.task_id not in assigned]

        # Optionally re-queue unassigned tasks for next cycle
        unassigned = [t for t in batch if t.task_id not in assigned]
        if unassigned:
            self.task_queue = unassigned + self.task_queue

        self._last_alloc_time_sec = now_sec

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

        task = Task(task_id=task_id, timestamp=self.get_clock().now().nanoseconds / 1e9, stations=stations_str, priority=priority)
        self.task_queue.append(task)

    def main(self, args=None):
        rclpy.init(args=args)
        node = TaskAllocationComparisonNode()
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            if node.metric_planner:
                node.metric_planner.shutdown()
            node.destroy_node()
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = None
    executor = None

    try:
        node = TaskAllocationComparisonNode()
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if executor is not None:
            executor.shutdown()
        if node is not None:
            if node.metric_planner is not None:
                node.metric_planner.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()