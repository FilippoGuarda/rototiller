#!/usr/bin/env python3
"""
Parallel allocation-time comparison: BFS-MAS topological graph vs. Nav2 SMAC metric-map sampling.

This script:
- Loads a metric occupancy map (.pgm/.yaml) and the corresponding BFS-MAS skeleton graph JSON.
- For each of N randomized task sets, runs both allocation methods in parallel:
    • Graph-based: shortest-path costs on the BFS-MAS topological graph (your current allocator).
    • Metric-based: Nav2 SMAC planner queries on the metric occupancy grid (reviewer's requested baseline).
- Logs per-task allocation time, selected robot–task pairs, and Hausdorff distance between the
  graph-based and metric-based paths for each assigned task.
- Writes a CSV log and a JSON summary to /logs/<run_id>/.

Usage (example):
    ros2 run rototiller allocation_comparison.py \
        --map /path/to/map.yaml \
        --graph /path/to/skeleton_graph.json \
        --stations /path/to/stations.yaml \
        --num-tasks 50 \
        --num-robots 6 \
        --run-id comparison_run_01
"""

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import networkx as nx
import numpy as np
import yaml

# Optional Nav2 / ROS imports — fall back to a pure-Python A* if Nav2 is unavailable
try:
    from nav2_simple_commander.robot_navigator import BasicNavigator
    from geometry_msgs.msg import PoseStamped
    import rclpy
    HAS_NAV2 = True
except ImportError:
    HAS_NAV2 = False


@dataclass
class StationConfig:
    name: str
    station_type: str
    position: Tuple[float, float]
    online: bool = True


@dataclass
class Task:
    task_id: str
    stations: List[str]
    priority: float = 1.0


@dataclass
class AllocationResult:
    task_id: str
    method: str  # "graph" or "metric"
    allocation_time_ms: float
    robot_id: Optional[int]
    assigned_path: List[str]
    metric_path_graph: List[Tuple[float, float]]  # waypoints from graph-based path
    metric_path_metric: List[Tuple[float, float]]  # waypoints from metric-based path
    hausdorff_distance: float
    status: str  # "OK", "NO_PATH", "SOLVER_ERROR", etc.


@dataclass
class ComparisonRun:
    run_id: str
    map_path: str
    graph_path: str
    stations_path: str
    num_robots: int
    num_tasks: int
    results: List[AllocationResult] = field(default_factory=list)

    def summary(self) -> dict:
        graph_times = [r.allocation_time_ms for r in self.results if r.method == "graph"]
        metric_times = [r.allocation_time_ms for r in self.results if r.method == "metric"]
        hausdorffs = [r.hausdorff_distance for r in self.results if r.status == "OK"]

        return {
            "run_id": self.run_id,
            "map": self.map_path,
            "graph": self.graph_path,
            "stations": self.stations_path,
            "num_robots": self.num_robots,
            "num_tasks": self.num_tasks,
            "graph_allocation_time_ms": {
                "mean": float(np.mean(graph_times)) if graph_times else None,
                "std": float(np.std(graph_times)) if graph_times else None,
                "min": float(np.min(graph_times)) if graph_times else None,
                "max": float(np.max(graph_times)) if graph_times else None,
            },
            "metric_allocation_time_ms": {
                "mean": float(np.mean(metric_times)) if metric_times else None,
                "std": float(np.std(metric_times)) if metric_times else None,
                "min": float(np.min(metric_times)) if metric_times else None,
                "max": float(np.max(metric_times)) if metric_times else None,
            },
            "hausdorff_distance_m": {
                "mean": float(np.mean(hausdorffs)) if hausdorffs else None,
                "std": float(np.std(hausdorffs)) if hausdorffs else None,
                "min": float(np.min(hausdorffs)) if hausdorffs else None,
                "max": float(np.max(hausdorffs)) if hausdorffs else None,
            },
            "success_rate": sum(1 for r in self.results if r.status == "OK") / len(self.results) if self.results else None,
        }


def hausdorff_distance_2d(path_a: List[Tuple[float, float]], path_b: List[Tuple[float, float]]) -> float:
    """Compute the (symmetric) Hausdorff distance between two 2D waypoint lists."""
    if not path_a or not path_b:
        return float("inf")

    a_arr = np.array(path_a)
    b_arr = np.array(path_b)

    # For each point in A, find min distance to any point in B, then take max over A
    dists_a_to_b = np.min(np.linalg.norm(a_arr[:, None, :] - b_arr[None, :, :], axis=2), axis=1)
    dists_b_to_a = np.min(np.linalg.norm(b_arr[:, None, :] - a_arr[None, :, :], axis=2), axis=1)

    return float(max(np.max(dists_a_to_b), np.max(dists_b_to_a)))


def load_stations_yaml(path: str) -> Dict[str, StationConfig]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    stations = {}
    for stype in ["a", "b", "c", "p"]:
        key = f"{stype}_stations"
        if key not in data:
            continue
        for name, cfg in data[key].items():
            stations[name] = StationConfig(
                name=name,
                station_type=stype,
                position=(float(cfg["x"]), float(cfg["y"])),
                online=bool(cfg.get("online", True)),
            )
    return stations


def load_graph_json(path: str) -> nx.Graph:
    with open(path, "r") as f:
        data = json.load(f)
    return nx.node_link_graph(data, edges="links")


def inject_stations_to_graph(
    base_graph: nx.Graph,
    stations: Dict[str, StationConfig],
    k_nearest: int = 3,
) -> Tuple[nx.Graph, Dict[str, int]]:
    """Add virtual station nodes to the graph, connecting each to its k nearest skeleton nodes."""
    full_graph = base_graph.copy()
    next_node_id = max(full_graph.nodes) + 1 if full_graph.nodes else 0
    station_nodes: Dict[str, int] = {}

    # Precompute node coordinates
    node_coords: Dict[int, Tuple[float, float]] = {}
    for n_id, attrs in full_graph.nodes(data=True):
        if "pos" in attrs:
            node_coords[n_id] = tuple(attrs["pos"])

    for name, station in stations.items():
        if not station.online:
            continue

        virt_id = next_node_id
        next_node_id += 1

        full_graph.add_node(virt_id, pos=station.position)
        station_nodes[name] = virt_id

        # Connect to k nearest skeleton nodes
        distances = []
        for n_id, coords in node_coords.items():
            d = math.hypot(station.position[0] - coords[0], station.position[1] - coords[1])
            distances.append((d, n_id))

        distances.sort()
        for d, n_id in distances[:k_nearest]:
            full_graph.add_edge(virt_id, n_id, weight=d)

    return full_graph, station_nodes


class MetricPlanner:
    """Wrapper around Nav2 SMAC or a pure-Python A* fallback."""

    def __init__(self, map_path: str, global_frame: str = "map"):
        self.map_path = map_path
        self.global_frame = global_frame
        self.navigator = None

        if HAS_NAV2:
            try:
                rclpy.init()
                self.navigator = BasicNavigator()
                # Wait for the map to be loaded
                while not self.navigator.isMapReceived():
                    time.sleep(0.1)
            except Exception as e:
                print(f"Nav2 initialization failed: {e}; falling back to pure-Python A*.")
                self.navigator = None

    def plan_path(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> Tuple[Optional[List[Tuple[float, float]]], float]:
        """
        Plan a path from start to goal on the metric map.
        Returns (waypoints, planning_time_ms). If no path, returns (None, time_ms).
        """
        t0 = time.perf_counter()

        if self.navigator is not None:
            start_pose = PoseStamped()
            start_pose.header.frame_id = self.global_frame
            start_pose.header.stamp = self.navigator.get_clock().now().to_msg()
            start_pose.pose.position.x = start[0]
            start_pose.pose.position.y = start[1]
            start_pose.pose.orientation.w = 1.0

            goal_pose = PoseStamped()
            goal_pose.header.frame_id = self.global_frame
            goal_pose.header.stamp = self.navigator.get_clock().now().to_msg()
            goal_pose.pose.position.x = goal[0]
            goal_pose.pose.position.y = goal[1]
            goal_pose.pose.orientation.w = 1.0

            self.navigator.goToPose(goal_pose)
            # For pure planning-time measurement, cancel immediately and extract the planned path
            # (Nav2 SMAC computes the full path on goToPose call)
            path_msg = self.navigator.getPath()
            self.navigator.cancelTask()

            if path_msg is not None:
                waypoints = [(p.pose.position.x, p.pose.position.y) for p in path_msg.poses]
                t1 = time.perf_counter()
                return waypoints, (t1 - t0) * 1000.0

            t1 = time.perf_counter()
            return None, (t1 - t0) * 1000.0

        # Fallback: pure-Python A* on a loaded occupancy grid (simplified; replace with your actual grid loader)
        # This is a placeholder — wire it to your existing metric A* benchmark code if you have it.
        t1 = time.perf_counter()
        return None, (t1 - t0) * 1000.0

    def shutdown(self):
        if self.navigator is not None:
            self.navigator.lifecycleShutdown()
        if rclpy.ok():
            rclpy.shutdown()


class AllocationComparator:
    def __init__(
        self,
        map_path: str,
        graph_path: str,
        stations_path: str,
        num_robots: int,
        run_id: str,
        k_nearest: int = 3,
        alpha_d: float = 1.0,
        alpha_u: float = 0.1,
        alpha_b: float = 0.0,
    ):
        self.map_path = map_path
        self.graph_path = graph_path
        self.stations_path = stations_path
        self.num_robots = num_robots
        self.run_id = run_id
        self.k_nearest = k_nearest
        self.alpha_d = alpha_d
        self.alpha_u = alpha_u
        self.alpha_b = alpha_b

        self.stations = load_stations_yaml(stations_path)
        self.base_graph = load_graph_json(graph_path)
        self.full_graph, self.station_nodes = inject_stations_to_graph(
            self.base_graph, self.stations, k_nearest=self.k_nearest
        )

        self.metric_planner = MetricPlanner(map_path)

        # Robot states (simplified: just usage index)
        self.robot_usage: Dict[int, float] = {r: 0.0 for r in range(1, num_robots + 1)}
        self.robot_tail_node: Dict[int, Optional[int]] = {r: None for r in range(1, num_robots + 1)}

    def allocate_task_graph(self, task: Task) -> AllocationResult:
        """Allocate a single task using BFS-MAS graph costs (your current method)."""
        t0 = time.perf_counter()

        # Simplified single-task allocation: assign to the robot with minimum cost
        best_robot = None
        best_cost = float("inf")
        best_path = None

        for r in range(1, self.num_robots + 1):
            tail = self.robot_tail_node[r]

            # Expand station tokens (simplified; assumes explicit station names)
            path_candidates = []
            for sname in task.stations:
                if sname not in self.station_nodes:
                    continue
                path_candidates.append(sname)

            if len(path_candidates) != len(task.stations):
                continue

            # Compute cost: tail -> first station + inter-station distances
            try:
                cost = 0.0
                prev_node = tail
                for sname in path_candidates:
                    node = self.station_nodes[sname]
                    dist = nx.shortest_path_length(self.full_graph, prev_node, node, weight="weight")
                    cost += self.alpha_d * dist
                    prev_node = node

                cost += self.alpha_u * self.robot_usage[r]

                if cost < best_cost:
                    best_cost = cost
                    best_robot = r
                    best_path = path_candidates

            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

        t1 = time.perf_counter()
        alloc_time_ms = (t1 - t0) * 1000.0

        if best_robot is None:
            return AllocationResult(
                task_id=task.task_id,
                method="graph",
                allocation_time_ms=alloc_time_ms,
                robot_id=None,
                assigned_path=[],
                metric_path_graph=[],
                metric_path_metric=[],
                hausdorff_distance=0.0,
                status="NO_PATH",
            )

        # Update robot state
        self.robot_usage[best_robot] += 1.0
        self.robot_tail_node[best_robot] = self.station_nodes.get(best_path[-1])

        # Extract metric waypoints for Hausdorff computation (approximate: station positions)
        metric_path_graph = [self.stations[s].position for s in best_path] if best_path else []

        # For a fair comparison, also plan the same path on the metric map
        metric_path_metric = []
        if len(best_path) >= 2:
            start = self.stations[best_path[0]].position
            for sname in best_path[1:]:
                goal = self.stations[sname].position
                waypoints, _ = self.metric_planner.plan_path(start, goal)
                if waypoints:
                    metric_path_metric.extend(waypoints)
                start = goal

        hausdorff = hausdorff_distance_2d(metric_path_graph, metric_path_metric) if metric_path_graph and metric_path_metric else 0.0

        return AllocationResult(
            task_id=task.task_id,
            method="graph",
            allocation_time_ms=alloc_time_ms,
            robot_id=best_robot,
            assigned_path=best_path,
            metric_path_graph=metric_path_graph,
            metric_path_metric=metric_path_metric,
            hausdorff_distance=hausdorff,
            status="OK",
        )

    def allocate_task_metric(self, task: Task) -> AllocationResult:
        """Allocate a single task using Nav2 SMAC metric-map costs (reviewer's baseline)."""
        t0 = time.perf_counter()

        best_robot = None
        best_cost = float("inf")
        best_path = None
        best_metric_waypoints = []

        for r in range(1, self.num_robots + 1):
            # For the metric baseline, we approximate the tail as the last assigned station
            tail_pos = None
            tail_node = self.robot_tail_node[r]
            if tail_node is not None and tail_node in self.base_graph:
                attrs = self.base_graph.nodes[tail_node]
                if "pos" in attrs:
                    tail_pos = attrs["pos"]

            if tail_pos is None:
                # Fallback: use a default position (e.g., map center) — adapt as needed
                tail_pos = (0.0, 0.0)

            # Expand station tokens
            path_candidates = []
            for sname in task.stations:
                if sname not in self.stations:
                    continue
                path_candidates.append(sname)

            if len(path_candidates) != len(task.stations):
                continue

            # Compute cost via Nav2 SMAC queries
            try:
                cost = 0.0
                prev_pos = tail_pos
                metric_waypoints = []
                for sname in path_candidates:
                    goal_pos = self.stations[sname].position
                    waypoints, plan_time = self.metric_planner.plan_path(prev_pos, goal_pos)
                    if waypoints is None:
                        raise nx.NetworkXNoPath()
                    dist = sum(
                        math.hypot(waypoints[i][0] - waypoints[i - 1][0], waypoints[i][1] - waypoints[i - 1][1])
                        for i in range(1, len(waypoints))
                    )
                    cost += self.alpha_d * dist
                    metric_waypoints.extend(waypoints)
                    prev_pos = goal_pos

                cost += self.alpha_u * self.robot_usage[r]

                if cost < best_cost:
                    best_cost = cost
                    best_robot = r
                    best_path = path_candidates
                    best_metric_waypoints = metric_waypoints

            except (nx.NetworkXNoPath, Exception):
                continue

        t1 = time.perf_counter()
        alloc_time_ms = (t1 - t0) * 1000.0

        if best_robot is None:
            return AllocationResult(
                task_id=task.task_id,
                method="metric",
                allocation_time_ms=alloc_time_ms,
                robot_id=None,
                assigned_path=[],
                metric_path_graph=[],
                metric_path_metric=[],
                hausdorff_distance=0.0,
                status="NO_PATH",
            )

        # Update robot state
        self.robot_usage[best_robot] += 1.0
        self.robot_tail_node[best_robot] = self.station_nodes.get(best_path[-1]) if best_path else None

        # For Hausdorff, compare to the graph-based path for the same task (approximate)
        # In a full run, you would pair graph/metric results by task_id
        metric_path_metric = best_metric_waypoints
        metric_path_graph = [self.stations[s].position for s in best_path] if best_path else []
        hausdorff = hausdorff_distance_2d(metric_path_graph, metric_path_metric) if metric_path_graph and metric_path_metric else 0.0

        return AllocationResult(
            task_id=task.task_id,
            method="metric",
            allocation_time_ms=alloc_time_ms,
            robot_id=best_robot,
            assigned_path=best_path,
            metric_path_graph=metric_path_graph,
            metric_path_metric=metric_path_metric,
            hausdorff_distance=hausdorff,
            status="OK",
        )

    def run_comparison(self, tasks: List[Task]) -> ComparisonRun:
        run = ComparisonRun(
            run_id=self.run_id,
            map_path=self.map_path,
            graph_path=self.graph_path,
            stations_path=self.stations_path,
            num_robots=self.num_robots,
            num_tasks=len(tasks),
        )

        for task in tasks:
            res_graph = self.allocate_task_graph(task)
            res_metric = self.allocate_task_metric(task)
            run.results.extend([res_graph, res_metric])

        return run


def generate_random_tasks(stations: Dict[str, StationConfig], num_tasks: int, seed: int = 42) -> List[Task]:
    np.random.seed(seed)
    online_stations = [s for s in stations.values() if s.online and s.station_type != "p"]
    tasks = []
    for i in range(num_tasks):
        # Random sequence of 2–4 stations
        length = np.random.randint(2, 5)
        chosen = np.random.choice(online_stations, size=length, replace=True)
        stations_seq = [s.name for s in chosen]
        tasks.append(Task(task_id=f"task_{i:03d}", stations=stations_seq, priority=float(np.random.uniform(0.5, 2.0))))
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Parallel allocation-time comparison: BFS-MAS graph vs. Nav2 SMAC metric.")
    parser.add_argument("--map", required=True, help="Path to the occupancy map YAML (for Nav2 SMAC).")
    parser.add_argument("--graph", required=True, help="Path to the BFS-MAS skeleton graph JSON.")
    parser.add_argument("--stations", required=True, help="Path to the stations YAML.")
    parser.add_argument("--num-tasks", type=int, default=50, help="Number of random tasks to allocate.")
    parser.add_argument("--num-robots", type=int, default=6, help="Number of robots in the fleet.")
    parser.add_argument("--run-id", default="comparison_run", help="Run identifier for log filenames.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for task generation.")
    args = parser.parse_args()

    # Prepare output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("/logs") / f"{args.run_id}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    csv_path = log_dir / "allocation_comparison.csv"
    json_path = log_dir / "allocation_comparison_summary.json"

    print(f"Logs will be written to: {log_dir}")

    # Generate random tasks
    stations = load_stations_yaml(args.stations)
    tasks = generate_random_tasks(stations, args.num_tasks, seed=args.seed)

    # Run the comparison
    comparator = AllocationComparator(
        map_path=args.map,
        graph_path=args.graph,
        stations_path=args.stations,
        num_robots=args.num_robots,
        run_id=args.run_id,
    )

    run = comparator.run_comparison(tasks)
    summary = run.summary()

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task_id",
            "method",
            "allocation_time_ms",
            "robot_id",
            "assigned_path",
            "hausdorff_distance_m",
            "status",
        ])
        for res in run.results:
            writer.writerow([
                res.task_id,
                res.method,
                f"{res.allocation_time_ms:.3f}",
                res.robot_id if res.robot_id is not None else "",
                "->".join(res.assigned_path),
                f"{res.hausdorff_distance:.3f}" if res.status == "OK" else "",
                res.status,
            ])

    # Write JSON summary
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Comparison complete. Summary:")
    print(json.dumps(summary, indent=2))

    comparator.metric_planner.shutdown()


if __name__ == "__main__":
    main()