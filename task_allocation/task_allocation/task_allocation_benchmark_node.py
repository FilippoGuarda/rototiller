#!/usr/bin/env python3
import csv
import heapq
import itertools
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from scipy import ndimage
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String
import json

from task_allocation_helpers import build_full_graph_with_stations


class CostmapAStar:
    """Grid A* on a Nav2 costmap published as nav_msgs/OccupancyGrid.

    Nav2 translates its internal 0-255 costs into the 0-100 OccupancyGrid range
    (see Costmap2DPublisher): 0 = free, 1..98 = inflated (traversable), 99 = inscribed
    (robot centre here is in collision), 100 = lethal, -1 = unknown.
    ``threshold`` is therefore expressed in *OccupancyGrid units*; the default of 99
    blocks every cell where the robot cannot be, and lets it pass through the inflation
    gradient. (The raw-costmap value 253 can never be reached on this scale, which
    silently disabled all obstacle checking.)
    """

    OCC_MAX = 100          # highest valid OccupancyGrid value
    OCC_INSCRIBED = 99     # Nav2 INSCRIBED_INFLATED_OBSTACLE after translation

    def __init__(self, node, topic, threshold=OCC_INSCRIBED, allow_unknown=False):
        self.node = node
        self.topic = topic
        if threshold > self.OCC_MAX:
            node.get_logger().warn(
                f"metric_obstacle_cost_threshold={threshold} is outside the OccupancyGrid "
                f"range (0-{self.OCC_MAX}) and would never block any cell; "
                f"using {self.OCC_INSCRIBED} (Nav2 inscribed cost) instead."
            )
            threshold = self.OCC_INSCRIBED
        self.threshold = int(threshold)
        self.allow_unknown = allow_unknown
        self.grid = None            # bool [h, w], True = robot cannot be in this cell
        self.labels = None          # int32 [h, w], 4-connected component id of free cells (0 = blocked)
        self._main_label = 0        # id of the largest free component
        self._free_cache = None
        self.width = self.height = 0
        self.resolution = 0.0
        self.origin = (0.0, 0.0)
        qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.subscription = node.create_subscription(
            OccupancyGrid, topic, self._callback, qos
        )

    def _callback(self, msg):
        w, h = int(msg.info.width), int(msg.info.height)
        if w <= 0 or h <= 0 or len(msg.data) != w * h:
            return
        raw = np.asarray(msg.data, dtype=np.int16).reshape((h, w))
        blocked = raw >= self.threshold
        if not self.allow_unknown:
            blocked |= raw < 0
        resolution = float(msg.info.resolution)
        origin = (
            float(msg.info.origin.position.x),
            float(msg.info.origin.position.y),
        )
        # Nav2 republishes the costmap periodically; skip the relabelling if nothing changed.
        if (
            self.grid is not None
            and self.grid.shape == blocked.shape
            and self.resolution == resolution
            and self.origin == origin
            and np.array_equal(self.grid, blocked)
        ):
            return
        # Label free space with 4-connectivity. This is exactly the reachability implied by
        # _neighbors(): a diagonal move needs both orthogonal cells free, so it never joins
        # regions that 4-connected moves would not already join.
        labels, count = ndimage.label(~blocked)
        if count > 0:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            self._main_label = int(np.argmax(sizes))
        else:
            self._main_label = 0
            self.node.get_logger().warn("Costmap contains no traversable cells")
        self.labels = labels
        self._free_cache = None
        self.grid = blocked
        self.width, self.height = w, h
        self.resolution = resolution
        self.origin = origin

    @property
    def ready(self):
        return self.grid is not None and self.width > 0 and self.height > 0 and self.resolution > 0

    def world_to_grid(self, p):
        return (
            int(math.floor((p[0] - self.origin[0]) / self.resolution)),
            int(math.floor((p[1] - self.origin[1]) / self.resolution)),
        )

    def grid_to_world(self, c):
        return (
            self.origin[0] + (c[0] + 0.5) * self.resolution,
            self.origin[1] + (c[1] + 0.5) * self.resolution,
        )

    def free_cells(self):
        """Cells the robot can occupy AND that belong to the main connected region,
        so every returned pair of cells is guaranteed to have a path between them."""
        if self._free_cache is None:
            ys, xs = np.where(self.labels == self._main_label) if self._main_label else ([], [])
            self._free_cache = [(int(x), int(y)) for x, y in zip(xs, ys)]
        return self._free_cache

    def _neighbors(self, c):
        x, y = c
        for dx, dy, step in (
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)), (-1, -1, math.sqrt(2.0)),
        ):
            nx_, ny_ = x + dx, y + dy
            if not (0 <= nx_ < self.width and 0 <= ny_ < self.height):
                continue
            if self.grid[ny_, nx_]:
                continue
            if dx and dy and (self.grid[y, nx_] or self.grid[ny_, x]):
                continue
            yield (nx_, ny_), step

    def path(self, start, goal):
        if not self.ready:
            return None
        s, g = self.world_to_grid(start), self.world_to_grid(goal)
        if not (0 <= s[0] < self.width and 0 <= s[1] < self.height):
            return None
        if not (0 <= g[0] < self.width and 0 <= g[1] < self.height):
            return None
        if self.grid[s[1], s[0]] or self.grid[g[1], g[0]]:
            return None
        if self.labels[s[1], s[0]] != self.labels[g[1], g[0]]:
            return None  # different free regions: no path exists, skip the full A* expansion
        q = [(math.hypot(g[0] - s[0], g[1] - s[1]), 0.0, s)]
        parent, best = {}, {s: 0.0}
        while q:
            _, cost, cur = heapq.heappop(q)
            if cur == g:
                out = [cur]
                while cur in parent:
                    cur = parent[cur]
                    out.append(cur)
                return [self.grid_to_world(c) for c in reversed(out)]
            for nxt, step in self._neighbors(cur):
                nc = cost + step
                if nc < best.get(nxt, float("inf")):
                    best[nxt] = nc
                    parent[nxt] = cur
                    heapq.heappush(q, (nc + math.hypot(g[0] - nxt[0], g[1] - nxt[1]), nc, nxt))
        return None


class BenchmarkNode(Node):
    def __init__(self):
        super().__init__("task_allocation_benchmark_node", allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)
        self.n = int(self._param("num_robots", 6))
        self.batches = int(self._param("benchmark_batches", 100))
        self.seed = int(self._param("benchmark_seed", 15))
        self.alpha_d = float(self._param("alpha_distance", 1.0))
        self.alpha_b = float(self._param("alpha_battery", 0.0))
        self.alpha_u = float(self._param("alpha_usage", 0.0))
        self.resolution = 0.0
        self.rng = np.random.default_rng(self.seed)
        topic = str(self._param("metric_costmap_topic", "/robot1/global_costmap/costmap"))
        self.metric = CostmapAStar(self, topic, int(self._param("metric_obstacle_cost_threshold", CostmapAStar.OCC_INSCRIBED)),
                                   bool(self._param("metric_allow_unknown", False)))
        self.graph = nx.Graph()
        self.coords = {}
        self.full_graph = nx.Graph()
        self.station_nodes = {}
        self.graph_ready = False

        qos = QoSProfile(
            depth=10,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.graph_sub = self.create_subscription(
            String, "skeleton_graph_json", self._graph_callback, qos
        )

        self._load_stations()
        base = Path(str(self._param("log_file_path", "/tmp/task_allocation_benchmark")))
        run = str(self._param("run_id", "benchmark"))
        self.log_dir = base / f"{run}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.comparison_path = self.log_dir / "allocation_comparison.csv"
        self.batch_path = self.log_dir / "batch_statistics.csv"
        self._write_headers()
        self.timer = self.create_timer(1.0, self._run_once)
        self.batch_index = 0

    def _param(self, name, default=None):
        if not self.has_parameter(name):
            self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def _load_stations(self):
        self.stations = {}
        self.stations_by_type = {"a": [], "b": [], "c": [], "p": []}
        for typ in self.stations_by_type:
            names = list(self._param(f"stations.{typ}.names", []))
            for name in names:
                x = float(self._param(f"stations.{typ}.{name}.x", 0.0))
                y = float(self._param(f"stations.{typ}.{name}.y", 0.0))
                online = bool(self._param(f"stations.{typ}.{name}.online", True))
                station = (name, typ, (x, y), online)
                self.stations[name] = station
                self.stations_by_type[typ].append(station)

    def _write_headers(self):
        with open(self.comparison_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp", "run_id", "batch_id", "task_id",
                "graph_robot_id", "graph_station", "graph_predicted_cost", "graph_allocation_time_ms",
                "metric_robot_id", "metric_station", "metric_predicted_cost", "metric_allocation_time_ms",
                "cost_difference_metric_minus_graph", "hausdorff_distance_m",
                "status",
            ])
        with open(self.batch_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp", "run_id", "batch_id", "batch_size",
                "robot_position_weighted_variance", "task_position_weighted_variance",
                "graph_successes", "metric_successes",
                "graph_time_ms", "metric_time_ms",
                "cost_diff_avg", "cost_diff_min", "cost_diff_max",
                "hausdorff_avg", "hausdorff_min", "hausdorff_max",
            ])

    def _graph_callback(self, msg):
        try:
            data = json.loads(msg.data)
            self.graph = nx.node_link_graph(data, edges="links")
            self.coords.clear()
            for n, d in self.graph.nodes(data=True):
                if "pos" in d:
                    self.coords[n] = (d["pos"][0], d["pos"][1])
            self._inject_stations()
            self.graph_ready = True
        except Exception as e:
            self.get_logger().error(f"Failed to parse skeleton graph: {e}")

    def _inject_stations(self):
        if not self.coords:
            return

        from task_allocation_helpers import StationConfig, build_full_graph_with_stations

        stations_cfg = {}
        stations_by_type = {"a": [], "b": [], "c": [], "p": []}

        for name, (sname, stype, pos, online) in self.stations.items():
            station = StationConfig(sname, stype, pos, online)
            stations_cfg[name] = station
            stations_by_type[stype].append(station)

        k = 3
        self.full_graph, self.station_nodes = build_full_graph_with_stations(
            self.graph, stations_cfg, k, self.coords
        )

    def _variance(self, points, weights=None):
        p = np.asarray(points, dtype=float)
        if len(p) == 0:
            return ""
        w = np.ones(len(p)) if weights is None else np.asarray(weights, dtype=float)
        center = np.average(p, axis=0, weights=w)
        return float(np.sum(w * np.sum((p - center) ** 2, axis=1)) / np.sum(w))

    def _sample_positions(self):
        cells = self.metric.free_cells()
        if len(cells) < 2 * self.n:
            raise RuntimeError(f"Costmap has fewer than {2 * self.n} reachable admissible cells")
        selected = self.rng.choice(len(cells), size=2 * self.n, replace=False)
        points = [self.metric.grid_to_world(cells[i]) for i in selected]
        return points[:self.n], points[self.n:]

    def _task_station(self, index, position):
        name = f"batch{self.batch_index}_station{index + 1}"
        typ = "a"
        self.stations[name] = (name, typ, position, True)
        return name

    def _distance_graph(self, a, b):
        if not self.coords:
            return None
        na = min(self.coords, key=lambda k: math.hypot(a[0] - self.coords[k][0], a[1] - self.coords[k][1]))
        nb = min(self.coords, key=lambda k: math.hypot(b[0] - self.coords[k][0], b[1] - self.coords[k][1]))
        try:
            return nx.shortest_path_length(self.full_graph, na, nb, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _solve(self, robots, tasks, method):
        costs = np.full((self.n, self.n), np.inf)
        paths = [[None] * self.n for _ in range(self.n)]
        t0 = time.perf_counter()
        for i, robot in enumerate(robots):
            for j, task in enumerate(tasks):
                if method == "metric":
                    path = self.metric.path(robot, task[1])
                    if path is None:
                        continue
                    cost = sum(math.hypot(path[k][0] - path[k - 1][0], path[k][1] - path[k - 1][1]) for k in range(1, len(path)))
                    paths[i][j] = path
                else:
                    cost = self._distance_graph(robot, task[1])
                    if cost is None:
                        continue
                    paths[i][j] = [robot, task[1]]
                costs[i, j] = self.alpha_d * cost
        from scipy.optimize import linear_sum_assignment
        if not np.isfinite(costs).any():
            return {}, costs, paths, (time.perf_counter() - t0) * 1000.0
        finite = np.where(np.isfinite(costs), costs, 1e12)
        rows, cols = linear_sum_assignment(finite)
        assignment = {}
        for r, c in zip(rows, cols):
            if np.isfinite(costs[r, c]):
                assignment[c] = (r + 1, costs[r, c], paths[r][c])
        return assignment, costs, paths, (time.perf_counter() - t0) * 1000.0

    def _run_once(self):
        if self.batch_index >= self.batches:
            self.get_logger().info("Benchmark complete: 100 batches")
            self.timer.cancel()
            return
        if not self.metric.ready:
            self.get_logger().info("Waiting for costmap", throttle_duration_sec=5.0)
            return
        if not self.graph_ready:
            self.get_logger().info("Waiting for skeleton graph", throttle_duration_sec=5.0)
            return
        robots, positions = self._sample_positions()
        tasks = [(f"task_{self.batch_index + 1:03d}_{j + 1:02d}", positions[j]) for j in range(self.n)]
        graph_result, _, _, graph_ms = self._solve(robots, tasks, "graph")
        metric_result, _, _, metric_ms = self._solve(robots, tasks, "metric")
        now = time.time()
        graph_points = np.asarray(robots)
        task_points = np.asarray(positions)
        cost_diffs = []
        hausdorffs = []

        for j in range(self.n):
            g = graph_result.get(j)
            m = metric_result.get(j)
            if g and m:
                gc = g[1]
                mc = m[1]
                cost_diffs.append(mc - gc)
                # Hausdorff between paths
                if g[2] and m[2]:
                    path_g = np.asarray(g[2])
                    path_m = np.asarray(m[2])
                    if len(path_g) > 0 and len(path_m) > 0:
                        d_g2m = np.min(np.linalg.norm(path_g[:, None, :] - path_m[None, :, :], axis=2), axis=1)
                        d_m2g = np.min(np.linalg.norm(path_m[:, None, :] - path_g[None, :, :], axis=2), axis=1)
                        hausdorffs.append(float(max(np.max(d_g2m), np.max(d_m2g))))

        def safe_avg(vals):
            return float(np.mean(vals)) if vals else ""

        def safe_min(vals):
            return float(np.min(vals)) if vals else ""

        def safe_max(vals):
            return float(np.max(vals)) if vals else ""

        with open(self.batch_path, "a", newline="") as f:
            csv.writer(f).writerow([
                f"{now:.3f}", str(self._param("run_id", "benchmark")), self.batch_index + 1, self.n,
                self._variance(graph_points), self._variance(task_points),
                len(graph_result), len(metric_result),
                f"{graph_ms:.3f}", f"{metric_ms:.3f}",
                safe_avg(cost_diffs), safe_min(cost_diffs), safe_max(cost_diffs),
                safe_avg(hausdorffs), safe_min(hausdorffs), safe_max(hausdorffs),
            ])
        with open(self.comparison_path, "a", newline="") as f:
            writer = csv.writer(f)
            for j, task in enumerate(tasks):
                g = graph_result.get(j)
                m = metric_result.get(j)
                gc = g[1] if g else None
                mc = m[1] if m else None
                writer.writerow([
                    f"{now:.3f}", str(self._param("run_id", "benchmark")), self.batch_index + 1, task[0],
                    g[0] if g else "", "batch_station_%02d" % (j + 1), f"{gc:.6f}" if gc is not None else "", f"{graph_ms:.3f}",
                    m[0] if m else "", "batch_station_%02d" % (j + 1), f"{mc:.6f}" if mc is not None else "", f"{metric_ms:.3f}",
                    f"{mc - gc:.6f}" if gc is not None and mc is not None else "", "",
                    "OK" if g and m else "PARTIAL",
                ])
        self.batch_index += 1


def main(args=None):
    rclpy.init(args=args)
    node = BenchmarkNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
