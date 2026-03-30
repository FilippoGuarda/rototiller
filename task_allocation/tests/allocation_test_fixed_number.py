import random
import time
from typing import List, Tuple, Dict, Optional

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import heapq

INF = float("inf")


# ================================================================
#  E* implementation (static, LSM kernel)
# ================================================================

class EStarPlanner:
    """
    Static E* (Philippsen 2006) on a 2D grid with LSM interpolation kernel.

    - Grid coordinates: (iy, ix), shape (H, W)
    - obstacle_mask: True where cell is not traversable
    - risk: float in [0, 1], same shape as grid

    Computes a continuous crossing-time map v[H, W] suitable for
    gradient-descent path extraction or cost lookups.
    """

    def __init__(
        self,
        obstacle_mask: np.ndarray,
        risk: Optional[np.ndarray] = None,
        resolution: float = 1.0,
        min_speed: float = 1e-3,
    ):
        self.obstacle = obstacle_mask.astype(bool)
        self.H, self.W = self.obstacle.shape
        if risk is None:
            self.risk = np.zeros_like(self.obstacle, dtype=float)
        else:
            assert risk.shape == self.obstacle.shape
            self.risk = np.clip(risk.astype(float), 0.0, 1.0)

        self.h = float(resolution)
        self.min_speed = float(min_speed)
        self.v = np.full_like(self.risk, INF, dtype=float)

    def speed_at(self, iy: int, ix: int) -> float:
        r = float(self.risk[iy, ix])
        F = max(self.min_speed, 1.0 - r)
        return F

    def _neighbors_4(self, iy: int, ix: int):
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            jy, jx = iy + dy, ix + dx
            if 0 <= jy < self.H and 0 <= jx < self.W:
                yield jy, jx

    def k_lsm(self, iy: int, ix: int) -> float:
        neigh_vals: List[Tuple[float, Tuple[int, int]]] = []
        for jy, jx in self._neighbors_4(iy, ix):
            v_n = self.v[jy, jx]
            if not math.isinf(v_n):
                neigh_vals.append((v_n, (jy, jx)))

        if not neigh_vals:
            return INF

        neigh_vals.sort(key=lambda p: p[0])
        TA, _ = neigh_vals[0]
        F = self.speed_at(iy, ix)
        if F <= 0.0:
            return INF

        if len(neigh_vals) == 1:
            return TA + self.h / F

        TC, _ = neigh_vals[1]

        if TC - TA >= self.h / F:
            return TA + self.h / F

        beta = -(TA + TC)
        gamma = 0.5 * (TA * TA + TC * TC - (self.h / F) ** 2)
        disc = beta * beta - 4.0 * gamma
        if disc < 0.0:
            return TA + self.h / F
        T = 0.5 * (-beta + math.sqrt(disc))
        if T <= max(TA, TC):
            return TA + self.h / F
        return T

    def compute_value_function(self, goal: Tuple[int, int]) -> np.ndarray:
        gy, gx = goal
        assert 0 <= gy < self.H and 0 <= gx < self.W
        assert not self.obstacle[gy, gx], "Goal must be in free space"

        self.v.fill(INF)
        self.v[gy, gx] = 0.0

        heap: List[Tuple[float, int, int]] = []
        heapq.heappush(heap, (0.0, gy, gx))

        while heap:
            v_cur, iy, ix = heapq.heappop(heap)
            if v_cur > self.v[iy, ix]:
                continue
            if self.obstacle[iy, ix]:
                continue

            for jy, jx in self._neighbors_4(iy, ix):
                if self.obstacle[jy, jx]:
                    continue
                u = self.k_lsm(jy, jx)
                if u < self.v[jy, jx]:
                    self.v[jy, jx] = u
                    heapq.heappush(heap, (u, jy, jx))

        return self.v.copy()


# ================================================================
#  Random connected graph + Dijkstra
# ================================================================

def make_connected_random_graph(n: int,
                                extra_edges_factor: float = 1.5,
                                w_min: int = 1,
                                w_max: int = 10) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(1, n):
        j = random.randrange(0, i)
        w = random.randint(w_min, w_max)
        G.add_edge(i, j, weight=w)

    m_extra = int(extra_edges_factor * n)
    for _ in range(m_extra):
        u, v = random.sample(range(n), 2)
        if not G.has_edge(u, v):
            w = random.randint(w_min, w_max)
            G.add_edge(u, v, weight=w)

    return G


def dijkstra_cost_matrix(
    G: nx.Graph,
    robots: List[int],
    stations: List[int],
) -> Tuple[np.ndarray, float]:
    R = len(robots)
    assert R == len(stations)
    C = np.zeros((R, R), dtype=float)

    t0 = time.perf_counter()
    for i, r in enumerate(robots):
        lengths = nx.single_source_dijkstra_path_length(G, r, weight="weight")
        for j, s in enumerate(stations):
            C[i, j] = lengths[s]
    elapsed = time.perf_counter() - t0
    return C, elapsed


# ================================================================
#  Metric map, A*, and E*
# ================================================================

def make_metric_map(width: int,
                    height: int,
                    obstacle_prob: float = 0.1) -> np.ndarray:
    grid = np.ones((height, width), dtype=bool)
    mask = np.random.rand(height, width) < obstacle_prob
    grid[mask] = False
    if not grid.any():
        y = np.random.randint(0, height)
        x = np.random.randint(0, width)
        grid[y, x] = True
    return grid


def grid_to_graph(grid: np.ndarray) -> nx.Graph:
    h, w = grid.shape
    G = nx.Graph()
    for y in range(h):
        for x in range(w):
            if not grid[y, x]:
                continue
            G.add_node((x, y))
    for y in range(h):
        for x in range(w):
            if not grid[y, x]:
                continue
            for dx, dy in [(1, 0), (0, 1)]:
                nx_ = x + dx
                ny_ = y + dy
                if 0 <= nx_ < w and 0 <= ny_ < h and grid[ny_, nx_]:
                    G.add_edge((x, y), (nx_, ny_), weight=1.0)
    return G


def sample_robots_stations_on_grid(
    G_grid: nx.Graph,
    R: int,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    comps = sorted(nx.connected_components(G_grid),
                   key=len,
                   reverse=True)
    for comp in comps:
        if len(comp) >= 2 * R:
            comp_nodes = list(comp)
            chosen = random.sample(comp_nodes, 2 * R)
            robots = chosen[:R]
            stations = chosen[R:]
            return robots, stations
    raise RuntimeError("Not enough connected free cells for given R")


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)


def astar_cost_matrix(
    G_grid: nx.Graph,
    robots: List[Tuple[int, int]],
    stations: List[Tuple[int, int]],
) -> Tuple[np.ndarray, float]:
    R = len(robots)
    assert R == len(stations)
    C = np.zeros((R, R), dtype=float)

    t0 = time.perf_counter()
    for i, r in enumerate(robots):
        for j, s in enumerate(stations):
            C[i, j] = nx.astar_path_length(
                G_grid,
                r,
                s,
                heuristic=manhattan,
                weight="weight",
            )
    elapsed = time.perf_counter() - t0
    return C, elapsed


def estar_cost_matrix(
    grid: np.ndarray,
    robots: List[Tuple[int, int]],
    stations: List[Tuple[int, int]],
) -> Tuple[np.ndarray, float]:
    """
    Compute C_ij = crossing-time cost from robot_i to station_j
    using E* on the metric grid.

    Strategy: one global E* per station (goal), then look up
    v[robot_i] to fill column j.
    """
    H, W = grid.shape
    obstacle = ~grid
    risk = np.zeros_like(grid, dtype=float)
    planner = EStarPlanner(obstacle_mask=obstacle, risk=risk, resolution=1.0)

    R = len(robots)
    assert R == len(stations)
    C = np.zeros((R, R), dtype=float)

    t0 = time.perf_counter()
    for j, s in enumerate(stations):
        gy, gx = s[1], s[0]  # (iy, ix) = (y, x)
        v_map = planner.compute_value_function((gy, gx))
        for i, r in enumerate(robots):
            ry, rx = r[1], r[0]
            C[i, j] = v_map[ry, rx]
    elapsed = time.perf_counter() - t0
    return C, elapsed


# ================================================================
#  Global benchmark driver (fixed n=200, varying R)
# ================================================================

def run_benchmark(
    graph_n: int = 200,
    robot_sizes = (3, 5, 8, 13, 21, 34, 55),
    obstacle_prob: float = 0.1,
    repeats: int = 3,
):
    sizes_R = []
    times_dijkstra = []
    times_astar = []
    times_estar = []

    for R in robot_sizes:
        print(f"\n=== n = {graph_n}, R = {R} robots / {R} stations ===")
        sizes_R.append(R)

        # --- Random connected graph (n fixed) + Dijkstra ---
        t_dij_total = 0.0
        for _ in range(repeats):
            G = make_connected_random_graph(graph_n)
            nodes = list(G.nodes())
            chosen = random.sample(nodes, 2 * R)
            robots = chosen[:R]
            stations = chosen[R:]
            _, t_dij = dijkstra_cost_matrix(G, robots, stations)
            t_dij_total += t_dij
        t_dij_mean = t_dij_total / repeats
        times_dijkstra.append(t_dij_mean)
        print(f"Dijkstra (graph): {t_dij_mean:.6f} s (avg over {repeats} runs)")

        # --- Metric map 10 x graph_n + A* and E* ---
        t_astar_total = 0.0
        t_estar_total = 0.0
        for _ in range(repeats):
            while True:
                grid = make_metric_map(width=graph_n, height=10,
                                       obstacle_prob=obstacle_prob)
                G_grid = grid_to_graph(grid)
                if G_grid.number_of_nodes() < 2 * R:
                    continue
                try:
                    robots_g, stations_g = sample_robots_stations_on_grid(
                        G_grid, R
                    )
                    break
                except RuntimeError:
                    continue

            # A*
            _, t_astar = astar_cost_matrix(G_grid, robots_g, stations_g)
            t_astar_total += t_astar

            # E*
            _, t_estar = estar_cost_matrix(grid, robots_g, stations_g)
            t_estar_total += t_estar

        t_astar_mean = t_astar_total / repeats
        t_estar_mean = t_estar_total / repeats
        times_astar.append(t_astar_mean)
        times_estar.append(t_estar_mean)
        print(f"A* (metric grid): {t_astar_mean:.6f} s (avg over {repeats} runs)")
        print(f"E*  (metric grid): {t_estar_mean:.6f} s (avg over {repeats} runs)")

    return (
        np.array(sizes_R),
        np.array(times_dijkstra),
        np.array(times_astar),
        np.array(times_estar),
    )


def plot_results(robot_sizes, t_dij, t_astar, t_estar):
    plt.figure()
    plt.plot(robot_sizes, t_dij, marker="o",
             label="n=200 graph + Dijkstra (NetworkX)")
    plt.plot(robot_sizes, t_astar, marker="s",
             label="10×200 grid + A* (NetworkX)")
    plt.plot(robot_sizes, t_estar, marker="^",
             label="10×200 grid + E* (LSM)")

    plt.xlabel("Number of robots / stations R")
    plt.ylabel("Time to fill R×R cost matrix [s]")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.title("Cost-matrix computation time vs R (fixed n = 200)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    R_vals, t_dij, t_astar, t_estar = run_benchmark()
    plot_results(R_vals, t_dij, t_astar, t_estar)