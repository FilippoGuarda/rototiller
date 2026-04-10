#!/usr/bin/env python3
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx


@dataclass
class StationConfig:
    name: str
    station_type: str
    position: Tuple[float, float]
    online: bool = True


@dataclass
class Task:
    task_id: str
    timestamp: float
    stations: List[str]
    priority: float = 1.0
    allocation_attempts: int = 0
    last_attempt_time: float = 0.0


@dataclass
class RobotState:
    robot_id: int
    battery_soc: float = 1.0
    max_range_m: float = 10000.0
    usage_index: float = 0.0


def build_full_graph_with_stations(
    base_graph: nx.Graph,
    stations: Dict[str, StationConfig],
    k_nearest: int,
    graph_nodes_map_coords: Dict[int, Tuple[float, float]],
) -> Tuple[nx.Graph, Dict[str, int]]:
    """
    Copy base_graph, add virtual nodes for all online stations and connect each station
    to its k nearest skeleton nodes, just like the original inject_stations_to_graph.
    """
    full_graph = base_graph.copy()
    next_node_id = max(full_graph.nodes) + 1 if full_graph.nodes else 0
    station_nodes: Dict[str, int] = {}

    for name, station in stations.items():
        if not station.online:
            continue

        virt_id = next_node_id
        next_node_id += 1

        full_graph.add_node(virt_id, pos=station.position)
        station_nodes[name] = virt_id

        distances: List[Tuple[float, int]] = []
        for n_id, coords in graph_nodes_map_coords.items():
            d = math.hypot(station.position[0] - coords[0], station.position[1] - coords[1])
            distances.append((d, n_id))

        distances.sort()
        for d, n_id in distances[:k_nearest]:
            full_graph.add_edge(virt_id, n_id, weight=d)

    return full_graph, station_nodes