#!/usr/bin/env python3
import math
import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import MarkerArray

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

@dataclass
class RobotState:
    robot_id: int
    battery_soc: float = 1.0
    max_range_m: float = 10000.0
    usage_index: float = 0.0

class TaskAllocationNode(Node):
    def __init__(self):
        super().__init__(
            'task_allocation_node',
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True
        )

        self.num_robots = int(self.get_parameter('num_robots').value)
        self.global_frame = str(self.get_parameter('global_frame').value)
        self.robot_prefix = str(self.get_parameter('robot_base_frame_prefix').value)
        self.robot_suffix = str(self.get_parameter('robot_base_frame_suffix').value)
        self.node_match_threshold = float(self.get_parameter('node_match_threshold_m').value)
        if not self.has_parameter('k_nearest_graph_nodes'):
            self.declare_parameter('k_nearest_graph_nodes', 3)
        self.k_nearest = int(self.get_parameter('k_nearest_graph_nodes').value)
        self.alpha_d = float(self.get_parameter('alpha_distance').value)
        self.alpha_u = float(self.get_parameter('alpha_usage').value)
        self.alpha_b = float(self.get_parameter('alpha_battery').value)
        self.update_rate_hz = float(self.get_parameter('update_rate_hz').value)
        default_battery = float(self.get_parameter('robot_defaults.battery_soc').value)
        default_max_range = float(self.get_parameter('robot_defaults.max_range_m').value)
        default_usage = float(self.get_parameter('robot_defaults.usage_index').value)

        self.reentrant_callback_group = ReentrantCallbackGroup()
        self.graph_callback_group = MutuallyExclusiveCallbackGroup()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.robot_states: Dict[int, RobotState] = {
            i: RobotState(i, battery_soc=default_battery, max_range_m=default_max_range, usage_index=default_usage)
            for i in range(1, self.num_robots + 1)
        }
        self.stations: Dict[str, StationConfig] = {}
        self.stations_by_type: Dict[str, List[StationConfig]] = {'a': [], 'b': [], 'c': []}
        self.load_station_config()
        self.task_queue: List[Task] = []
        self.robot_tasks: Dict[int, Optional[dict]] = {i: None for i in range(1, self.num_robots + 1)}
        self.occupied_stations = set()
        self.graph = nx.Graph()
        self.full_graph = nx.Graph()
        self.graph_nodes_map_coords: Dict[int, Tuple[float, float]] = {}
        self.station_nodes: Dict[str, int] = {}

        qos_profile = QoSProfile(
            depth=10,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        self.graph_sub = self.create_subscription(
            MarkerArray, '/skeleton_graph/graph_markers',
            self.graph_callback, qos_profile,
            callback_group=self.graph_callback_group
        )
        self.task_sub = self.create_subscription(
            String, '/tasks', self.task_callback, 10,
            callback_group=self.reentrant_callback_group
        )
        self.goal_pubs = {}
        for robot_id in range(1, self.num_robots + 1):
            topic = f"/{self.robot_prefix}{robot_id}/spades_goal"
            self.goal_pubs[robot_id] = self.create_publisher(PoseStamped, topic, 10)

        self.create_timer(
            1.0 / self.update_rate_hz,
            self.update_callback,
            callback_group=self.reentrant_callback_group
        )
        self.get_logger().info(f"Task Allocation Node initialized. Robots: {self.num_robots}, Update rate: {self.update_rate_hz} Hz")

    def load_station_config(self):
        for stype in ['a', 'b', 'c']:
            names_param = f"stations.{stype}.names"
            names = list(self.get_parameter(names_param).value)
            for sname in names:
                x = float(self.get_parameter(f"stations.{stype}.{sname}.x").value)
                y = float(self.get_parameter(f"stations.{stype}.{sname}.y").value)
                online = bool(self.get_parameter(f"stations.{stype}.{sname}.online").value)
                station = StationConfig(sname, stype, (x, y), online)
                self.stations[sname] = station
                self.stations_by_type[stype].append(station)
        self.get_logger().info(f"Loaded stations: A={len(self.stations_by_type['a'])}, B={len(self.stations_by_type['b'])}, C={len(self.stations_by_type['c'])}")


    def graph_callback(self, msg: MarkerArray):
        nodes_marker = next((m for m in msg.markers if 'nodes' in m.ns), None)
        edges_marker = next((m for m in msg.markers if 'edges' in m.ns), None)

        if not nodes_marker or not edges_marker:
            return

        self.graph.clear()
        self.graph_nodes_map_coords.clear()

        coord_to_idx: Dict[Tuple[float, float], int] = {}
        for idx, point in enumerate(nodes_marker.points):
            self.graph.add_node(idx, pos=(point.x, point.y))
            self.graph_nodes_map_coords[idx] = (point.x, point.y)
            coord_to_idx[(point.x, point.y)] = idx

        points = edges_marker.points
        for i in range(0, len(points) - 1, 2):
            p1, p2 = points[i], points[i + 1]

            n1 = coord_to_idx.get((p1.x, p1.y))
            n2 = coord_to_idx.get((p2.x, p2.y))

            if n1 is None:
                n1 = min(self.graph_nodes_map_coords,
                        key=lambda k: math.hypot(p1.x - self.graph_nodes_map_coords[k][0],
                                                p1.y - self.graph_nodes_map_coords[k][1]))
            if n2 is None:
                n2 = min(self.graph_nodes_map_coords,
                        key=lambda k: math.hypot(p2.x - self.graph_nodes_map_coords[k][0],
                                                p2.y - self.graph_nodes_map_coords[k][1]))

            if n1 != n2:
                weight = math.hypot(p2.x - p1.x, p2.y - p1.y)
                self.graph.add_edge(n1, n2, weight=weight)

        self.get_logger().info(
            f"Graph rebuilt: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )
        self.inject_stations_to_graph()


    def inject_stations_to_graph(self):
        self.full_graph = self.graph.copy()
        next_node_id = max(self.full_graph.nodes) + 1 if self.full_graph.nodes else 0
        self.station_nodes.clear()

        for name, station in self.stations.items():
            if not station.online: continue
            
            virt_id = next_node_id
            next_node_id += 1
            self.full_graph.add_node(virt_id, pos=station.position)
            self.station_nodes[name] = virt_id

            distances = []
            for n_id, coords in self.graph_nodes_map_coords.items():
                d = math.hypot(station.position[0] - coords[0], station.position[1] - coords[1])
                distances.append((d, n_id))
            
            distances.sort()
            for d, n_id in distances[:self.k_nearest]:
                self.full_graph.add_edge(virt_id, n_id, weight=d)

    def find_closest_node(self, x: float, y: float, in_full_graph=True, threshold=None) -> Optional[int]:
        best_node = None
        best_dist = float('inf')
        max_dist = threshold if threshold is not None else self.node_match_threshold
        
        nodes = self.full_graph.nodes(data=True) if in_full_graph else self.graph.nodes(data=True)
        if not nodes:
            for n_id, coords in self.graph_nodes_map_coords.items():
                d = math.hypot(x - coords[0], y - coords[1])
                if d < best_dist and d <= max_dist:
                    best_dist = d
                    best_node = n_id
            return best_node

        for n_id, data in nodes:
            pos = data.get('pos')
            if not pos: continue
            d = math.hypot(x - pos[0], y - pos[1])
            if d < best_dist and d <= max_dist:
                best_dist = d
                best_node = n_id
        return best_node

    def get_robot_position(self, robot_id) -> Optional[Tuple[float, float]]:
        frame = f"{self.robot_prefix}{robot_id}{self.robot_suffix}"
        try:
            t = self.tf_buffer.lookup_transform(self.global_frame,
                                                frame,
                                                rclpy.time.Time())
            return (t.transform.translation.x, t.transform.translation.y)
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed for {frame}: {e}", throttle_duration_sec=5.0)
            return None

    def task_callback(self, msg: String):
        parts = [p.strip() for p in msg.data.split(',')]
        if len(parts) < 2:
            self.get_logger().error(f"Bad task format: {msg.data}")
            return
        
        task_id = parts[0]
        stations_str = parts[1].split('|')
        priority = float(parts[2]) if len(parts) > 2 else 1.0

        task = Task(
            task_id=task_id,
            timestamp=self.get_clock().now().nanoseconds / 1e9,
            stations=stations_str,
            priority=priority
        )
        self.task_queue.append(task)
        self.get_logger().info(f"Queued task {task_id} for sequence: {' -> '.join(stations_str)}")

    def allocate_task(self, task: Task) -> Tuple[Optional[int], float, List[str]]:
        if not self.full_graph.nodes:
            self.get_logger().warn("Skeleton graph is currently empty or uninitialized!", throttle_duration_sec=2.0)
            return None, float('inf'), []

        groups = []
        for item in task.stations:
            if item in self.stations:
                groups.append([item])
            elif item in self.stations_by_type:
                groups.append([s.name for s in self.stations_by_type[item] if s.online])
            else:
                self.get_logger().error(f"Unknown station or type: {item}")
                return None, float('inf'), []

        if any(not g for g in groups):
            self.get_logger().warn("One or more station groups are empty/offline.")
            return None, float('inf'), []

        W_R_S = {r: {} for r in range(1, self.num_robots + 1)}
        for r in range(1, self.num_robots + 1):
            pos = self.get_robot_position(r)
            if pos is None:
                self.get_logger().warn(f"No TF for robot_{r}; excluding it from allocation.")
                rn = None
            else:
                rn = self.find_closest_node(pos[0], pos[1], threshold=float('inf'))
                if rn is None:
                    self.get_logger().warn(f"No graph node near robot_{r} at {pos}; excluding from allocation.")
            robot = self.robot_states[r]
            remaining = robot.battery_soc * robot.max_range_m

            for s_name in set(itertools.chain(*groups)):
                if rn is None or s_name in self.occupied_stations:
                    W_R_S[r][s_name] = float('inf')
                    continue
                    
                s_node = self.station_nodes.get(s_name)
                if s_node is None:
                    W_R_S[r][s_name] = float('inf')
                    continue

                try:
                    dist = nx.shortest_path_length(self.full_graph, rn, s_node, weight='weight')
                    d_cost = self.alpha_d * dist
                    b_cost = float('inf') if remaining <= 0.0 else self.alpha_b * (dist / remaining)
                    u_cost = self.alpha_u * robot.usage_index
                    W_R_S[r][s_name] = d_cost + b_cost + u_cost
                except nx.NetworkXNoPath:
                    W_R_S[r][s_name] = float('inf')

        W_S_S = {}
        for s1 in set(itertools.chain(*groups)):
            W_S_S[s1] = {}
            n1 = self.station_nodes.get(s1)
            for s2 in set(itertools.chain(*groups)):
                n2 = self.station_nodes.get(s2)
                if s1 == s2:
                    W_S_S[s1][s2] = 0.0
                elif n1 is None or n2 is None or s2 in self.occupied_stations:
                    W_S_S[s1][s2] = float('inf')
                else:
                    try:
                        W_S_S[s1][s2] = nx.shortest_path_length(self.full_graph, n1, n2, weight='weight')
                    except nx.NetworkXNoPath:
                        W_S_S[s1][s2] = float('inf')

        best_cost = float('inf')
        best_robot = None
        best_path = []

        for r in range(1, self.num_robots + 1):
            if self.robot_tasks[r] is not None:
                continue 

            for path in itertools.product(*groups):
                if len(set(path)) != len(path):
                    continue
                if any(s in self.occupied_stations for s in path):
                    continue

                cost = W_R_S[r][path[0]]
                for i in range(len(path) - 1):
                    cost += W_S_S[path[i]][path[i+1]] * self.alpha_d

                if cost < best_cost:
                    best_cost = cost
                    best_robot = r
                    best_path = list(path)

        return best_robot, best_cost, best_path

    def send_to_station(self, robot_id: int, station_name: str):
        station = self.stations[station_name]
        msg = PoseStamped()
        msg.header.frame_id = self.global_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(station.position[0])
        msg.pose.position.y = float(station.position[1])
        msg.pose.orientation.w = 1.0
        
        self.goal_pubs[robot_id].publish(msg)
        self.get_logger().info(f"Dispatched robot_{robot_id} to {station_name} via multi_chomp")

    def update_callback(self):
        for r in range(1, self.num_robots + 1):
            task_info = self.robot_tasks[r]
            if not task_info:
                continue

            path = task_info['path']
            curr_idx = task_info['current_idx']
            curr_station_name = path[curr_idx]
            curr_station = self.stations[curr_station_name]

            r_pos = self.get_robot_position(r)
            if r_pos:
                dist = math.hypot(r_pos[0] - curr_station.position[0], r_pos[1] - curr_station.position[1])
                if dist < 0.75:  
                    self.get_logger().info(f"Robot_{r} securely reached {curr_station_name}. Releasing occupancy.")
                    self.occupied_stations.discard(curr_station_name)
                    
                    task_info['current_idx'] += 1
                    if task_info['current_idx'] < len(path):
                        next_station = path[task_info['current_idx']]
                        self.send_to_station(r, next_station)
                    else:
                        self.get_logger().info(f"Robot_{r} completed task {task_info['task_id']}")
                        self.robot_tasks[r] = None

        if self.task_queue:
            self.task_queue.sort(key=lambda x: x.priority, reverse=True)
            
            idle_robots = sum(1 for r in range(1, self.num_robots + 1) if self.robot_tasks[r] is None)
            if idle_robots == 0:
                return

            task = self.task_queue[0]
            robot_id, cost, path = self.allocate_task(task)

            if robot_id is not None and not math.isinf(cost):
                self.task_queue.pop(0)
                self.get_logger().info(f"Assigned task {task.task_id} to robot_{robot_id} (Cost: {cost:.2f}, Path: {' -> '.join(path)})")
                
                self.robot_tasks[robot_id] = {
                    'task_id': task.task_id,
                    'path': path,
                    'current_idx': 0
                }
                
                for s in path:
                    self.occupied_stations.add(s)
                    
                self.robot_states[robot_id].usage_index += 1.0
                self.send_to_station(robot_id, path[0])
            else:
                self.get_logger().warn(f"Task {task.task_id} pending: Evaluating feasible paths, waiting for valid Graph/TF...", throttle_duration_sec=3.0)

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

if __name__ == '__main__':
    main()
