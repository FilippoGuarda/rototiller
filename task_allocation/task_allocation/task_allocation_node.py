#!/usr/bin/env python3
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String
from tf2_ros import Buffer, ConnectivityException, ExtrapolationException, LookupException, TransformListener
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
    target_c_station: str
    priority: float = 1.0


@dataclass
class RobotState:
    robot_id: int
    current_pose: Optional[PoseStamped] = None
    battery_soc: float = 1.0
    max_range_m: float = 10000.0
    usage_index: float = 0.0


class TaskAllocationNode(Node):
    def __init__(self):
        super().__init__(
            'task_allocation_node',
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        self.num_robots = int(self.get_parameter('num_robots').value)
        self.global_frame = str(self.get_parameter('global_frame').value)
        self.robot_prefix = str(self.get_parameter('robot_base_frame_prefix').value)
        self.robot_suffix = str(self.get_parameter('robot_base_frame_suffix').value)
        self.node_match_threshold = float(self.get_parameter('node_match_threshold_m').value)
        self.alpha_d = float(self.get_parameter('alpha_distance').value)
        self.alpha_u = float(self.get_parameter('alpha_usage').value)
        self.alpha_b = float(self.get_parameter('alpha_battery').value)
        self.update_rate_hz = float(self.get_parameter('update_rate_hz').value)

        default_battery_soc = float(self.get_parameter('robot_defaults.battery_soc').value)
        default_max_range = float(self.get_parameter('robot_defaults.max_range_m').value)
        default_usage = float(self.get_parameter('robot_defaults.usage_index').value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.robot_states: Dict[int, RobotState] = {
            i: RobotState(
                robot_id=i,
                battery_soc=default_battery_soc,
                max_range_m=default_max_range,
                usage_index=default_usage,
            )
            for i in range(1, self.num_robots + 1)
        }

        self.stations: Dict[str, StationConfig] = {}
        self.stations_by_type: Dict[str, List[StationConfig]] = {'a': [], 'b': [], 'c': []}
        self.task_queue: List[Task] = []
        self.graph: nx.Graph = nx.Graph()
        self.graph_nodes_map_coords: Dict[int, Tuple[float, float]] = {}

        self.D_RA: Optional[np.ndarray] = None
        self.D_AB: Optional[np.ndarray] = None
        self.D_BC: Optional[np.ndarray] = None

        self.load_station_config()

        self.graph_sub = self.create_subscription(
            MarkerArray,
            '/skeleton_graph/graph_markers',
            self.graph_callback,
            QoSProfile(
                depth=1,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                reliability=QoSReliabilityPolicy.RELIABLE,
            ),
        )
        self.task_sub = self.create_subscription(String, '/tasks', self.task_callback, 10)

        self.nav_clients: Dict[int, ActionClient] = {}
        for robot_id in range(1, self.num_robots + 1):
            action_name = f'/{self.robot_prefix}{robot_id}/navigate_to_pose'
            self.nav_clients[robot_id] = ActionClient(self, NavigateToPose, action_name)

        self.create_timer(1.0 / self.update_rate_hz, self.update_callback)
        self.get_logger().info('Task allocation node initialized.')

    def load_station_config(self):
        self.stations.clear()
        self.stations_by_type = {'a': [], 'b': [], 'c': []}

        for station_type in ['a', 'b', 'c']:
            names_param = f'stations.{station_type}.names'
            names = list(self.get_parameter(names_param).value)
            for station_name in names:
                x = float(self.get_parameter(f'stations.{station_type}.{station_name}.x').value)
                y = float(self.get_parameter(f'stations.{station_type}.{station_name}.y').value)
                online = bool(self.get_parameter(f'stations.{station_type}.{station_name}.online').value)
                station = StationConfig(
                    name=station_name,
                    station_type=station_type,
                    position=(x, y),
                    online=online,
                )
                self.stations[station_name] = station
                self.stations_by_type[station_type].append(station)

        self.get_logger().info(
            f"Loaded stations: A={len(self.stations_by_type['a'])}, "
            f"B={len(self.stations_by_type['b'])}, C={len(self.stations_by_type['c'])}"
        )

    def graph_callback(self, msg: MarkerArray):
        nodes_marker = None
        edges_marker = None
        for marker in msg.markers:
            if 'nodes' in marker.ns:
                nodes_marker = marker
            elif 'edges' in marker.ns:
                edges_marker = marker

        if nodes_marker is None or edges_marker is None:
            self.get_logger().warn('Graph markers missing nodes or edges.')
            return

        self.graph = nx.Graph()
        self.graph_nodes_map_coords.clear()

        for idx, point in enumerate(nodes_marker.points):
            self.graph.add_node(idx, pos=(point.x, point.y))
            self.graph_nodes_map_coords[idx] = (point.x, point.y)

        points = edges_marker.points
        for i in range(0, len(points) - 1, 2):
            p1 = points[i]
            p2 = points[i + 1]
            n1 = self._find_closest_node(p1.x, p1.y)
            n2 = self._find_closest_node(p2.x, p2.y)
            if n1 is None or n2 is None or n1 == n2:
                continue
            weight = math.hypot(p2.x - p1.x, p2.y - p1.y)
            self.graph.add_edge(n1, n2, weight=weight)

        self.compute_distance_matrices()

    def _find_closest_node(self, x: float, y: float) -> Optional[int]:
        best_node = None
        best_dist = float('inf')
        for node_id, (nx_, ny_) in self.graph_nodes_map_coords.items():
            d = math.hypot(x - nx_, y - ny_)
            if d < best_dist and d <= self.node_match_threshold:
                best_dist = d
                best_node = node_id
        return best_node

    def _get_robot_positions(self) -> List[Optional[Tuple[float, float]]]:
        positions: List[Optional[Tuple[float, float]]] = []
        for robot_id in range(1, self.num_robots + 1):
            robot_frame = f'{self.robot_prefix}{robot_id}{self.robot_suffix}'
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.global_frame,
                    robot_frame,
                    rclpy.time.Time(),
                )
                x = transform.transform.translation.x
                y = transform.transform.translation.y
                positions.append((x, y))

                pose = PoseStamped()
                pose.header = transform.header
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = transform.transform.translation.z
                pose.pose.orientation = transform.transform.rotation
                self.robot_states[robot_id].current_pose = pose
            except (LookupException, ConnectivityException, ExtrapolationException):
                positions.append(None)
        return positions

    def compute_distance_matrices(self):
        if self.graph.number_of_nodes() == 0:
            return

        a_stations = self.stations_by_type['a']
        b_stations = self.stations_by_type['b']
        c_stations = self.stations_by_type['c']

        if not a_stations or not b_stations or not c_stations:
            return

        self.D_RA = np.full((self.num_robots, len(a_stations)), np.inf)
        self.D_AB = np.full((len(a_stations), len(b_stations)), np.inf)
        self.D_BC = np.full((len(b_stations), len(c_stations)), np.inf)

        robot_positions = self._get_robot_positions()

        for r in range(self.num_robots):
            if robot_positions[r] is None:
                continue
            robot_node = self._find_closest_node(*robot_positions[r])
            if robot_node is None:
                continue
            for i, station_a in enumerate(a_stations):
                if not station_a.online:
                    continue
                a_node = self._find_closest_node(*station_a.position)
                if a_node is None:
                    continue
                try:
                    self.D_RA[r, i] = nx.shortest_path_length(self.graph, robot_node, a_node, weight='weight')
                except nx.NetworkXNoPath:
                    pass

        for i, station_a in enumerate(a_stations):
            if not station_a.online:
                continue
            a_node = self._find_closest_node(*station_a.position)
            if a_node is None:
                continue
            for j, station_b in enumerate(b_stations):
                if not station_b.online:
                    continue
                b_node = self._find_closest_node(*station_b.position)
                if b_node is None:
                    continue
                try:
                    self.D_AB[i, j] = nx.shortest_path_length(self.graph, a_node, b_node, weight='weight')
                except nx.NetworkXNoPath:
                    pass

        for j, station_b in enumerate(b_stations):
            if not station_b.online:
                continue
            b_node = self._find_closest_node(*station_b.position)
            if b_node is None:
                continue
            for k, station_c in enumerate(c_stations):
                if not station_c.online:
                    continue
                c_node = self._find_closest_node(*station_c.position)
                if c_node is None:
                    continue
                try:
                    self.D_BC[j, k] = nx.shortest_path_length(self.graph, b_node, c_node, weight='weight')
                except nx.NetworkXNoPath:
                    pass

    @staticmethod
    def tropical_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return np.min(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)

    def task_callback(self, msg: String):
        parts = [p.strip() for p in msg.data.split(',')]
        if len(parts) < 2:
            self.get_logger().error(f'Bad task format: {msg.data}')
            return
        priority = float(parts[2]) if len(parts) > 2 else 1.0
        task = Task(
            task_id=parts[0],
            timestamp=self.get_clock().now().nanoseconds / 1e9,
            target_c_station=parts[1],
            priority=priority,
        )
        self.task_queue.append(task)

    def allocate_task(self, target_c_index: int) -> Tuple[Optional[int], float]:
        if self.D_RA is None or self.D_AB is None or self.D_BC is None:
            return None, float('inf')

        R = self.num_robots
        A = self.D_RA.shape[1]
        W_RA = np.full((R, A), np.inf)
        W_AB = self.alpha_d * self.D_AB
        W_BC = self.alpha_d * self.D_BC[:, target_c_index:target_c_index + 1]

        for r in range(R):
            robot = self.robot_states[r + 1]
            remaining_range = robot.battery_soc * robot.max_range_m
            for i in range(A):
                if np.isinf(self.D_RA[r, i]):
                    continue
                d_cost = self.alpha_d * self.D_RA[r, i]
                b_cost = np.inf if remaining_range <= 0.0 else self.alpha_b * (self.D_RA[r, i] / remaining_range)
                u_cost = self.alpha_u * robot.usage_index
                W_RA[r, i] = d_cost + b_cost + u_cost

        M_RB = self.tropical_matmul(W_RA, W_AB)
        M_RC = self.tropical_matmul(M_RB, W_BC)
        costs = M_RC[:, 0]
        idx = int(np.argmin(costs))
        if np.isinf(costs[idx]):
            return None, float('inf')
        return idx + 1, float(costs[idx])

    def send_navigation_goal(self, robot_id: int, task: Task):
        station = self.stations[task.target_c_station]
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = self.global_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = station.position[0]
        goal.pose.pose.position.y = station.position[1]
        goal.pose.pose.orientation.w = 1.0

        client = self.nav_clients[robot_id]
        if not client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn(f'Nav2 server unavailable for robot{robot_id}')
            return
        client.send_goal_async(goal)
        self.robot_states[robot_id].usage_index += 1.0

    def update_callback(self):
        if not self.task_queue:
            return
        task = self.task_queue.pop(0)
        target_station = self.stations.get(task.target_c_station)
        if target_station is None or target_station.station_type != 'c':
            self.get_logger().error(f'Unknown C station: {task.target_c_station}')
            return
        target_idx = self.stations_by_type['c'].index(target_station)
        robot_id, cost = self.allocate_task(target_idx)
        if robot_id is None:
            self.get_logger().warn(f'No feasible robot for task {task.task_id}')
            self.task_queue.insert(0, task)
            return
        self.get_logger().info(f'Assigned task {task.task_id} to robot{robot_id} with cost {cost:.3f}')
        self.send_navigation_goal(robot_id, task)


def main(args=None):
    rclpy.init(args=args)
    node = TaskAllocationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
