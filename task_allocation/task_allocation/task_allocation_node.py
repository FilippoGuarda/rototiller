#!/usr/bin/env python3
"""
Multi-Robot Task Allocation Node using Tropical Algebra (Min-Plus Semiring)
Implements efficient task-robot assignment with distance, battery, and usage costs.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# ROS2 message imports
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav2_msgs.action import NavigateToPose
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import String, ColorRGBA
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
import tf2_geometry_msgs


@dataclass
class StationConfig:
    """Station configuration from config file"""
    name: str
    station_type: str  # 'a', 'b', or 'c'
    position: Tuple[float, float]  # (x, y) in map frame
    online: bool = True


@dataclass
class Task:
    """Task definition requiring route A->B->C"""
    task_id: str
    timestamp: float
    target_c_station: str  # Final destination station name
    priority: float = 1.0


@dataclass
class RobotState:
    """Current state of a robot"""
    robot_id: int
    current_pose: Optional[PoseStamped] = None
    battery_soc: float = 1.0  # State of charge [0,1]
    max_range_m: float = 10000.0  # Maximum range in meters
    usage_index: float = 0.0  # Cumulative usage metric


class TaskAllocationNode(Node):
    """
    Multi-robot task allocation using tropical (min-plus) algebra
    for efficient distance-based cost computation with battery and usage weighting.
    """

    def __init__(self):
        super().__init__('task_allocation_node')

        # Parameters
        self.declare_parameter('num_robots', 6)
        self.declare_parameter('config_file', 'config/stations.yaml')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('robot_base_frame_prefix', 'robot')
        self.declare_parameter('alpha_distance', 1.0)  # Distance weight
        self.declare_parameter('alpha_usage', 0.5)     # Usage weight
        self.declare_parameter('alpha_battery', 2.0)   # Battery weight
        self.declare_parameter('update_rate_hz', 1.0)

        self.num_robots = self.get_parameter('num_robots').value
        self.global_frame = self.get_parameter('global_frame').value
        self.robot_prefix = self.get_parameter('robot_base_frame_prefix').value
        self.alpha_d = self.get_parameter('alpha_distance').value
        self.alpha_u = self.get_parameter('alpha_usage').value
        self.alpha_b = self.get_parameter('alpha_battery').value

        # TF2 setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # State tracking
        self.robot_states: Dict[int, RobotState] = {}
        for i in range(1, self.num_robots + 1):
            self.robot_states[i] = RobotState(robot_id=i)

        self.stations: Dict[str, StationConfig] = {}
        self.stations_by_type: Dict[str, List[StationConfig]] = {'a': [], 'b': [], 'c': []}
        self.task_queue: List[Task] = []
        self.graph: Optional[nx.Graph] = None
        self.graph_nodes_map_coords: Dict[int, Tuple[float, float]] = {}  # node_id -> (x,y)

        # Distance matrices (computed from graph)
        self.D_RA: Optional[np.ndarray] = None  # Robot to A stations
        self.D_AB: Optional[np.ndarray] = None  # A to B stations
        self.D_BC: Optional[np.ndarray] = None  # B to C stations

        # Subscribers
        self.graph_sub = self.create_subscription(
            MarkerArray,
            '/skeleton_graph/graph_markers',
            self.graph_callback,
            QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                      reliability=QoSReliabilityPolicy.RELIABLE)
        )

        self.task_sub = self.create_subscription(
            String,
            '/tasks',
            self.task_callback,
            10
        )

        # Action clients for each robot
        self.nav_clients: Dict[int, ActionClient] = {}
        for robot_id in range(1, self.num_robots + 1):
            action_name = f'/{self.robot_prefix}{robot_id}/navigate_to_pose'
            self.nav_clients[robot_id] = ActionClient(
                self, NavigateToPose, action_name
            )

        # Publishers
        self.assignment_pub = self.create_publisher(
            MarkerArray, '/task_assignments', 10
        )

        # Timers
        update_period = 1.0 / self.get_parameter('update_rate_hz').value
        self.timer = self.create_timer(update_period, self.update_callback)

        # Load station configuration
        self.load_station_config()

        self.get_logger().info(f'Task Allocation Node initialized with {self.num_robots} robots')


    def load_station_config(self):
        """Load station configurations from parameter server or config file"""
        # Example configuration - in production, load from YAML file
        # For this example, we'll create a simple configuration

        # Declare parameters for stations
        self.declare_parameter('stations.a', [
            {'name': 'station_a1', 'x': 1.0, 'y': 0.0},
            {'name': 'station_a2', 'x': 2.0, 'y': 1.0}
        ])
        self.declare_parameter('stations.b', [
            {'name': 'station_b1', 'x': 5.0, 'y': 2.0},
            {'name': 'station_b2', 'x': 6.0, 'y': 3.0}
        ])
        self.declare_parameter('stations.c', [
            {'name': 'station_c1', 'x': 10.0, 'y': 5.0},
            {'name': 'station_c2', 'x': 11.0, 'y': 6.0}
        ])

        # Load stations from parameters
        for station_type in ['a', 'b', 'c']:
            param_name = f'stations.{station_type}'
            try:
                stations_list = self.get_parameter(param_name).value
                for station_data in stations_list:
                    station = StationConfig(
                        name=station_data['name'],
                        station_type=station_type,
                        position=(station_data['x'], station_data['y']),
                        online=station_data.get('online', True)
                    )
                    self.stations[station.name] = station
                    self.stations_by_type[station_type].append(station)

                self.get_logger().info(
                    f'Loaded {len(self.stations_by_type[station_type])} type-{station_type} stations'
                )
            except Exception as e:
                self.get_logger().warn(f'Failed to load {station_type} stations: {e}')


    def graph_callback(self, msg: MarkerArray):
        """Process skeleton graph from graph_generator_node"""
        # Extract nodes and edges from MarkerArray
        # Nodes are in marker with ns='skeleton_graph/nodes'
        # Edges are in marker with ns='skeleton_graph/edges'

        nodes_marker = None
        edges_marker = None

        for marker in msg.markers:
            if 'nodes' in marker.ns:
                nodes_marker = marker
            elif 'edges' in marker.ns:
                edges_marker = marker

        if nodes_marker is None or edges_marker is None:
            self.get_logger().warn('Incomplete graph markers received')
            return

        # Build NetworkX graph from markers
        self.graph = nx.Graph()
        self.graph_nodes_map_coords.clear()

        # Add nodes (from SPHERE_LIST marker points)
        for idx, point in enumerate(nodes_marker.points):
            node_id = idx
            self.graph.add_node(node_id, pos=(point.x, point.y))
            self.graph_nodes_map_coords[node_id] = (point.x, point.y)

        # Add edges (from LINE_LIST marker - pairs of points)
        if len(edges_marker.points) >= 2:
            for i in range(0, len(edges_marker.points), 2):
                p1 = edges_marker.points[i]
                p2 = edges_marker.points[i+1]

                # Find closest nodes to these points
                n1 = self._find_closest_node(p1.x, p1.y)
                n2 = self._find_closest_node(p2.x, p2.y)

                if n1 is not None and n2 is not None and n1 != n2:
                    dist = np.hypot(p2.x - p1.x, p2.y - p1.y)
                    self.graph.add_edge(n1, n2, weight=dist)

        self.get_logger().info(
            f'Graph updated: {self.graph.number_of_nodes()} nodes, '
            f'{self.graph.number_of_edges()} edges'
        )

        # Recompute distance matrices
        self.compute_distance_matrices()


    def _find_closest_node(self, x: float, y: float, threshold: float = 0.1) -> Optional[int]:
        """Find graph node closest to given (x,y) position"""
        min_dist = float('inf')
        closest_node = None

        for node_id, (nx, ny) in self.graph_nodes_map_coords.items():
            dist = np.hypot(x - nx, y - ny)
            if dist < min_dist and dist < threshold:
                min_dist = dist
                closest_node = node_id

        return closest_node


    def compute_distance_matrices(self):
        """
        Compute distance cost matrices using NetworkX shortest path on topological graph.
        Implements tropical (min-plus) computation.
        """
        if self.graph is None or self.graph.number_of_nodes() == 0:
            self.get_logger().warn('Cannot compute distances: graph not ready')
            return

        # Get counts
        R = self.num_robots
        A = len(self.stations_by_type['a'])
        B = len(self.stations_by_type['b'])
        C = len(self.stations_by_type['c'])

        if A == 0 or B == 0 or C == 0:
            self.get_logger().warn('Cannot compute distances: missing station types')
            return

        # Initialize distance matrices with infinity (tropical zero in min-plus)
        self.D_RA = np.full((R, A), np.inf)
        self.D_AB = np.full((A, B), np.inf)
        self.D_BC = np.full((B, C), np.inf)

        # Get robot positions
        robot_positions = self._get_robot_positions()

        # Compute D_RA: Robot to A-stations
        for r in range(R):
            if robot_positions[r] is None:
                continue
            robot_node = self._find_closest_node(*robot_positions[r])
            if robot_node is None:
                continue

            for i, station_a in enumerate(self.stations_by_type['a']):
                if not station_a.online:
                    continue

                station_node = self._find_closest_node(*station_a.position)
                if station_node is None:
                    continue

                try:
                    path_len = nx.shortest_path_length(
                        self.graph, robot_node, station_node, weight='weight'
                    )
                    self.D_RA[r, i] = path_len
                except nx.NetworkXNoPath:
                    pass  # Keep inf

        # Compute D_AB: A-stations to B-stations
        for i, station_a in enumerate(self.stations_by_type['a']):
            if not station_a.online:
                continue
            node_a = self._find_closest_node(*station_a.position)
            if node_a is None:
                continue

            for j, station_b in enumerate(self.stations_by_type['b']):
                if not station_b.online:
                    continue
                node_b = self._find_closest_node(*station_b.position)
                if node_b is None:
                    continue

                try:
                    path_len = nx.shortest_path_length(
                        self.graph, node_a, node_b, weight='weight'
                    )
                    self.D_AB[i, j] = path_len
                except nx.NetworkXNoPath:
                    pass

        # Compute D_BC: B-stations to C-stations
        for j, station_b in enumerate(self.stations_by_type['b']):
            if not station_b.online:
                continue
            node_b = self._find_closest_node(*station_b.position)
            if node_b is None:
                continue

            for k, station_c in enumerate(self.stations_by_type['c']):
                if not station_c.online:
                    continue
                node_c = self._find_closest_node(*station_c.position)
                if node_c is None:
                    continue

                try:
                    path_len = nx.shortest_path_length(
                        self.graph, node_b, node_c, weight='weight'
                    )
                    self.D_BC[j, k] = path_len
                except nx.NetworkXNoPath:
                    pass

        self.get_logger().info('Distance matrices computed using graph shortest paths')


    def _get_robot_positions(self) -> List[Optional[Tuple[float, float]]]:
        """Get current positions of all robots from TF"""
        positions = []

        for robot_id in range(1, self.num_robots + 1):
            robot_frame = f'{self.robot_prefix}{robot_id}/base_link'

            try:
                transform = self.tf_buffer.lookup_transform(
                    self.global_frame,
                    robot_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )

                x = transform.transform.translation.x
                y = transform.transform.translation.y
                positions.append((x, y))

                # Update robot state
                pose = PoseStamped()
                pose.header = transform.header
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = transform.transform.translation.z
                pose.pose.orientation = transform.transform.rotation
                self.robot_states[robot_id].current_pose = pose

            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                self.get_logger().debug(
                    f'Could not get transform for robot {robot_id}: {e}',
                    throttle_duration_sec=5.0
                )
                positions.append(None)

        return positions


    def task_callback(self, msg: String):
        """Receive new task from /tasks topic"""
        # Parse task (simple format: "task_id,target_c_station,priority")
        try:
            parts = msg.data.split(',')
            task = Task(
                task_id=parts[0].strip(),
                timestamp=self.get_clock().now().nanoseconds / 1e9,
                target_c_station=parts[1].strip(),
                priority=float(parts[2].strip()) if len(parts) > 2 else 1.0
            )
            self.task_queue.append(task)
            self.get_logger().info(f'Received task: {task.task_id} -> {task.target_c_station}')
        except Exception as e:
            self.get_logger().error(f'Failed to parse task: {msg.data}, error: {e}')


    def update_callback(self):
        """Main update loop: allocate tasks to robots"""
        if not self.task_queue:
            return

        if self.D_RA is None or self.D_AB is None or self.D_BC is None:
            self.get_logger().warn('Distance matrices not ready', throttle_duration_sec=5.0)
            return

        # Process first task in queue
        task = self.task_queue.pop(0)

        # Find index of target C station
        target_station = self.stations.get(task.target_c_station)
        if target_station is None or target_station.station_type != 'c':
            self.get_logger().error(f'Invalid target station: {task.target_c_station}')
            return

        k = self.stations_by_type['c'].index(target_station)

        # Compute tropical cost matrix and allocate
        robot_id, cost = self.allocate_task(k)

        if robot_id is not None:
            self.get_logger().info(
                f'Allocated task {task.task_id} to robot{robot_id} with cost {cost:.2f}'
            )
            # Send navigation goal
            self.send_navigation_goal(robot_id, task)
        else:
            self.get_logger().warn(f'Could not allocate task {task.task_id}')
            # Re-queue task
            self.task_queue.append(task)


    def allocate_task(self, k: int) -> Tuple[Optional[int], float]:
        """
        Allocate task to robot using tropical (min-plus) algebra.

        Args:
            k: Index of target C station

        Returns:
            (robot_id, min_cost) or (None, inf) if allocation fails
        """
        R = self.num_robots
        A = len(self.stations_by_type['a'])
        B = len(self.stations_by_type['b'])

        # Compute weighted distance matrices
        W_RA = np.full((R, A), np.inf)
        W_AB = self.alpha_d * self.D_AB  # Pure distance for inter-station
        W_BC = self.alpha_d * self.D_BC[:, k:k+1]  # Only to target C station

        # Add robot-state terms to W_RA
        for r in range(R):
            robot = self.robot_states[r+1]

            for i in range(A):
                if np.isinf(self.D_RA[r, i]):
                    continue

                # Distance cost
                d_cost = self.alpha_d * self.D_RA[r, i]

                # Battery penalty: distance / remaining_range
                range_remaining = robot.battery_soc * robot.max_range_m
                if range_remaining > 0:
                    battery_penalty = self.alpha_b * (self.D_RA[r, i] / range_remaining)
                else:
                    battery_penalty = np.inf

                # Usage penalty
                usage_penalty = self.alpha_u * robot.usage_index

                W_RA[r, i] = d_cost + battery_penalty + usage_penalty

        # Tropical matrix products (min-plus)
        # M_RB = W_RA ⊗ W_AB
        M_RB = self.tropical_matmul(W_RA, W_AB)

        # M_RC = M_RB ⊗ W_BC
        M_RC = self.tropical_matmul(M_RB, W_BC)

        # Find minimum cost robot
        costs = M_RC[:, 0]  # Single column (target C station)
        min_idx = np.argmin(costs)
        min_cost = costs[min_idx]

        if np.isinf(min_cost):
            return None, min_cost

        return min_idx + 1, min_cost  # robot_id is 1-indexed


    @staticmethod
    def tropical_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Tropical (min-plus) matrix multiplication.
        (A ⊗ B)_ij = min_k (A_ik + B_kj)
        """
        # Use broadcasting for efficient computation
        # A: (m, n), B: (n, p) -> C: (m, p)
        m, n = A.shape
        n2, p = B.shape
        assert n == n2, "Matrix dimension mismatch"

        # Reshape for broadcasting: A (m, n, 1), B (1, n, p)
        A_expanded = A[:, :, np.newaxis]  # (m, n, 1)
        B_expanded = B[np.newaxis, :, :]  # (1, n, p)

        # Element-wise tropical multiplication (addition) then tropical sum (min)
        C = np.min(A_expanded + B_expanded, axis=1)  # (m, p)

        return C


    def send_navigation_goal(self, robot_id: int, task: Task):
        """Send navigation goal to selected robot via Nav2"""
        # For full implementation, would compute optimal route through A->B->C
        # Here we send goal to final C station

        target_station = self.stations[task.target_c_station]

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = self.global_frame
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = target_station.position[0]
        goal_msg.pose.pose.position.y = target_station.position[1]
        goal_msg.pose.pose.position.z = 0.0
        goal_msg.pose.pose.orientation.w = 1.0

        client = self.nav_clients[robot_id]

        if not client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn(f'Nav2 action server not available for robot{robot_id}')
            return

        self.get_logger().info(f'Sending goal to robot{robot_id}')
        client.send_goal_async(goal_msg)

        # Update robot usage
        self.robot_states[robot_id].usage_index += 1.0


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
