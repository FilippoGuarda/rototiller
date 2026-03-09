#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from nav2_msgs.action import ComputePathToPose, FollowPath
from extended_spades.action import MultiChompOptimize
from action_msgs.msg import GoalStatus
import nav_msgs.msg
import tf2_ros
from tf2_ros import Buffer, TransformListener
import math
from rclpy.callback_groups import ReentrantCallbackGroup


class FleetCoordinator(Node):
    def __init__(self):
        super().__init__('fleet_coordinator')

        self.cb_group = ReentrantCallbackGroup()

        self.declare_parameter('robot_count', 6)
        self.declare_parameter('controller_id', 'FollowPath')

        self.robot_count = self.get_parameter('robot_count').value
        self.controller_id = self.get_parameter('controller_id').value
        self.robot_names = [f'robot{i}' for i in range(1, self.robot_count + 1)]

        self.get_logger().info(f"Fleet Coordinator Active: {self.robot_names}")

        # --- State Management ---
        self.goals = {}          
        self.active_goals = {}   
        self.new_plan_buffer = {} 
        self.optimization_in_progress = False
        self.pending_plan_requests = set()
        self.optimizing_plans = []  

        self.active_paths = {}       
        self.moving_robots = set()   

        # --- TF Buffer ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- Clients ---
        self.nav2_plan_clients = {}
        self.nav2_exec_clients = {}
        self.path_debug_pubs = {}
        for name in self.robot_names:
            self.nav2_plan_clients[name] = ActionClient(
                self, ComputePathToPose, f'/{name}/compute_path_to_pose', callback_group=self.cb_group)
            self.nav2_exec_clients[name] = ActionClient(
                self, FollowPath, f'/{name}/follow_path', callback_group=self.cb_group)
            self.path_debug_pubs[name] = self.create_publisher(
                nav_msgs.msg.Path, f'/{name}/debug/chomp_optimized_path', 10)

        self.chomp_client = ActionClient(
            self, MultiChompOptimize, 'multi_chomp_optimize', callback_group=self.cb_group)

        # --- Subscribers ---
        self.goal_subs = []
        for name in self.robot_names:
            self.goal_subs.append(
                self.create_subscription(
                    PoseStamped, f'/{name}/spades_goal', 
                    lambda msg, n=name: self.goal_callback(msg, n), 10, callback_group=self.cb_group)
            )

        self.create_timer(0.5, self.coordination_loop, callback_group=self.cb_group)

    def get_robot_pose(self, robot_name):
        """Get the current pose of the robot in the map frame."""
        try:
            target_frame = 'map'
            source_frame = f'{robot_name}/base_link'
            
            if not self.tf_buffer.can_transform(target_frame, source_frame, rclpy.time.Time()):
                return None

            t = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            
            pose = PoseStamped()
            pose.header.frame_id = target_frame
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = t.transform.translation.x
            pose.pose.position.y = t.transform.translation.y
            pose.pose.position.z = t.transform.translation.z
            pose.pose.orientation = t.transform.rotation
            return pose

        except Exception:
            return None

    def _create_stationary_path(self, pose, length=20):
        """Internal helper to create a holding path from a pose."""
        path = nav_msgs.msg.Path()
        path.header = pose.header
        path.poses = [pose for _ in range(length)]
        return path

    def create_holding_path(self, robot_name, length=20):
        """Generates a static path at the robot's current position."""
        pose = self.get_robot_pose(robot_name)
        if not pose:
            return None
        return self._create_stationary_path(pose, length)

    def _clip_path_to_robot(self, path, current_pose):
        """Clips the stale start of a Nav2 path to where the robot CURRENTLY is, 
        eliminating minor rubberbanding caused by planner latency."""
        if not current_pose or len(path.poses) < 2:
            return path

        min_dist = float('inf')
        min_idx = 0
        cx = current_pose.pose.position.x
        cy = current_pose.pose.position.y

        search_horizon = min(len(path.poses), 20)
        for i in range(search_horizon):
            px = path.poses[i].pose.position.x
            py = path.poses[i].pose.position.y
            dist = math.hypot(px - cx, py - cy)
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        if min_idx > 0:
            path.poses = path.poses[min_idx:]
        return path

    def goal_callback(self, msg, robot_name):
        self.get_logger().info(f"Goal received for {robot_name}")
        self.goals[robot_name] = msg
        self.moving_robots.add(robot_name)
        
        self.new_plan_buffer.pop(robot_name, None)
        self.pending_plan_requests.discard(robot_name)

    def coordination_loop(self):
        if not self.chomp_client.server_is_ready() or self.optimization_in_progress:
            return

        # Deviation check: Completely recompute if thrown off track
        for name in self.robot_names:
            current_pose = self.get_robot_pose(name)
            if name in self.active_paths and name in self.moving_robots and current_pose:
                gx = self.active_paths[name].poses[-1].pose.position.x
                gy = self.active_paths[name].poses[-1].pose.position.y
                cx = current_pose.pose.position.x
                cy = current_pose.pose.position.y

                # Goal reached check
                if math.hypot(gx - cx, gy - cy) < 0.35: 
                    self.get_logger().info(f"{name} securely reached its destination.")
                    self.moving_robots.discard(name)
                    self.goals.pop(name, None)
                    self.active_goals.pop(name, None)
                    continue

                # Deviation check -> Force Nav2 Replan
                first_pose = self.active_paths[name].poses[0].pose.position
                if math.hypot(first_pose.x - cx, first_pose.y - cy) > 0.6:
                    self.get_logger().warn(f"Robot {name} deviated heavily. Recomputing path entirely.")
                    if name in self.active_goals:
                        self.goals[name] = self.active_goals[name]
                    self.active_paths.pop(name, None)
                    self.new_plan_buffer.pop(name, None)
                    self.pending_plan_requests.discard(name)

        # Request Nav2 Plans ONLY for robots with pending goals
        # Using list() safely iterates while dictionary size changes
        for name in list(self.goals.keys()):
            if name not in self.new_plan_buffer and name not in self.pending_plan_requests:
                if self.nav2_plan_clients[name].server_is_ready():
                    self.get_logger().info(f"Requesting Global Plan for {name}...")
                    self.pending_plan_requests.add(name)

                    goal_msg = ComputePathToPose.Goal()
                    goal_msg.goal = self.goals[name]
                    goal_msg.planner_id = "GridBased"
                    goal_msg.use_start = False 

                    future = self.nav2_plan_clients[name].send_goal_async(goal_msg)
                    future.add_done_callback(lambda f, n=name: self.nav2_plan_response_callback(f, n))

        # Check Optimization Readiness
        robots_with_new_goals = [r for r in self.goals]
        robots_with_new_plans_ready = [r for r in robots_with_new_goals if r in self.new_plan_buffer]
        
        # Wait until all newly requested plans have been returned by Nav2
        if len(robots_with_new_goals) > 0 and len(robots_with_new_plans_ready) != len(robots_with_new_goals):
            return
            
        # Trigger closed-loop optimization if there are active trajectories or new plans to compute
        if len(robots_with_new_plans_ready) > 0 or len(self.moving_robots) > 0:
             self.trigger_fleet_optimization()

    def nav2_plan_response_callback(self, future, robot_name):
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.pending_plan_requests.discard(robot_name)
                return
            
            goal_handle.get_result_async().add_done_callback(
                lambda f, n=robot_name: self.nav2_plan_result_callback(f, n)
            )
        except Exception:
            self.pending_plan_requests.discard(robot_name)

    def nav2_plan_result_callback(self, future, robot_name):
        try:
            result = future.result().result
            if len(result.path.poses) > 0:
                self.new_plan_buffer[robot_name] = result.path
                self.get_logger().info(f"Plan received for {robot_name}")
                # Transfer from goals queue to active tracker
                if robot_name in self.goals:
                    self.active_goals[robot_name] = self.goals.pop(robot_name)
            else:
                self.get_logger().warn(f"Planner returned empty path for {robot_name}")
        finally:
            self.pending_plan_requests.discard(robot_name)

    def trigger_fleet_optimization(self):
        self.optimization_in_progress = True
        
        goal_msg = MultiChompOptimize.Goal()
        goal_msg.num_robots = self.robot_count

        # Dynamic iterations: 100 for initial convergence, 10 for sliding-window updates
        if len(self.new_plan_buffer) > 0:
            goal_msg.max_iterations = 100
        else:
            goal_msg.max_iterations = 10

        # Keep track of which plans we are processing so we can cleanly pop them later
        self.optimizing_plans = list(self.new_plan_buffer.keys())
        inputs_valid = True
        
        for name in self.robot_names:
            path_to_send = None
            current_pose = self.get_robot_pose(name)

            if name in self.new_plan_buffer:
                # Clip the stale Nav2 path to eliminate latency rubber-banding
                path_to_send = self._clip_path_to_robot(self.new_plan_buffer[name], current_pose)
                self.active_paths[name] = path_to_send
            elif name in self.active_paths and name in self.moving_robots:
                # Empty path -> tells C++ to keep sliding its existing optimized path
                path_to_send = nav_msgs.msg.Path()
            else:
                # Stationary obstacle
                path_to_send = self.create_holding_path(name)
            
            if path_to_send is None:
                self.get_logger().warn(f"Skipping optimization: Could not get state for {name}")
                inputs_valid = False
                break
                
            goal_msg.input_paths.append(path_to_send)

        if not inputs_valid:
            self.optimization_in_progress = False
            self.optimizing_plans.clear()
            return

        if len(self.new_plan_buffer) > 0:
            self.get_logger().info(f"Triggering Fleet Optimization for {self.robot_count} robots...")
        
        self.chomp_client.send_goal_async(goal_msg).add_done_callback(lambda f: self.optimization_response_callback(f))

    def optimization_response_callback(self, future):
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.optimization_in_progress = False
                self.optimizing_plans.clear()
                return
            
            goal_handle.get_result_async().add_done_callback(
                lambda f: self.optimization_result_callback(f)
            )
        except Exception as e:
            self.get_logger().error(f"Optimization request failed: {e}")
            self.optimization_in_progress = False
            self.optimizing_plans.clear()

    def optimization_result_callback(self, future):
        try:
            result = future.result().result
            optimized_paths = result.optimized_paths
            
            if len(optimized_paths) != self.robot_count:
                self.get_logger().error("Mismatch in optimized paths count!")
                self.optimization_in_progress = False
                self.optimizing_plans.clear()
                return

            for i, robot_name in enumerate(self.robot_names):
                opt_path = optimized_paths[i]
                
                if len(opt_path.poses) < 2:
                    continue
                
                if robot_name in self.moving_robots:
                    self.active_paths[robot_name] = opt_path
                    self.execute_path(robot_name, opt_path)
                else:
                    self.active_paths.pop(robot_name, None)

            for n in self.optimizing_plans:
                self.new_plan_buffer.pop(n, None)
            self.optimizing_plans.clear()
            self.optimization_in_progress = False
            
        except Exception as e:
            self.get_logger().error(f"Optimization callback exception: {e}")
            self.optimization_in_progress = False
            self.optimizing_plans.clear()

    def execute_path(self, robot_name, path):
        client = self.nav2_exec_clients.get(robot_name)
        if not client: return

        now = self.get_clock().now().to_msg()
        path.header.stamp = now
        path.header.frame_id = "map"
        for pose in path.poses:
            pose.header.stamp = now
            pose.header.frame_id = "map"

        self.path_debug_pubs[robot_name].publish(path)

        if not client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn(f"Action server not available for {robot_name}")
            return
        
        goal_msg = FollowPath.Goal()
        goal_msg.path = path
        goal_msg.controller_id = self.controller_id
        
        client.send_goal_async(goal_msg).add_done_callback(
            lambda f, n=robot_name: self.execute_response_callback(f, n)
        )

    def execute_response_callback(self, future, robot_name):
        try:
            goal_handle = future.result()
            if goal_handle.accepted:
                goal_handle.get_result_async().add_done_callback(
                    lambda f, n=robot_name: self.execute_result_callback(f, n)
                )
            else:
                self.get_logger().error(f"Controller REJECTED path for {robot_name}")
        except Exception:
            pass

    def execute_result_callback(self, future, robot_name):
        try:
            status = future.result().status
            if status == GoalStatus.STATUS_SUCCEEDED:
                pass
            elif status == GoalStatus.STATUS_ABORTED:
                self.get_logger().warn(f"FollowPath aborted for {robot_name}, keeping as moving")
                # TODO: set a failure counter / re-request Nav2 global plan if repeated aborts
        except Exception:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = FleetCoordinator()
    executor = MultiThreadedExecutor(num_threads=4)
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
