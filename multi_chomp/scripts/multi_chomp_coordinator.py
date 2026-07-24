#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from nav2_msgs.action import ComputePathToPose, FollowPath
from multi_chomp.action import MultiChompOptimize
from action_msgs.msg import GoalStatus
import nav_msgs.msg
import tf2_ros
from tf2_ros import Buffer, TransformListener
import math
from rclpy.callback_groups import ReentrantCallbackGroup
import os
from datetime import datetime

from task_allocation.task_logger import TaskLogger, TaskLogRecord


class FleetCoordinator(Node):
    def __init__(self):
        super().__init__('fleet_coordinator')

        self.cb_group = ReentrantCallbackGroup()

        self.declare_parameter('robot_count', 6)
        self.declare_parameter('controller_id', 'FollowPath')
        self.declare_parameter('max_optimized_segment_length', 0.25)
        self.declare_parameter('stretch_factor', 2.0)
        self.declare_parameter('replan_cooldown_sec', 2.0)
        self.declare_parameter('stuck_timeout_sec', 5.0)
        self.declare_parameter('stuck_motion_epsilon', 0.05)
        self.declare_parameter('stuck_replan_cooldown_sec', 3.0)
        self.declare_parameter('logfilepath', os.path.join(os.getcwd(), 'multichomp_metrics.csv'))
        self.declare_parameter('runid', 'ours')



        self.robot_count = self.get_parameter('robot_count').value
        self.controller_id = self.get_parameter('controller_id').value
        self.robot_names = [f'robot{i}' for i in range(1, self.robot_count + 1)]

        self.run_id = str(self.get_parameter('runid').value)
        base_log_path = str(self.get_parameter('logfilepath').value)
        base, ext = os.path.splitext(base_log_path)
        if not ext:
            ext = '.csv'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.metrics_log_path = f"{base}_{self.run_id}_coordinator_{timestamp}{ext}"
        self.metrics_logger = TaskLogger(self.metrics_log_path)

        
        self.stuck_timeout_sec = self.get_parameter('stuck_timeout_sec').value
        self.stuck_motion_epsilon = self.get_parameter('stuck_motion_epsilon').value
        self.stuck_replan_cooldown_sec = self.get_parameter('stuck_replan_cooldown_sec').value

        
        self.max_optimized_segment_length = self.get_parameter('max_optimized_segment_length').value
        self.stretch_factor = self.get_parameter('stretch_factor').value
        self.replan_cooldown_sec = self.get_parameter('replan_cooldown_sec').value


        self.get_logger().info(f"Fleet Coordinator Active: {self.robot_names}")

        # --- State Management ---
        self.goals = {}          
        self.active_goals = {}   
        self.new_plan_buffer = {} 
        self.optimization_in_progress = False
        self.pending_plan_requests = set()
        self.optimizing_plans = []
        
        self.last_robot_pose = {}
        self.last_motion_time = {}
        self.last_stuck_replan_time = {}


        self.active_paths = {}       
        self.moving_robots = set()   
        self.last_forced_replan_time = {}
        self.plan_request_seq = {name: 0 for name in self.robot_names}


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
        self.create_timer(1.0, self._log_minimum_distances, callback_group=self.cb_group)


    def _log_full_replan(self, reason):
        self.metrics_logger.log(TaskLogRecord(
            timestamp=self._now_sec(),
            run_id=self.run_id,
            task_id='-',
            robot_id='fleet',
            event='REPLAN_FULL',
            status='OK',
            allocation_cost=None,
            duration=None,
            path='|'.join(self.robot_names),
            collision_flag=0,
            message=reason,
        ))

    def _log_minimum_distances(self):
        poses = {}
        for name in self.robot_names:
            pose = self.get_robot_pose(name)
            if pose is None:
                continue
            poses[name] = (pose.pose.position.x, pose.pose.position.y)

        if len(poses) < 2:
            return

        timestamp = self._now_sec()
        min_distances = {}

        for name, (x1, y1) in poses.items():
            nearest_name = None
            nearest_distance = float('inf')

            for other_name, (x2, y2) in poses.items():
                if other_name == name:
                    continue
                distance = math.hypot(x2 - x1, y2 - y1)
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_name = other_name

            if nearest_name is not None:
                min_distances[name] = (nearest_distance, nearest_name)

        if not min_distances:
            return

        average_min_distance = sum(distance for distance, _ in min_distances.values()) / len(min_distances)

        for robot_name, (min_distance, nearest_name) in min_distances.items():
            self.metrics_logger.log(TaskLogRecord(
                timestamp=timestamp,
                run_id=self.run_id,
                task_id='-',
                robot_id=robot_name,
                event='MIN_DISTANCE',
                status='OK',
                allocation_cost=min_distance,
                duration=average_min_distance,
                path='',
                collision_flag=0,
                message=f'nearest_robot={nearest_name}',
            ))

        self.metrics_logger.log(TaskLogRecord(
            timestamp=timestamp,
            run_id=self.run_id,
            task_id='-',
            robot_id='fleet',
            event='MIN_DISTANCE_AVG',
            status='OK',
            allocation_cost=average_min_distance,
            duration=None,
            path='',
            collision_flag=0,
            message=f'robots_sampled={len(min_distances)}',
        ))

    # Replanning helper functs
    def _segment_lengths(self, path):
        """Return distances between consecutive poses in a path."""
        lengths = []

        if not path or len(path.poses) < 2:
            return lengths

        for i in range(len(path.poses) - 1):
            p1 = path.poses[i].pose.position
            p2 = path.poses[i + 1].pose.position
            lengths.append(math.hypot(p2.x - p1.x, p2.y - p1.y))

        return lengths

    def _is_path_too_stretched(self, robot_name, optimized_path, reference_path=None):
        """ Detects whether Multi-CHOMP produced path segments that are too long.
        This can happen when the costmap changes and the optimizer deforms
        the path too aggressively instead of producing a clean trajectory."""
        opt_lengths = self._segment_lengths(optimized_path)

        if not opt_lengths:
            return False, 0.0, self.max_optimized_segment_length

        max_opt_segment = max(opt_lengths)

        threshold = self.max_optimized_segment_length

        # If we have a reference path, compare against its typical segment length.
        # This avoids false positives when paths are naturally sampled coarsely.
        if reference_path is not None:
            ref_lengths = [
                d for d in self._segment_lengths(reference_path)
                if d > 1e-4
            ]

            if ref_lengths:
                ref_lengths_sorted = sorted(ref_lengths)
                median_ref = ref_lengths_sorted[len(ref_lengths_sorted) // 2]
                threshold = max(
                    threshold,
                    self.stretch_factor * median_ref
                )

        return max_opt_segment > threshold, max_opt_segment, threshold


    def _can_force_replan(self, robot_name):
        """Avoid repeatedly forcing replans every optimization cycle."""
        now_sec = self.get_clock().now().nanoseconds / 1e9
        last_sec = self.last_forced_replan_time.get(robot_name, -float('inf'))

        return (now_sec - last_sec) >= self.replan_cooldown_sec


    def _force_new_initialization_trajectory(self, robot_name, reason):
        """
        Throw away the current optimized path and ask Nav2 for a new initial path.
        """
        if not self._can_force_replan(robot_name):
            self.get_logger().warn(
                f"Skipping forced replan for {robot_name}: cooldown active."
            )
            return False

        if robot_name not in self.active_goals and robot_name not in self.goals:
            self.get_logger().warn(
                f"Cannot force replan for {robot_name}: no active goal available."
            )
            return False

        self.get_logger().warn(
            f"Forcing new initialization trajectory for {robot_name}: {reason}"
        )

        # If the robot already has an active goal, put it back into the planning queue.
        if robot_name in self.active_goals:
            self.goals[robot_name] = self.active_goals[robot_name]

        # Invalidate old path state.
        self.active_paths.pop(robot_name, None)
        self.new_plan_buffer.pop(robot_name, None)
        self.pending_plan_requests.discard(robot_name)
        self.moving_robots.add(robot_name)

        # Invalidate any in-flight Nav2 planning result for this robot.
        self.plan_request_seq[robot_name] += 1

        now_sec = self.get_clock().now().nanoseconds / 1e9
        self.last_forced_replan_time[robot_name] = now_sec

        return True
    
    # Stuck robot helper functions
    def _now_sec(self):
        return self.get_clock().now().nanoseconds / 1e9


    def _pose_xy(self, pose):
        return (
            pose.pose.position.x,
            pose.pose.position.y
        )


    def _update_robot_motion_state(self, robot_name, current_pose):
        """
        Updates the last time the robot was observed moving.

        A robot is considered moving only if its current position changed by more
        than stuck_motion_epsilon compared to the previous stored pose.
        """
        now = self._now_sec()

        if current_pose is None:
            return

        if robot_name not in self.last_robot_pose:
            self.last_robot_pose[robot_name] = current_pose
            self.last_motion_time[robot_name] = now
            return

        old_x, old_y = self._pose_xy(self.last_robot_pose[robot_name])
        new_x, new_y = self._pose_xy(current_pose)

        dist = math.hypot(new_x - old_x, new_y - old_y)

        if dist > self.stuck_motion_epsilon:
            self.last_motion_time[robot_name] = now
            self.last_robot_pose[robot_name] = current_pose


    def _is_robot_stuck(self, robot_name):
        """
        Returns True if robot has not moved enough for stuck_timeout_sec.
        """
        if robot_name not in self.last_motion_time:
            return False

        now = self._now_sec()
        stopped_duration = now - self.last_motion_time[robot_name]

        return stopped_duration > self.stuck_timeout_sec


    def _can_stuck_replan(self, robot_name):
        """
        Prevents repeated replans every control loop while the robot is stuck.
        """
        now = self._now_sec()
        last = self.last_stuck_replan_time.get(robot_name, -float('inf'))

        return (now - last) > self.stuck_replan_cooldown_sec


    def _force_complete_replan_due_to_stuck(self, robot_name):
        """
        Completely discards the current path for this robot and asks Nav2
        to compute a fresh initialization path from the robot's current pose.
        """
        if not self._can_stuck_replan(robot_name):
            return False

        if robot_name not in self.active_goals and robot_name not in self.goals:
            self.get_logger().warn(
                f"{robot_name} seems stuck, but no goal is available for replanning."
            )
            return False

        self.get_logger().warn(
            f"{robot_name} has not moved for more than "
            f"{self.stuck_timeout_sec:.1f} seconds. Recomputing full path."
        )

        # Put the current active goal back into the planning queue.
        if robot_name in self.active_goals:
            self.goals[robot_name] = self.active_goals[robot_name]

        # Remove old path data.
        self.active_paths.pop(robot_name, None)
        self.new_plan_buffer.pop(robot_name, None)
        self.pending_plan_requests.discard(robot_name)

        # Keep robot marked as moving because it still needs to reach the goal.
        self.moving_robots.add(robot_name)

        # Reset stuck timer so we do not instantly trigger again.
        now = self._now_sec()
        self.last_motion_time[robot_name] = now
        self.last_stuck_replan_time[robot_name] = now

        return True


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

        # Reset stuck detection for this new task.
        current_pose = self.get_robot_pose(robot_name)
        now = self._now_sec()

        if current_pose is not None:
            self.last_robot_pose[robot_name] = current_pose

        self.last_motion_time[robot_name] = now
        self.last_stuck_replan_time.pop(robot_name, None)


    def coordination_loop(self):
        if not self.chomp_client.server_is_ready() or self.optimization_in_progress:
            return

        # Deviation check: Completely recompute if thrown off track
            
        for name in self.robot_names:
            current_pose = self.get_robot_pose(name)

            if current_pose:
                self._update_robot_motion_state(name, current_pose)

            if name in self.active_paths and name in self.moving_robots and current_pose:
                # Check if robot is stuck before doing normal path checks.
                if self._is_robot_stuck(name):
                    replanned = self._force_complete_replan_due_to_stuck(name)

                    if replanned:
                        # Skip the rest of the checks for this robot in this cycle.
                        continue

                # Align the active path to the robot's current pose
                self.active_paths[name] = self._clip_path_to_robot(
                    self.active_paths[name], current_pose
                )

                cx = current_pose.pose.position.x
                cy = current_pose.pose.position.y

                   # # Goal reached check
                # # TODO: this is a problem, if a robot path is recalled it initializes with the same position of old destination
                # # it automatically goes into destination reached, gotta check the path update logic
                # if math.hypot(gx - cx, gy - cy) < 0.35: 
                #     self.get_logger().info(f"{name} securely reached its destination.")
                #     self.moving_robots.discard(name)
                #     self.goals.pop(name, None)
                #     self.active_goals.pop(name, None)
                #     self.active_paths.pop(name, None) # TODO: test fix
                #     continue

                # Deviation check -> Force Nav2 Replan only if robot is truly off-path
                first_pose = self.active_paths[name].poses[0].pose.position

                if math.hypot(first_pose.x - cx, first_pose.y - cy) > 0.6:
                    self.get_logger().warn(
                        f"Robot {name} deviated heavily. Recomputing path entirely."
                    )
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
                    self.plan_request_seq[name] += 1
                    request_seq = self.plan_request_seq[name]

                    future = self.nav2_plan_clients[name].send_goal_async(goal_msg)
                    future.add_done_callback(
                        lambda f, n=name, seq=request_seq:
                        self.nav2_plan_response_callback(f, n, seq)
                    )

        # Check Optimization Readiness
        robots_with_new_goals = [r for r in self.goals]
        robots_with_new_plans_ready = [r for r in robots_with_new_goals if r in self.new_plan_buffer]
        
        # Wait until all newly requested plans have been returned by Nav2
        if len(robots_with_new_goals) > 0 and len(robots_with_new_plans_ready) != len(robots_with_new_goals):
            return
            
        # Trigger closed-loop optimization if there are active trajectories or new plans to compute
        if len(robots_with_new_plans_ready) > 0 or len(self.moving_robots) > 0:
             self.trigger_fleet_optimization()

    def nav2_plan_response_callback(self, future, robot_name, request_seq):
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.pending_plan_requests.discard(robot_name)
                return
            
            goal_handle.get_result_async().add_done_callback(
                lambda f, n=robot_name, seq=request_seq:
                self.nav2_plan_result_callback(f, n, seq)
            )
        except Exception:
            self.pending_plan_requests.discard(robot_name)

    def nav2_plan_result_callback(self, future, robot_name, request_seq):
        
        try:
            if request_seq != self.plan_request_seq[robot_name]:
                self.get_logger().warn(
                    f"Ignoring stale Nav2 plan result for {robot_name}"
                )
                self.pending_plan_requests.discard(robot_name)
                return
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
        full_replan = len(self.new_plan_buffer) > 0
        if full_replan:
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

            robots_requiring_replan = set()

            for i, robot_name in enumerate(self.robot_names):
                opt_path = optimized_paths[i]
                
                if len(opt_path.poses) < 2:
                    continue
                
                if robot_name in self.moving_robots:
                    reference_path = self.active_paths.get(robot_name)

                    too_stretched, max_segment, threshold = self._is_path_too_stretched(
                        robot_name,
                        opt_path,
                        reference_path
                    )

                    if too_stretched:
                        self.get_logger().warn(
                            f"Optimized path for {robot_name} is too stretched. "
                            f"Max segment: {max_segment:.3f} m, "
                            f"threshold: {threshold:.3f} m."
                        )

                        replanned = self._force_new_initialization_trajectory(
                            robot_name,
                            reason=(
                                f"optimized path segment too long "
                                f"({max_segment:.3f} m > {threshold:.3f} m)"
                            )
                        )

                        if replanned:
                            robots_requiring_replan.add(robot_name)

                        continue

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
                self.get_logger().info(f"{robot_name} securely reached its destination.")

                # Only treat as finished if there is no newer goal or active goal.
                if robot_name not in self.goals and robot_name not in self.new_plan_buffer:
                    self.moving_robots.discard(robot_name)
                    self.active_goals.pop(robot_name, None)
                    self.active_paths.pop(robot_name, None)

            elif status == GoalStatus.STATUS_ABORTED:
                self.get_logger().warn(f"FollowPath aborted for {robot_name}, keeping as moving")
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
