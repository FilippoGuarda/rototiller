#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose, FollowPath
from multi_chomp.action import MultiChompOptimize
from action_msgs.msg import GoalStatus
import nav_msgs.msg
from tf2_ros import Buffer, TransformListener
import math
import time
from rclpy.callback_groups import ReentrantCallbackGroup
import os
from datetime import datetime

from task_allocation.task_logger import TaskLogger, TaskLogRecord


class FleetCoordinator(Node):
    def __init__(self):
        super().__init__('fleet_coordinator')

        self.cb_group = ReentrantCallbackGroup()

        self.declare_parameter('num_robots', 6)
        self.declare_parameter('controller_id', 'FollowPath')
        self.declare_parameter('max_optimized_segment_length', 0.25)
        self.declare_parameter('stretch_factor', 2.0)
        self.declare_parameter('replan_cooldown_sec', 2.0)
        self.declare_parameter('stuck_timeout_sec', 5.0)
        self.declare_parameter('stuck_motion_epsilon', 0.05)
        self.declare_parameter('stuck_replan_cooldown_sec', 3.0)
        self.declare_parameter('goal_arrival_radius', 0.35)
        self.declare_parameter('logfilepath', os.path.join(os.getcwd(), 'multichomp_metrics.csv'))
        self.declare_parameter('runid', 'ours')
        self.declare_parameter('plan_barrier_timeout_sec', 2.0)

        # New path-publication filter parameters.
        # A path is sent to FollowPath only if its geometric difference from
        # the last path sent to FollowPath exceeds this threshold.
        self.declare_parameter('path_update_threshold', 2.0)
        self.declare_parameter('path_update_metric', 'max')
        self.declare_parameter('path_update_min_points', 2)

        self.num_robots = int(self.get_parameter('num_robots').value)
        self.controller_id = str(self.get_parameter('controller_id').value)
        self.robot_names = [f'robot{i}' for i in range(1, self.num_robots + 1)]

        self.run_id = str(self.get_parameter('runid').value)
        base_log_path = str(self.get_parameter('logfilepath').value)
        base, ext = os.path.splitext(base_log_path)
        if not ext:
            ext = '.csv'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.metrics_log_path = f'{base}_{self.run_id}_coordinator_{timestamp}{ext}'
        self.metrics_logger = TaskLogger(self.metrics_log_path)

        self.stuck_timeout_sec = float(self.get_parameter('stuck_timeout_sec').value)
        self.stuck_motion_epsilon = float(self.get_parameter('stuck_motion_epsilon').value)
        self.stuck_replan_cooldown_sec = float(self.get_parameter('stuck_replan_cooldown_sec').value)
        self.plan_barrier_timeout_sec = float(self.get_parameter('plan_barrier_timeout_sec').value)

        self.max_optimized_segment_length = float(self.get_parameter('max_optimized_segment_length').value)
        self.stretch_factor = float(self.get_parameter('stretch_factor').value)
        self.replan_cooldown_sec = float(self.get_parameter('replan_cooldown_sec').value)
        self.goal_arrival_radius = float(self.get_parameter('goal_arrival_radius').value)

        self.path_update_threshold = float(self.get_parameter('path_update_threshold').value)
        self.path_update_metric = str(self.get_parameter('path_update_metric').value).lower()
        self.path_update_min_points = int(self.get_parameter('path_update_min_points').value)
        if self.path_update_threshold < 0.0:
            raise ValueError('path_update_threshold must be non-negative')
        if self.path_update_metric not in ('max', 'mean', 'rms'):
            raise ValueError("path_update_metric must be 'max', 'mean', or 'rms'")

        self.get_logger().info(f'Fleet Coordinator Active: {self.robot_names}')
        self.get_logger().info(
            f'Path update filter: metric={self.path_update_metric}, '
            f'threshold={self.path_update_threshold:.3f} m')

        self.goals = {}
        self.active_goals = {}
        self.new_plan_buffer = {}
        self.optimization_in_progress = False
        self.pending_plan_requests = set()
        self.optimizing_plans = []
        self._goals_pending_since = None

        self.last_robot_pose = {}
        self.last_motion_time = {}
        self.last_stuck_replan_time = {}

        self.active_paths = {}
        self.moving_robots = set()
        self.last_forced_replan_time = {}
        self.plan_request_seq = {name: 0 for name in self.robot_names}

        # Last path that was actually sent to FollowPath. This is deliberately
        # separate from active_paths, which tracks the coordinator's latest
        # optimized path even when that path was not sent to the controller.
        self.last_sent_paths = {}

        self.exec_goal_handles = {}
        self.exec_in_flight = set()

        self._opt_wall_start = None
        self._opt_iterations = 0
        self._opt_full_replan = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.nav2_plan_clients = {}
        self.nav2_exec_clients = {}
        self.path_debug_pubs = {}
        for name in self.robot_names:
            self.nav2_plan_clients[name] = ActionClient(
                self, ComputePathToPose, f'/{name}/compute_path_to_pose',
                callback_group=self.cb_group)
            self.nav2_exec_clients[name] = ActionClient(
                self, FollowPath, f'/{name}/follow_path',
                callback_group=self.cb_group)
            self.path_debug_pubs[name] = self.create_publisher(
                nav_msgs.msg.Path, f'/{name}/debug/chomp_optimized_path', 10)

        self.chomp_client = ActionClient(
            self, MultiChompOptimize, 'multi_chomp_optimize',
            callback_group=self.cb_group)

        self.goal_subs = []
        for name in self.robot_names:
            self.goal_subs.append(self.create_subscription(
                PoseStamped,
                f'/{name}/spades_goal',
                lambda msg, n=name: self.goal_callback(msg, n),
                10,
                callback_group=self.cb_group))

        self.create_timer(0.5, self.coordination_loop, callback_group=self.cb_group)
        self.create_timer(1.0, self._log_minimum_distances, callback_group=self.cb_group)

    def _now_sec(self):
        return self.get_clock().now().nanoseconds / 1e9

    def _pose_xy(self, pose):
        return pose.pose.position.x, pose.pose.position.y

    def _copy_path(self, path):
        copied = nav_msgs.msg.Path()
        copied.header = path.header
        copied.poses = list(path.poses)
        return copied

    def _path_xy(self, path):
        return [(p.pose.position.x, p.pose.position.y) for p in path.poses]

    def _resample_polyline(self, points, count):
        if count <= 0:
            return []
        if not points:
            return None
        if len(points) == 1:
            return [points[0]] * count

        cumulative = [0.0]
        for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
            cumulative.append(cumulative[-1] + math.hypot(x1 - x0, y1 - y0))

        total = cumulative[-1]
        if total <= 1e-9:
            return [points[0]] * count

        result = []
        segment = 0
        for i in range(count):
            target = total * i / (count - 1) if count > 1 else 0.0
            while segment < len(points) - 2 and cumulative[segment + 1] < target:
                segment += 1
            s0, s1 = cumulative[segment], cumulative[segment + 1]
            alpha = 0.0 if s1 - s0 <= 1e-9 else (target - s0) / (s1 - s0)
            x0, y0 = points[segment]
            x1, y1 = points[segment + 1]
            result.append((x0 + alpha * (x1 - x0), y0 + alpha * (y1 - y0)))
        return result

    def _path_difference(self, old_path, new_path):
        """Compare two paths by arc-length-normalized XY displacement.

        Paths may have different waypoint counts. Both are resampled to the
        same number of points before comparison, so the threshold is expressed
        in metres rather than in waypoint-index-dependent error.
        """
        if old_path is None or new_path is None:
            return float('inf')
        if len(old_path.poses) < self.path_update_min_points:
            return float('inf')
        if len(new_path.poses) < self.path_update_min_points:
            return float('inf')

        count = max(self.path_update_min_points,
                    min(max(len(old_path.poses), len(new_path.poses)), 200))
        old_points = self._resample_polyline(self._path_xy(old_path), count)
        new_points = self._resample_polyline(self._path_xy(new_path), count)
        if old_points is None or new_points is None:
            return float('inf')

        errors = [math.hypot(nx - ox, ny - oy)
                  for (ox, oy), (nx, ny) in zip(old_points, new_points)]
        if not errors:
            return float('inf')
        if self.path_update_metric == 'mean':
            return sum(errors) / len(errors)
        if self.path_update_metric == 'rms':
            return math.sqrt(sum(e * e for e in errors) / len(errors))
        return max(errors)

    def _should_send_path(self, robot_name, candidate_path):
        previous = self.last_sent_paths.get(robot_name)
        difference = self._path_difference(previous, candidate_path)
        should_send = previous is None or difference > self.path_update_threshold
        return should_send, difference

    def _record_sent_path(self, robot_name, path):
        self.last_sent_paths[robot_name] = self._copy_path(path)

    def _clear_path_filter(self, robot_name):
        self.last_sent_paths.pop(robot_name, None)

    def _log_full_replan(self, reason):
        self.metrics_logger.log(TaskLogRecord(
            timestamp=self._now_sec(), run_id=self.run_id, task_id='-',
            robot_id='fleet', event='REPLAN_FULL', status='OK',
            allocation_cost=None, duration=None,
            path='|'.join(self.robot_names), collision_flag=0,
            message=reason))

    def _log_compute_time(self, result=None, status='OK', error_message=''):
        if self._opt_wall_start is None:
            return
        round_trip = time.perf_counter() - self._opt_wall_start
        self._opt_wall_start = None
        algo_time = getattr(result, 'computation_time', None) if result is not None else None
        compute_time = float(algo_time) if algo_time is not None else round_trip
        source = 'algorithm' if algo_time is not None else 'round_trip'
        iters_executed = getattr(result, 'iterations_executed', None) if result is not None else None
        event = 'CHOMP_TIME_FULL' if self._opt_full_replan else 'CHOMP_TIME_PARTIAL'
        message = f'iterations={self._opt_iterations};'
        if iters_executed is not None:
            message += f'iterations_executed={int(iters_executed)};'
        message += f'source={source};round_trip={round_trip:.6f}'
        if error_message:
            message += f';{error_message}'
        self.metrics_logger.log(TaskLogRecord(
            timestamp=self._now_sec(), run_id=self.run_id, task_id='-',
            robot_id='fleet', event=event, status=status,
            allocation_cost=None, duration=compute_time, path='',
            collision_flag=0, message=message))

    def _log_minimum_distances(self):
        poses = {}
        for name in self.robot_names:
            pose = self.get_robot_pose(name)
            if pose is not None:
                poses[name] = self._pose_xy(pose)
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
        average = sum(d for d, _ in min_distances.values()) / len(min_distances)
        for robot_name, (distance, nearest_name) in min_distances.items():
            self.metrics_logger.log(TaskLogRecord(
                timestamp=timestamp, run_id=self.run_id, task_id='-',
                robot_id=robot_name, event='MIN_DISTANCE', status='OK',
                allocation_cost=distance, duration=average, path='',
                collision_flag=0, message=f'nearest_robot={nearest_name}'))
        self.metrics_logger.log(TaskLogRecord(
            timestamp=timestamp, run_id=self.run_id, task_id='-',
            robot_id='fleet', event='MIN_DISTANCE_AVG', status='OK',
            allocation_cost=average, duration=None, path='', collision_flag=0,
            message=f'robots_sampled={len(min_distances)}'))

    def _segment_lengths(self, path):
        if not path or len(path.poses) < 2:
            return []
        return [math.hypot(
            path.poses[i + 1].pose.position.x - path.poses[i].pose.position.x,
            path.poses[i + 1].pose.position.y - path.poses[i].pose.position.y)
            for i in range(len(path.poses) - 1)]

    def _is_path_too_stretched(self, robot_name, optimized_path, reference_path=None):
        lengths = self._segment_lengths(optimized_path)
        if not lengths:
            return False, 0.0, self.max_optimized_segment_length
        max_segment = max(lengths)
        threshold = self.max_optimized_segment_length
        if reference_path is not None:
            ref = [d for d in self._segment_lengths(reference_path) if d > 1e-4]
            if ref:
                threshold = max(threshold, self.stretch_factor * sorted(ref)[len(ref) // 2])
        return max_segment > threshold, max_segment, threshold

    def _can_force_replan(self, robot_name):
        return self._now_sec() - self.last_forced_replan_time.get(
            robot_name, -float('inf')) >= self.replan_cooldown_sec

    def _force_new_initialization_trajectory(self, robot_name, reason):
        if not self._can_force_replan(robot_name):
            self.get_logger().warn(f'Skipping forced replan for {robot_name}: cooldown active.')
            return False
        if robot_name not in self.active_goals and robot_name not in self.goals:
            return False
        self.get_logger().warn(f'Forcing new initialization trajectory for {robot_name}: {reason}')
        if robot_name in self.active_goals:
            self.goals[robot_name] = self.active_goals[robot_name]
        self.active_paths.pop(robot_name, None)
        self.new_plan_buffer.pop(robot_name, None)
        self.pending_plan_requests.discard(robot_name)
        self._clear_path_filter(robot_name)
        self.moving_robots.add(robot_name)
        self.plan_request_seq[robot_name] += 1
        self.last_forced_replan_time[robot_name] = self._now_sec()
        return True

    def _update_robot_motion_state(self, robot_name, current_pose):
        now = self._now_sec()
        if current_pose is None:
            return
        if robot_name not in self.last_robot_pose:
            self.last_robot_pose[robot_name] = current_pose
            self.last_motion_time[robot_name] = now
            return
        old_x, old_y = self._pose_xy(self.last_robot_pose[robot_name])
        new_x, new_y = self._pose_xy(current_pose)
        if math.hypot(new_x - old_x, new_y - old_y) > self.stuck_motion_epsilon:
            self.last_motion_time[robot_name] = now
            self.last_robot_pose[robot_name] = current_pose

    def _is_robot_stuck(self, robot_name):
        return (robot_name in self.last_motion_time and
                self._now_sec() - self.last_motion_time[robot_name] > self.stuck_timeout_sec)

    def _can_stuck_replan(self, robot_name):
        return self._now_sec() - self.last_stuck_replan_time.get(
            robot_name, -float('inf')) > self.stuck_replan_cooldown_sec

    def _force_complete_replan_due_to_stuck(self, robot_name):
        if not self._can_stuck_replan(robot_name):
            return False
        if robot_name not in self.active_goals and robot_name not in self.goals:
            return False
        self.get_logger().warn(
            f'{robot_name} has not moved for more than {self.stuck_timeout_sec:.1f} seconds. '
            'Recomputing full path.')
        if robot_name in self.active_goals:
            self.goals[robot_name] = self.active_goals[robot_name]
        self.active_paths.pop(robot_name, None)
        self.new_plan_buffer.pop(robot_name, None)
        self.pending_plan_requests.discard(robot_name)
        self._clear_path_filter(robot_name)
        self.moving_robots.add(robot_name)
        now = self._now_sec()
        self.last_motion_time[robot_name] = now
        self.last_stuck_replan_time[robot_name] = now
        return True

    def get_robot_pose(self, robot_name):
        try:
            target_frame = 'map'
            source_frame = f'{robot_name}/base_link'
            if not self.tf_buffer.can_transform(target_frame, source_frame, rclpy.time.Time()):
                return None
            transform = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time())
            pose = PoseStamped()
            pose.header.frame_id = target_frame
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = transform.transform.translation.x
            pose.pose.position.y = transform.transform.translation.y
            pose.pose.position.z = transform.transform.translation.z
            pose.pose.orientation = transform.transform.rotation
            return pose
        except Exception:
            return None

    def _create_stationary_path(self, pose, length=20):
        path = nav_msgs.msg.Path()
        path.header = pose.header
        path.poses = [pose for _ in range(length)]
        return path

    def create_holding_path(self, robot_name, length=20):
        pose = self.get_robot_pose(robot_name)
        return self._create_stationary_path(pose, length) if pose else None

    def _clip_path_to_robot(self, path, current_pose):
        if not current_pose or len(path.poses) < 2:
            return path
        cx, cy = self._pose_xy(current_pose)
        distances = [math.hypot(p.pose.position.x - cx, p.pose.position.y - cy)
                     for p in path.poses[:20]]
        min_idx = min(range(len(distances)), key=distances.__getitem__)
        if min_idx > 0:
            path.poses = path.poses[min_idx:]
        return path

    def _is_near_path_end(self, robot_name, current_pose):
        path = self.active_paths.get(robot_name)
        if current_pose is None or not path or not path.poses:
            return False
        end = path.poses[-1].pose.position
        cx, cy = self._pose_xy(current_pose)
        return math.hypot(end.x - cx, end.y - cy) <= self.goal_arrival_radius

    def goal_callback(self, msg, robot_name):
        self.get_logger().info(f'Goal received for {robot_name}')
        self.goals[robot_name] = msg
        self.moving_robots.add(robot_name)
        self.new_plan_buffer.pop(robot_name, None)
        self.pending_plan_requests.discard(robot_name)
        self.active_paths.pop(robot_name, None)
        self._clear_path_filter(robot_name)
        current_pose = self.get_robot_pose(robot_name)
        now = self._now_sec()
        if current_pose is not None:
            self.last_robot_pose[robot_name] = current_pose
        self.last_motion_time[robot_name] = now
        self.last_stuck_replan_time.pop(robot_name, None)

    def coordination_loop(self):
        if not self.chomp_client.server_is_ready() or self.optimization_in_progress:
            return

        for name in self.robot_names:
            current_pose = self.get_robot_pose(name)
            if current_pose:
                self._update_robot_motion_state(name, current_pose)
            if name in self.active_paths and name in self.moving_robots and current_pose:
                if self._is_robot_stuck(name) and self._force_complete_replan_due_to_stuck(name):
                    continue
                self.active_paths[name] = self._clip_path_to_robot(
                    self.active_paths[name], current_pose)
                first = self.active_paths[name].poses[0].pose.position
                cx, cy = self._pose_xy(current_pose)
                if math.hypot(first.x - cx, first.y - cy) > 0.6:
                    if name in self.active_goals:
                        self.goals[name] = self.active_goals[name]
                    self.active_paths.pop(name, None)
                    self.new_plan_buffer.pop(name, None)
                    self.pending_plan_requests.discard(name)
                    self._clear_path_filter(name)

        for name in list(self.goals.keys()):
            if name not in self.new_plan_buffer and name not in self.pending_plan_requests:
                if self.nav2_plan_clients[name].server_is_ready():
                    self.pending_plan_requests.add(name)
                    goal_msg = ComputePathToPose.Goal()
                    goal_msg.goal = self.goals[name]
                    goal_msg.planner_id = 'GridBased'
                    goal_msg.use_start = False
                    self.plan_request_seq[name] += 1
                    seq = self.plan_request_seq[name]
                    future = self.nav2_plan_clients[name].send_goal_async(goal_msg)
                    future.add_done_callback(
                        lambda f, n=name, s=seq: self.nav2_plan_response_callback(f, n, s))

        robots_with_goals = list(self.goals)
        ready = [r for r in robots_with_goals if r in self.new_plan_buffer]
        if robots_with_goals:
            if self._goals_pending_since is None:
                self._goals_pending_since = self._now_sec()
        else:
            self._goals_pending_since = None

        if robots_with_goals and len(ready) != len(robots_with_goals):
            waited = self._now_sec() - (self._goals_pending_since or self._now_sec())
            if waited < self.plan_barrier_timeout_sec:
                return

        if ready or self.moving_robots:
            self.trigger_fleet_optimization()

    def nav2_plan_response_callback(self, future, robot_name, request_seq):
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.pending_plan_requests.discard(robot_name)
                return
            goal_handle.get_result_async().add_done_callback(
                lambda f, n=robot_name, s=request_seq:
                self.nav2_plan_result_callback(f, n, s))
        except Exception:
            self.pending_plan_requests.discard(robot_name)

    def nav2_plan_result_callback(self, future, robot_name, request_seq):
        try:
            if request_seq != self.plan_request_seq[robot_name]:
                return
            result = future.result().result
            if result.path.poses:
                self.new_plan_buffer[robot_name] = result.path
                if robot_name in self.goals:
                    self.active_goals[robot_name] = self.goals.pop(robot_name)
        finally:
            self.pending_plan_requests.discard(robot_name)

    def trigger_fleet_optimization(self):
        self.optimization_in_progress = True
        goal_msg = MultiChompOptimize.Goal()
        goal_msg.num_robots = self.num_robots
        full_replan = bool(self.new_plan_buffer)
        goal_msg.max_iterations = 1000 if full_replan else 100
        self.optimizing_plans = list(self.new_plan_buffer)

        for name in self.robot_names:
            current_pose = self.get_robot_pose(name)
            if name in self.new_plan_buffer:
                path_to_send = self._clip_path_to_robot(
                    self.new_plan_buffer[name], current_pose)
                self.active_paths[name] = path_to_send
            elif name in self.active_paths and name in self.moving_robots:
                path_to_send = nav_msgs.msg.Path()
            else:
                path_to_send = self.create_holding_path(name)
            if path_to_send is None:
                self.optimization_in_progress = False
                self.optimizing_plans.clear()
                return
            goal_msg.input_paths.append(path_to_send)

        self._opt_wall_start = time.perf_counter()
        self._opt_iterations = goal_msg.max_iterations
        self._opt_full_replan = full_replan
        self.chomp_client.send_goal_async(goal_msg).add_done_callback(
            self.optimization_response_callback)

    def optimization_response_callback(self, future):
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self._log_compute_time(status='ERROR', error_message='goal_rejected')
                self.optimization_in_progress = False
                self.optimizing_plans.clear()
                return
            goal_handle.get_result_async().add_done_callback(
                self.optimization_result_callback)
        except Exception as exc:
            self._log_compute_time(status='ERROR', error_message=f'goal_send_failed:{exc}')
            self.optimization_in_progress = False
            self.optimizing_plans.clear()

    def optimization_result_callback(self, future):
        try:
            result = future.result().result
            self._log_compute_time(result=result)
            optimized_paths = result.optimized_paths
            if len(optimized_paths) != self.num_robots:
                self.get_logger().error('Mismatch in optimized paths count!')
                return

            for i, robot_name in enumerate(self.robot_names):
                opt_path = optimized_paths[i]
                if len(opt_path.poses) < 2:
                    continue
                if robot_name in self.moving_robots:
                    current_pose = self.get_robot_pose(robot_name)
                    if self._is_near_path_end(robot_name, current_pose):
                        continue

                reference_path = self.active_paths.get(robot_name)
                too_stretched, max_segment, threshold = self._is_path_too_stretched(
                    robot_name, opt_path, reference_path)
                if too_stretched:
                    self._force_new_initialization_trajectory(
                        robot_name,
                        f'optimized path segment too long '
                        f'({max_segment:.3f} m > {threshold:.3f} m)')
                    continue

                # Keep the latest optimized path for CHOMP/coordinator state.
                self.active_paths[robot_name] = opt_path

                should_send, difference = self._should_send_path(robot_name, opt_path)
                if not should_send:
                    self.get_logger().debug(
                        f'{robot_name}: suppressing controller update; '
                        f'path difference={difference:.4f} m <= '
                        f'threshold={self.path_update_threshold:.4f} m')
                    continue

                self.execute_path(robot_name, opt_path)

            for name in self.optimizing_plans:
                self.new_plan_buffer.pop(name, None)
            self.optimizing_plans.clear()
        except Exception as exc:
            self.get_logger().error(f'Optimization callback exception: {exc}')
            self._log_compute_time(status='ERROR', error_message=f'result_callback_exception:{exc}')
        finally:
            self.optimization_in_progress = False

    def execute_path(self, robot_name, path):
        client = self.nav2_exec_clients.get(robot_name)
        if not client:
            return

        path = self._copy_path(path)
        now = self.get_clock().now().to_msg()
        path.header.stamp = now
        path.header.frame_id = 'map'
        for pose in path.poses:
            pose.header.stamp = now
            pose.header.frame_id = 'map'

        self.path_debug_pubs[robot_name].publish(path)
        if not client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn(f'Action server not available for {robot_name}')
            return

        # Record only after the path has passed the threshold test and is
        # actually being submitted to FollowPath.
        self._record_sent_path(robot_name, path)

        old_handle = self.exec_goal_handles.pop(robot_name, None)
        self.exec_in_flight.discard(robot_name)
        if old_handle is not None:
            old_handle.cancel_goal_async().add_done_callback(
                lambda f, n=robot_name, p=path: self._send_follow_path(n, p))
        else:
            self._send_follow_path(robot_name, path)

    def _send_follow_path(self, robot_name, path):
        client = self.nav2_exec_clients.get(robot_name)
        if not client:
            return
        goal_msg = FollowPath.Goal()
        goal_msg.path = path
        goal_msg.controller_id = self.controller_id
        self.exec_in_flight.add(robot_name)
        client.send_goal_async(goal_msg).add_done_callback(
            lambda f, n=robot_name: self.execute_response_callback(f, n))

    def execute_response_callback(self, future, robot_name):
        try:
            goal_handle = future.result()
            if goal_handle.accepted:
                self.exec_goal_handles[robot_name] = goal_handle
                goal_handle.get_result_async().add_done_callback(
                    lambda f, n=robot_name: self.execute_result_callback(f, n))
            else:
                self.exec_in_flight.discard(robot_name)
        except Exception:
            self.exec_in_flight.discard(robot_name)

    def execute_result_callback(self, future, robot_name):
        try:
            status = future.result().status
            self.exec_in_flight.discard(robot_name)
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info(f'{robot_name} securely reached its destination.')
                self.exec_goal_handles.pop(robot_name, None)
                if robot_name not in self.goals and robot_name not in self.new_plan_buffer:
                    self.moving_robots.discard(robot_name)
                    self.active_goals.pop(robot_name, None)
                    self.active_paths.pop(robot_name, None)
                    self._clear_path_filter(robot_name)
            elif status in (GoalStatus.STATUS_ABORTED, GoalStatus.STATUS_CANCELED):
                self.exec_goal_handles.pop(robot_name, None)
        except Exception:
            self.exec_in_flight.discard(robot_name)

    def main_cleanup(self):
        try:
            if hasattr(self.metrics_logger, 'close'):
                self.metrics_logger.close()
            elif hasattr(self.metrics_logger, 'flush'):
                self.metrics_logger.flush()
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
    except Exception as exc:
        node.get_logger().info(f'Executor exited: {exc}')
    finally:
        node.main_cleanup()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
