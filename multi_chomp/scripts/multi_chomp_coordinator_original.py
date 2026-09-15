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
        self.declare_parameter('logfilepath', os.path.join(os.getcwd(), 'multichomp_metrics.csv'))
        self.declare_parameter('runid', 'original')

        self.num_robots = self.get_parameter('num_robots').value
        self.controller_id = self.get_parameter('controller_id').value
        self.robot_names = [f'robot{i}' for i in range(1, self.num_robots + 1)]

        self.run_id = str(self.get_parameter('runid').value)
        base_log_path = str(self.get_parameter('logfilepath').value)
        base, ext = os.path.splitext(base_log_path)
        if not ext:
            ext = '.csv'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.metrics_log_path = f"{base}_{self.run_id}_coordinator_{timestamp}{ext}"
        self.metrics_logger = TaskLogger(self.metrics_log_path)

        self.get_logger().info(f"Fleet Coordinator Active: {self.robot_names}")

        # --- State Management ---
        self.goals = {}
        self.active_goals = {}
        self.exec_goal_handles = {}
        self.plan_buffer = {}
        self.optimization_in_progress = False
        self.optimize = False
        self.pending_plan_requests = set()
        self.optimizing_plans = []
        self.optimizing_robot_names = set()

        self.active_paths = {}
        self.moving_robots = set()

        # --- Optimization compute-time tracking ---
        self._opt_wall_start = None
        self._opt_iterations = 0
        # The original multi chomp always performs complete replans
        # (fresh Nav2 initializations, 100 iterations, full fleet)
        self._opt_full_replan = True

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

    def _now_sec(self):
        return self.get_clock().now().nanoseconds / 1e9

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

    def _log_compute_time(self, result=None, status='OK', error_message=''):
        """Logs the optimization time of one MultiChompOptimize run.

        Primary source: the algorithm-reported time in result.computation_time
        and the executed solver iterations in result.iterations_executed, both
        set by the C++ action server. Fallback (if the fields are absent):
        the coordinator-measured round trip from goal submission to result
        reception.

        Events:
            CHOMP_TIME_FULL    -> complete replan (fresh initializations)
            CHOMP_TIME_PARTIAL -> partial replan (sliding-window update)

        The time in seconds is stored in the 'duration' column. The 'message'
        column records the source (algorithm|round_trip) so the two can be
        told apart in the CSV.
        """
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
            message = f'{message};{error_message}'

        self.metrics_logger.log(TaskLogRecord(
            timestamp=self._now_sec(),
            run_id=self.run_id,
            task_id='-',
            robot_id='fleet',
            event=event,
            status=status,
            allocation_cost=None,
            duration=compute_time,
            path='',
            collision_flag=0,
            message=message,
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

    # TODO: clean this shit up
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

        self.plan_buffer.pop(robot_name, None)
        self.pending_plan_requests.discard(robot_name)
        self.optimize = True

    def coordination_loop(self):
        if not self.chomp_client.server_is_ready() or self.optimization_in_progress or not self.optimize:
            return

        # # Deviation check: Completely recompute if thrown off track
        # for name in self.robot_names:
        #     current_pose = self.get_robot_pose(name)
        #     if name in self.active_paths and name in self.moving_robots and current_pose:
        #         # Align the active path to the robot's current pose
        #         self.active_paths[name] = self._clip_path_to_robot(
        #             self.active_paths[name], current_pose
        #         )
        #         cx = current_pose.pose.position.x
        #         cy = current_pose.pose.position.y

        #         # Deviation check -> Force Nav2 Replan only if robot is truly off-path
        #         # In the original multi chomp, we purge all buffers and reinstate goals
        #         first_pose = self.active_paths[name].poses[0].pose.position
        #         if math.hypot(first_pose.x - cx, first_pose.y - cy) > 0.6:
        #             self.get_logger().warn(f"Robot {name} deviated heavily. Recomputing path entirely.")
        #             self.goals = self.active_goals
        #             self.active_paths.pop(name, None)
        #             self.plan_buffer.clear()
        #             self.pending_plan_requests.discard(name)

        # Request Nav2 Plans ONLY for robots with pending goals
        # Using list() safely iterates while dictionary size changes
        for name in list(self.goals.keys()):
            if name not in self.plan_buffer and name not in self.pending_plan_requests:
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
        robots_with_goals = [r for r in self.goals]
        robots_waiting = [r for r in robots_with_goals if r in self.pending_plan_requests]
        if len(robots_waiting) > 0:
            return
        if len(self.plan_buffer) > 0 or len(self.moving_robots) > 0:
            self.trigger_fleet_optimization()

        self.optimize = False

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
                self.plan_buffer[robot_name] = result.path
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
        goal_msg.num_robots = self.num_robots
        goal_msg.max_iterations = 100

        self.optimizing_plans = list(self.plan_buffer.keys())
        self.optimizing_robot_names = set(self.optimizing_plans)
        inputs_valid = True

        for name in self.robot_names:
            path_to_send = None
            current_pose = self.get_robot_pose(name)

            if name in self.plan_buffer:
                path_to_send = self._clip_path_to_robot(self.plan_buffer[name], current_pose)
                self.active_paths[name] = path_to_send
            elif name in self.active_paths and name in self.optimizing_robot_names:
                path_to_send = self.active_paths[name]
            else:
                path_to_send = self.create_holding_path(name)

            if path_to_send is None:
                self.get_logger().warn(f"Skipping optimization: Could not get state for {name}")
                inputs_valid = False
                break

            goal_msg.input_paths.append(path_to_send)

        if not inputs_valid:
            self.optimization_in_progress = False
            self.optimizing_plans.clear()
            self.optimizing_robot_names.clear()
            return

        # Start the compute-time measurement for this optimization run
        self._opt_wall_start = time.perf_counter()
        self._opt_iterations = goal_msg.max_iterations
        self._opt_full_replan = True  # every run is a complete replan

        self._log_full_replan('original_full_optimization')

        if len(self.plan_buffer) > 0:
            self.get_logger().info(f"Triggering Fleet Optimization for {self.num_robots} robots...")

        self.chomp_client.send_goal_async(goal_msg).add_done_callback(
            lambda f: self.optimization_response_callback(f)
        )

    def optimization_response_callback(self, future):
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self._log_compute_time(status='ERROR', error_message='goal_rejected')
                self.optimization_in_progress = False
                self.optimizing_plans.clear()
                return

            goal_handle.get_result_async().add_done_callback(
                lambda f: self.optimization_result_callback(f)
            )

        except Exception as e:
            self.get_logger().error(f"Optimization request failed: {e}")
            self._log_compute_time(status='ERROR', error_message=f'goal_send_failed:{e}')
            self.optimization_in_progress = False
            self.optimizing_plans.clear()

    def optimization_result_callback(self, future):
        try:
            result = future.result().result

            # Log the algorithm-reported optimization time for this run
            # (falls back to round-trip time if the action server does not
            # provide computation_time)
            self._log_compute_time(result=result)

            optimized_paths = result.optimized_paths

            if len(optimized_paths) != self.num_robots:
                self.get_logger().error("Mismatch in optimized paths count!")
                self.optimization_in_progress = False
                self.optimizing_plans.clear()
                self.optimizing_robot_names.clear()
                return

            for i, robot_name in enumerate(self.robot_names):
                opt_path = optimized_paths[i]

                if len(opt_path.poses) < 2:
                    continue

                if robot_name in self.optimizing_plans:
                    self.active_paths[robot_name] = opt_path
                    self.execute_path(robot_name, opt_path)

            for n in self.optimizing_plans:
                self.plan_buffer.pop(n, None)

            self.optimizing_plans.clear()
            self.optimizing_robot_names.clear()
            self.optimization_in_progress = False

        except Exception as e:
            self.get_logger().error(f"Optimization callback exception: {e}")
            self._log_compute_time(status='ERROR', error_message=f'result_callback_exception:{e}')
            self.optimization_in_progress = False
            self.optimizing_plans.clear()
            self.optimizing_robot_names.clear()

    def execute_path(self, robot_name, path):
        client = self.nav2_exec_clients.get(robot_name)
        if not client:
            return

        # Timestamp the path
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

        old_handle = self.exec_goal_handles.pop(robot_name, None)
        if old_handle is not None:
            cancel_future = old_handle.cancel_goal_async()
            cancel_future.add_done_callback(
                lambda f, n=robot_name, p=path: self._send_follow_path(n, p)
            )

        else:
            # No prior goal — send immediately
            self._send_follow_path(robot_name, path)

    def _send_follow_path(self, robot_name, path):
        client = self.nav2_exec_clients.get(robot_name)
        if not client:
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
                self.exec_goal_handles[robot_name] = goal_handle
                goal_handle.get_result_async().add_done_callback(
                    lambda f, n=robot_name: self.execute_result_callback(f, n)
                )

            else:
                self.get_logger().error(f"Controller REJECTED path for {robot_name}")
                self._mark_robot_for_replan(robot_name)
        except Exception:
            self._mark_robot_for_replan(robot_name)

    def _mark_robot_for_replan(self, robot_name):
        if robot_name in self.active_goals:
            self.goals[robot_name] = self.active_goals[robot_name]

        self.active_paths.pop(robot_name, None)
        self.exec_goal_handles.pop(robot_name, None)
        self.moving_robots.add(robot_name)
        self.pending_plan_requests.discard(robot_name)
        self.optimize = True

    def execute_result_callback(self, future, robot_name):
        try:
            status = future.result().status

            if status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info(f"{robot_name} securely reached its destination.")
                self.exec_goal_handles.pop(robot_name, None)

                if robot_name not in self.goals and robot_name not in self.active_goals:
                    self.moving_robots.discard(robot_name)
                    self.active_goals.pop(robot_name, None)
                    self.active_paths.pop(robot_name, None)

            elif status == GoalStatus.STATUS_ABORTED:
                self.get_logger().warn(f"FollowPath aborted for {robot_name}, forcing replan")
                self._mark_robot_for_replan(robot_name)

            elif status == GoalStatus.STATUS_CANCELED:
                self.exec_goal_handles.pop(robot_name, None)

        except Exception:
            self._mark_robot_for_replan(robot_name)


def main(args=None):
    rclpy.init(args=args)
    node = FleetCoordinator()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        # ExternalShutdownException fires when the launch file's benchmark
        # timeout shuts the stack down
        node.get_logger().info(f"Executor exited: {e}")
    finally:
        # Ensure the metrics CSV is flushed to disk before the process dies
        try:
            if hasattr(node.metrics_logger, 'close'):
                node.metrics_logger.close()
            elif hasattr(node.metrics_logger, 'flush'):
                node.metrics_logger.flush()
        except Exception:
            pass
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()