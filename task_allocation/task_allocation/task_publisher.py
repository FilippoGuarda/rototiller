#!/usr/bin/env python3
"""
Simple task publisher for testing task allocation system.
Publishes tasks to /tasks topic in format: "task_id,target_c_station,priority"
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sys


class TaskPublisher(Node):
    """Publishes task messages for testing"""

    def __init__(self):
        super().__init__('task_publisher')
        self.publisher = self.create_publisher(String, '/tasks', 10)
        self.get_logger().info('Task Publisher initialized')

    def publish_task(self, task_id: str, target_c_station: str, priority: float = 1.0):
        """Publish a task to the /tasks topic"""
        msg = String()
        msg.data = f'{task_id},{target_c_station},{priority}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Published task: {msg.data}')


def main(args=None):
    rclpy.init(args=args)

    node = TaskPublisher()

    # Parse command line arguments
    if len(sys.argv) < 3:
        node.get_logger().info('Usage: ros2 run task_allocation task_publisher <task_id> <target_c_station> [priority]')
        node.get_logger().info('Example: ros2 run task_allocation task_publisher task_001 station_c1 1.5')

        # Publish a default test task
        node.publish_task('task_default', 'station_c1', 1.0)
    else:
        task_id = sys.argv[1]
        target_c_station = sys.argv[2]
        priority = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

        node.publish_task(task_id, target_c_station, priority)

    # Keep node alive briefly to ensure message is sent
    rclpy.spin_once(node, timeout_sec=0.5)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
