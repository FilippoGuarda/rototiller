#!/usr/bin/env python3
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class TaskPublisher(Node):
    def __init__(self):
        super().__init__('task_publisher')
        self.publisher = self.create_publisher(String, '/tasks', 10)

    def publish_task(self, task_id: str, target_c_station: str, priority: float = 1.0):
        msg = String()
        msg.data = f'{task_id},{target_c_station},{priority}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Published task: {msg.data}')


def main(args=None):
    rclpy.init(args=args)
    node = TaskPublisher()

    if len(sys.argv) < 3:
        node.publish_task('task_default', 'station_c1', 1.0)
    else:
        task_id = sys.argv[1]
        target_c_station = sys.argv[2]
        priority = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
        node.publish_task(task_id, target_c_station, priority)

    rclpy.spin_once(node, timeout_sec=0.5)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
