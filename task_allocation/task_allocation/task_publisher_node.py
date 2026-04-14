#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import random

class TaskPublisherNode(Node):
    """
    ROS 2 node that automatically publishes random multi-station task messages
    using station types to trigger optimal allocation.
    """
    def __init__(self):
        super().__init__("task_publisher_node")
        self.publisher = self.create_publisher(String, "/tasks", 30)

        self.declare_parameter("seed", 13)
        self.declare_parameter("num_tasks", 10)
        self.declare_parameter("min_delay_s", 0.5)
        self.declare_parameter("max_delay_s", 1.0)
        # NEW: declare station_types with a default
        self.declare_parameter("station_types", ["a", "b", "c"])

        random.seed(self.get_parameter("seed").value)
        self.num_tasks = int(self.get_parameter("num_tasks").value)
        self.min_delay = float(self.get_parameter("min_delay_s").value)
        self.max_delay = float(self.get_parameter("max_delay_s").value)

        # After declaration, this is safe:
        self.station_types = list(self.get_parameter("station_types").value)

        self.tasks_published = 0

        self.get_logger().info(
            f"Random task publisher active (Seed: {self.get_parameter('seed').value}, "
            f"Station types: {self.station_types})."
        )
        self.schedule_next()

    def schedule_next(self) -> None:
        if self.tasks_published >= self.num_tasks:
            self.get_logger().info("All random tasks published.")
            return

        # First task has long initial delay to allow graph to stabilize
        delay = (
            random.uniform(self.min_delay, self.max_delay)
            if self.tasks_published > 0
            else 5.0
        )
        self.timer = self.create_timer(delay, self.publish_task)

    def publish_task(self) -> None:
        self.timer.cancel()
        self.tasks_published += 1
        
        task_id = f"task_{self.tasks_published:03d}"
        
        # Generate random sequence of 2 to 3 station types
        if len(self.station_types) >= 3:
            path_length = random.randint(2, 3)
        elif len(self.station_types) == 2:
            path_length = 2
        else:
            path_length = 1

        path_length = max(1, min(path_length, len(self.station_types)))
        stations = random.sample(self.station_types, k=path_length)
        priority = round(random.uniform(1.0, 5.0), 1)
        
        msg_data = f"{task_id},{'|'.join(stations)},{priority}"
        
        msg = String()
        msg.data = msg_data
        self.publisher.publish(msg)
        self.get_logger().info(f"Published task: {msg_data}")
        
        self.schedule_next()

def main(args=None):
    rclpy.init(args=args)
    node = TaskPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
