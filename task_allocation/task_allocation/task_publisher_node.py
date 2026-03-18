#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class TaskPublisherNode(Node):
    """
    ROS 2 node that automatically publishes a multi-station task message 
    to trigger the allocation system based on launch parameters.
    """
    def __init__(self):
        super().__init__("task_publisher_node")
        self.publisher = self.create_publisher(String, "tasks", 10)
        
        # Declare parameters for the task payload
        self.declare_parameter("task_id", "task_001")
        self.declare_parameter("stations", ["stationa1", "stationc1"])
        self.declare_parameter("priority", 1.0)
        # TODO: instead of delays check for message availability
        self.declare_parameter("publish_delay_s", 2.0)
    
        delay = self.get_parameter("publish_delay_s").value
        self.timer = self.create_timer(delay, self.publish_task)
        self.get_logger().info(f"Task publisher active. Waiting {delay} seconds before calling task...")

    def publish_task(self):
        # Retrieve parameters
        task_id = self.get_parameter("task_id").value
        stations = self.get_parameter("stations").value
        priority = self.get_parameter("priority").value
        
        station_sequence = "|".join(stations)
        msg_data = f"{task_id},{station_sequence},{priority}"
        
        # Call the task through a String message
        msg = String()
        msg.data = msg_data
        self.publisher.publish(msg)
        
        self.get_logger().info(f"Successfully published task: {msg_data}")
        
        self.timer.cancel()

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
