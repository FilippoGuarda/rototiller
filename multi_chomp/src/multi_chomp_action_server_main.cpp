#include "rclcpp/rclcpp.hpp"
#include "multi_chomp/multi_chomp_action_server.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MultiChompActionServer>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}