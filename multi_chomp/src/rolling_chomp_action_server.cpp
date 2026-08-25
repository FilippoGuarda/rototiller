#include "multi_chomp/rolling_chomp_action_server.hpp"
#include "multi_chomp/rolling_chomp.hpp"

using std::placeholders::_1;
using std::placeholders::_2;

MultiChompActionServer::MultiChompActionServer(
    const rclcpp::NodeOptions & options) 
    : Node("multi_chomp_action_server", options),
      is_optimizing_(false) // Initialize atomic flag
{
    optimizer_ = std::make_shared<MultiChompNode>();

    auto exec = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
    exec->add_node(optimizer_);
    std::thread([exec]() { exec->spin();}).detach();

    action_server_ = rclcpp_action::create_server<MultiChompOptimize>(
        this,
        "multi_chomp_optimize",
        std::bind(&MultiChompActionServer::handle_goal, this, _1, _2),
        std::bind(&MultiChompActionServer::handle_cancel,   this, _1),
        std::bind(&MultiChompActionServer::handle_accepted, this, _1));        

    RCLCPP_INFO(this->get_logger(), "multi chomp action server started");
}

rclcpp_action::GoalResponse
MultiChompActionServer::handle_goal(
    const rclcpp_action::GoalUUID &,
    std::shared_ptr<const MultiChompOptimize::Goal> goal)
{
  if (goal->num_robots == 0 ||
      static_cast<int>(goal->input_paths.size()) != goal->num_robots) {
    RCLCPP_WARN(this->get_logger(), "Rejecting goal: invalid path dimensions");
    return rclcpp_action::GoalResponse::REJECT;
  }

  bool expected = false;
  if (!is_optimizing_.compare_exchange_strong(expected, true)) {
    RCLCPP_WARN(this->get_logger(), "Rejecting goal: Optimizer is currently processing a trajectory");
    return rclcpp_action::GoalResponse::REJECT;
  }

  if (optimizer_->get_num_robots() != static_cast<int>(goal->num_robots)) {
    RCLCPP_INFO(this->get_logger(), "Dynamic reconfiguration: resizing optimizer");
  }

  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse
MultiChompActionServer::handle_cancel(
    const std::shared_ptr<GoalHandleMultiChomp> /*goal_handle*/)
{
  RCLCPP_INFO(this->get_logger(), "Cancel request received");
  return rclcpp_action::CancelResponse::ACCEPT;
}

void MultiChompActionServer::handle_accepted(
    const std::shared_ptr<GoalHandleMultiChomp> goal_handle)
{
  std::thread(
      std::bind(&MultiChompActionServer::execute_goal, this, goal_handle)
  ).detach();
}

bool MultiChompActionServer::load_paths_into_state(
    const std::vector<nav_msgs::msg::Path> & paths)
{
  return optimizer_->set_paths(paths);
}

std::vector<nav_msgs::msg::Path>
MultiChompActionServer::export_state_to_paths(
    const std::vector<nav_msgs::msg::Path> & template_paths) const
{
  return optimizer_->get_paths(template_paths);
}

void MultiChompActionServer::execute_goal(
    const std::shared_ptr<GoalHandleMultiChomp> goal_handle)
{
  const auto goal = goal_handle->get_goal();
  MultiChompOptimize::Result result;
  MultiChompOptimize::Feedback feedback;

  auto cleanup_state = [this]() {
      is_optimizing_.store(false);
  };

  rclcpp::Rate wait_rate(1.0); 
  while (rclcpp::ok() && !optimizer_->has_map()) {
      if (goal_handle->is_canceling()) {
          goal_handle->canceled(std::make_shared<MultiChompOptimize::Result>(result));
          cleanup_state();
          return;
      }
      RCLCPP_WARN(this->get_logger(), "Waiting for global costmap...");
      wait_rate.sleep();
  }

  if (!load_paths_into_state(goal->input_paths)) {
    goal_handle->abort(std::make_shared<MultiChompOptimize::Result>(result));
    cleanup_state();
    return;
  }

  const uint32_t max_iter = (goal->max_iterations > 0) ? goal->max_iterations : 100;
  const double min_cost_change = 1e-4;
  double prev_cost = 1e9;
  uint32_t plateau_count = 0;

  for (uint32_t iter = 0; iter < max_iter; ++iter) {
    if (goal_handle->is_canceling()) {
      goal_handle->canceled(std::make_shared<MultiChompOptimize::Result>(result));
      cleanup_state();
      return;
    }

    optimizer_->solve_step();
    double current_cost = optimizer_->compute_current_cost();
    
    if (iter > 10) {
      if (std::abs(prev_cost - current_cost) < min_cost_change) {
        plateau_count++;
        if (plateau_count >= 3) {
          RCLCPP_INFO(this->get_logger(), "Converged at iteration %u", iter);
          break;
        }
      } else {
        plateau_count = 0;
      }
    }

    prev_cost = current_cost;
    
    if (iter % 10 == 0) {
      feedback.progress = static_cast<float>(iter) / static_cast<float>(max_iter);
      feedback.current_iteration = iter;
      feedback.current_cost = current_cost;
      goal_handle->publish_feedback(std::make_shared<MultiChompOptimize::Feedback>(feedback));
    }
  }

  optimizer_->update_starts_from_tf();
  result.optimized_paths = export_state_to_paths(goal->input_paths);
  goal_handle->succeed(std::make_shared<MultiChompOptimize::Result>(result));
  cleanup_state();
}