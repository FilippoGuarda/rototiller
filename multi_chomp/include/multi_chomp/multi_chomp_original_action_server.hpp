#ifndef MULTI_CHOMP__MULTI_CHOMP_ORIGINAL_ACTION_SERVER_HPP_
#define MULTI_CHOMP__MULTI_CHOMP_ORIGINAL_ACTION_SERVER_HPP_

#include <atomic>
#include <memory>
#include <vector>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

#include "multi_chomp/action/multi_chomp_optimize.hpp"
#include "nav_msgs/msg/path.hpp"

#include "multi_chomp_original.hpp"

class MultiChompOriginalActionServer : public rclcpp::Node
{
public:
    using MultiChompOptimize    = multi_chomp::action::MultiChompOptimize;
    using GoalHandleMultiChomp  = rclcpp_action::ServerGoalHandle<MultiChompOptimize>;

    explicit MultiChompOriginalActionServer(
        const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
    rclcpp_action::Server<MultiChompOptimize>::SharedPtr action_server_;
    std::shared_ptr<MultiChompOriginalNode>              optimizer_;
    std::atomic<bool>                                    is_optimizing_;

    rclcpp_action::GoalResponse handle_goal(
        const rclcpp_action::GoalUUID & uuid,
        std::shared_ptr<const MultiChompOptimize::Goal> goal);

    rclcpp_action::CancelResponse handle_cancel(
        std::shared_ptr<GoalHandleMultiChomp> goal_handle);

    void handle_accepted(
        std::shared_ptr<GoalHandleMultiChomp> goal_handle);

    void execute_goal(
        std::shared_ptr<GoalHandleMultiChomp> goal_handle);
};

#endif  // MULTI_CHOMP__MULTI_CHOMP_ORIGINAL_ACTION_SERVER_HPP_
