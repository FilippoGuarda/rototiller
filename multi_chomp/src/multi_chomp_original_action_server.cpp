#include "multi_chomp/multi_chomp_original_action_server.hpp"

using std::placeholders::_1;
using std::placeholders::_2;
using MultiChompOptimize   = multi_chomp::action::MultiChompOptimize;
using GoalHandleMultiChomp = rclcpp_action::ServerGoalHandle<MultiChompOptimize>;

// ═══════════════════════════════════════════════════════════════════════════
//  Constructor
// ═══════════════════════════════════════════════════════════════════════════

MultiChompOriginalActionServer::MultiChompOriginalActionServer(
    const rclcpp::NodeOptions & options)
: Node("multi_chomp_original_action_server", options),
  is_optimizing_(false)
{
    // Spin the optimizer node on its own executor so map callbacks are
    // processed independently of the action server.
    optimizer_ = std::make_shared<MultiChompOriginalNode>();
    auto exec  = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
    exec->add_node(optimizer_);
    std::thread([exec]() { exec->spin(); }).detach();

    action_server_ = rclcpp_action::create_server<MultiChompOptimize>(
        this,
        "multi_chomp_optimize",
        std::bind(&MultiChompOriginalActionServer::handle_goal,     this, _1, _2),
        std::bind(&MultiChompOriginalActionServer::handle_cancel,   this, _1),
        std::bind(&MultiChompOriginalActionServer::handle_accepted,  this, _1));

    RCLCPP_INFO(this->get_logger(),
        "MultiChompOriginal action server started (original pp2d behaviour)");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Goal handling
// ═══════════════════════════════════════════════════════════════════════════

rclcpp_action::GoalResponse
MultiChompOriginalActionServer::handle_goal(
    const rclcpp_action::GoalUUID &,
    std::shared_ptr<const MultiChompOptimize::Goal> goal)
{
    if (goal->num_robots == 0 ||
        static_cast<int>(goal->input_paths.size()) != goal->num_robots)
    {
        RCLCPP_WARN(this->get_logger(), "Rejecting goal: invalid dimensions");
        return rclcpp_action::GoalResponse::REJECT;
    }

    bool expected = false;
    if (!is_optimizing_.compare_exchange_strong(expected, true)) {
        RCLCPP_WARN(this->get_logger(),
            "Rejecting goal: optimizer busy");
        return rclcpp_action::GoalResponse::REJECT;
    }

    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse
MultiChompOriginalActionServer::handle_cancel(
    std::shared_ptr<GoalHandleMultiChomp> /*goal_handle*/)
{
    RCLCPP_INFO(this->get_logger(), "Cancel request received");
    return rclcpp_action::CancelResponse::ACCEPT;
}

void MultiChompOriginalActionServer::handle_accepted(
    std::shared_ptr<GoalHandleMultiChomp> goal_handle)
{
    std::thread(
        std::bind(&MultiChompOriginalActionServer::execute_goal, this, goal_handle)
    ).detach();
}

// ═══════════════════════════════════════════════════════════════════════════
//  execute_goal — run CHOMP to convergence, then return full optimized path.
//
//  Faithful to pp2d: the optimizer runs synchronously until it converges
//  (or hits max_iterations).  No intermediate path execution occurs.
//  The coordinator receives the result only after the full optimization.
// ═══════════════════════════════════════════════════════════════════════════

void MultiChompOriginalActionServer::execute_goal(
    std::shared_ptr<GoalHandleMultiChomp> goal_handle)
{
    const auto goal = goal_handle->get_goal();
    MultiChompOptimize::Result   result;
    MultiChompOptimize::Feedback feedback;

    auto cleanup = [this]() { is_optimizing_.store(false); };

    // ── Wait for costmap ────────────────────────────────────────────────────
    rclcpp::Rate wait_rate(1.0);
    while (rclcpp::ok() && !optimizer_->has_map()) {
        if (goal_handle->is_canceling()) {
            goal_handle->canceled(std::make_shared<MultiChompOptimize::Result>(result));
            cleanup();
            return;
        }
        RCLCPP_WARN(this->get_logger(), "Waiting for global costmap...");
        wait_rate.sleep();
    }

    // ── Load start/goal, initialise xi ─────────────────────────────────────
    if (!optimizer_->set_paths(goal->input_paths)) {
        RCLCPP_ERROR(this->get_logger(), "set_paths failed — aborting");
        goal_handle->abort(std::make_shared<MultiChompOptimize::Result>(result));
        cleanup();
        return;
    }

    // ── Run optimisation loop to convergence ────────────────────────────────
    const uint32_t max_iter       = (goal->max_iterations > 0)
                                    ? goal->max_iterations : 200u;
    const double   min_cost_delta = 1.0e-4;
    double prev_cost   = 1.0e9;
    uint32_t plateau   = 0;

    for (uint32_t iter = 0; iter < max_iter; ++iter) {
        if (goal_handle->is_canceling()) {
            goal_handle->canceled(
                std::make_shared<MultiChompOptimize::Result>(result));
            cleanup();
            return;
        }

        optimizer_->solve_step();

        double cost = optimizer_->compute_current_cost();

        // Convergence check (identical to multi_chomp_action_server.cpp)
        if (iter > 10) {
            if (std::abs(prev_cost - cost) < min_cost_delta) {
                ++plateau;
                if (plateau >= 3) {
                    RCLCPP_INFO(this->get_logger(),
                        "Converged at iteration %u (cost=%.6f)", iter, cost);
                    break;
                }
            } else {
                plateau = 0;
            }
        }
        prev_cost = cost;

        if (iter % 10 == 0) {
            feedback.progress          = static_cast<float>(iter) /
                                         static_cast<float>(max_iter);
            feedback.current_iteration = iter;
            feedback.current_cost      = cost;
            goal_handle->publish_feedback(
                std::make_shared<MultiChompOptimize::Feedback>(feedback));
        }
    }

    // ── Return fully optimised paths ────────────────────────────────────────
    result.optimized_paths = optimizer_->get_paths();
    goal_handle->succeed(
        std::make_shared<MultiChompOptimize::Result>(result));
    cleanup();
}

// ═══════════════════════════════════════════════════════════════════════════
//  main
// ═══════════════════════════════════════════════════════════════════════════

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MultiChompOriginalActionServer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
