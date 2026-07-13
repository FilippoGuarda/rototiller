#ifndef MULTI_CHOMP__MULTI_CHOMP_ORIGINAL_HPP_
#define MULTI_CHOMP__MULTI_CHOMP_ORIGINAL_HPP_

#include <mutex>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <cmath>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/path.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

struct ChompOriginalParameters {
    double dt         = 1.0;
    double eta        = 100.0;
    double lambda     = 1.0;   // smoothness weight  (pp2d default)
    double mu         = 0.4;   // interference weight (pp2d default)
    double obstacle_gain      = 2.0;
    double obstacle_max_dist  = 4.0;
    double inter_gain         = 1.0;
    double inter_max_dist     = 3.0;
    double robot_radius       = 0.5;
    int    num_robots         = 2;
    int    waypoints_per_robot = 20;
};

class MultiChompOriginalNode : public rclcpp::Node {
public:
    MultiChompOriginalNode();

    // Load fixed start/goal from Nav2 paths and initialise xi.
    // Returns false if any path is degenerate.
    bool set_paths(const std::vector<nav_msgs::msg::Path> & paths);

    // Execute one full CHOMP gradient step (called by action server).
    void solve_step();

    // Export current xi as nav_msgs::Path per robot.
    std::vector<nav_msgs::msg::Path> get_paths() const;

    // Accessors used by action server.
    bool has_map()      const { return map_received_; }
    int  get_num_robots() const { return params_.num_robots; }
    double compute_current_cost() const;

private:
    // ── ROS infrastructure ──────────────────────────────────────────────────
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_sub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr             path_pub_;   // debug

    void map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);

    // ── Costmap / distance field ────────────────────────────────────────────
    mutable std::mutex               map_mutex_;
    nav_msgs::msg::OccupancyGrid     current_map_;
    bool                             map_received_ = false;
    cv::Mat dist_map_, dist_grad_x_, dist_grad_y_;
    float   map_resolution_ = 0.05f;
    double  map_origin_x_ = 0.0, map_origin_y_ = 0.0;
    int     map_width_ = 0, map_height_ = 0;

    void   update_distance_map(const nav_msgs::msg::OccupancyGrid & grid);
    double get_environment_cost(double x, double y, Eigen::Vector2d & grad) const;

    // ── CHOMP state ─────────────────────────────────────────────────────────
    ChompOriginalParameters params_;

    static constexpr size_t cdim_ = 2;   // config-space dimension (planar)

    size_t              xidim_ = 0;       // total trajectory dimension
    Eigen::VectorXd     xi_;              // stacked trajectory [q1_r0 … qN_r0 | … | q1_rR … qN_rR]
    Eigen::MatrixXd     AAR_;             // block-diagonal metric
    Eigen::MatrixXd     AARinv_;          // its inverse
    Eigen::VectorXd     bbR_;             // acceleration bias (recomputed every step)

    std::vector<Eigen::Vector2d> start_states_;  // fixed at set_paths time
    std::vector<Eigen::Vector2d> goal_states_;   // fixed at set_paths time

    // ── Helpers ─────────────────────────────────────────────────────────────
    void   init_matrices();
    void   load_parameters();

    std::vector<Eigen::Vector2d> resample_path(
        const nav_msgs::msg::Path & path, int n) const;
};

#endif  // MULTI_CHOMP__MULTI_CHOMP_ORIGINAL_HPP_
