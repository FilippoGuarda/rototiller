#ifndef MULTI_CHOMP__MULTI_CHOMP_HPP_
#define MULTI_CHOMP__MULTI_CHOMP_HPP_

#include <vector>
#include <memory>
#include <mutex>
#include <string>
#include <cmath>
#include <Eigen/Dense>
#include <random>
#include <opencv2/opencv.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/path.hpp"

typedef Eigen::Vector2d Vector2d;
typedef Eigen::VectorXd VectorXd;
typedef Eigen::MatrixXd MatrixXd;

struct ChompParameters {
  double dt = 1.0;
  double eta = 100.0;
  double lambda = 0.01;      // MODIFIED: Reduced smoothness weight
  double mu = 0.4;
  double robot_radius = 0.5;
  double obstacle_max_dist = 50.0;
  int num_robots = 2;
  int waypoints_per_robot = 20;
  double alpha = 100.0;
};

class MultiChompNode : public rclcpp::Node {

public:

  MultiChompNode();

  void solve_step();
  int get_num_robots() const { return params_.num_robots; }

  // multi chomp accessors
  // start with imput global paths (computed with nav2)
  bool set_paths(const std::vector<nav_msgs::msg::Path> & paths);
  bool has_map() const { return map_received_; };
  double compute_current_cost() const;

  // export state as optimized path
  std::vector<nav_msgs::msg::Path> get_paths(const std::vector<nav_msgs::msg::Path> & templates) const;
  rclcpp::TimerBase::SharedPtr optim_timer_;
  void timer_callback();
  void update_starts_from_tf();
  std::vector<rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr> path_pubs_;
  int iteration_count_ = 0;

private:
  // visualization publisher
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

  // costmap subscriber
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_sub_;
  bool optimization_active_ = false;
  std::mutex trajectory_mutex_;

  // state protection lock
  mutable std::mutex map_mutex_;
  nav_msgs::msg::OccupancyGrid current_map_;
  bool map_received_ = false;

  // state
  ChompParameters params_;

  // distance field state
  cv::Mat dist_map_;
  cv::Mat dist_grad_x_;
  cv::Mat dist_grad_y_;
  float map_resolution_;
  double map_origin_x_, map_origin_y_;
  int map_width_, map_height_;
  void update_distance_map(const nav_msgs::msg::OccupancyGrid& grid);

  size_t cdim_ = 2;
  size_t xidim_;
  VectorXd xi_;  // trajectory vector
  VectorXd xi_init_;  
  MatrixXd AA_; // cost matrix    
  MatrixXd AAinv_; // cost matrix inverse
  MatrixXd AA_interior;
  MatrixXd AAinv_interior_; 
  MatrixXd AAR_; // multiple robots AA   
  MatrixXd AARinv_; // AAR inverse 
  VectorXd bbR_; // acceleration bias

  // state scenarios
  std::vector<Vector2d> start_states_;
  std::vector<Vector2d> goal_states_;
  // extend a provided path into a fixed number of waypoints
  std::vector<Vector2d> resample_path(const nav_msgs::msg::Path & path, int num_points) const;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::vector<int> freeze_indices_; 

  void map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);

  double get_environment_cost(double x, double y, Eigen::Vector2d& gradient) const;

  // helpers initialization
  void load_parameters();
  void init_matrices();
  void publish_state();
  
};


#endif //MULTI_CHOMP__MULTI_CHOMP_HPP_