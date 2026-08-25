#include "multi_chomp/rolling_chomp.hpp"

using std::placeholders::_1;
using Eigen::MatrixXd;
using Eigen::VectorXd;

MultiChompNode::MultiChompNode() : Node("multi_chomp_server") {
  load_parameters();
  init_matrices();

  start_states_.resize(params_.num_robots, Eigen::Vector2d::Zero());
  goal_states_.resize(params_.num_robots, Eigen::Vector2d::Zero());

  marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("plan_markers", 200);

  for(int r = 0; r < params_.num_robots; ++r) {
      path_pubs_.push_back(this->create_publisher<nav_msgs::msg::Path>(
          "robot" + std::to_string(r + 1) + "/optimized_path", 10));
  }

  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  rclcpp::QoS map_qos(1);
  map_qos.transient_local();

  grid_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
      "robot1/global_costmap/costmap", map_qos, std::bind(&MultiChompNode::map_callback, this, _1));

  iteration_count_ = 0;
  optim_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(20),
      std::bind(&MultiChompNode::timer_callback, this));

  optimization_active_ = false;

  RCLCPP_INFO(this->get_logger(), "Multi-CHOMP server initialized for %d robots", params_.num_robots);
}

void MultiChompNode::load_parameters() {
  this->declare_parameter<int>("num_robots", 6);
  this->declare_parameter<int>("waypoints_per_robot", 100);
  this->declare_parameter<double>("dt", 0.1);
  this->declare_parameter<double>("eta", 10000.0);
  this->declare_parameter<double>("lambda", 0.01);
  this->declare_parameter<double>("mu", 10.0);

  params_.robot_radius = 0.5;
  params_.obstacle_max_dist = 4.0;
  
  params_.num_robots = this->get_parameter("num_robots").as_int();
  params_.waypoints_per_robot = this->get_parameter("waypoints_per_robot").as_int();
  params_.dt = this->get_parameter("dt").as_double();
  params_.eta = this->get_parameter("eta").as_double();
  params_.lambda = this->get_parameter("lambda").as_double();
  params_.mu = this->get_parameter("mu").as_double();

  xidim_ = params_.num_robots * params_.waypoints_per_robot * cdim_;
  xi_  = VectorXd::Zero(xidim_);
}

void MultiChompNode::init_matrices() {
  size_t nq = params_.waypoints_per_robot; 
  AA_ = MatrixXd::Zero(nq * cdim_, nq * cdim_);
  
  for (size_t i=0; i < nq; ++i) { 
      AA_.block(cdim_ * i, cdim_ * i, cdim_, cdim_) = 2.0 * MatrixXd::Identity(cdim_, cdim_);
      if (i > 0) {
          AA_.block(cdim_ * (i - 1), cdim_ * i, cdim_, cdim_) = -1.0 * MatrixXd::Identity(cdim_, cdim_);
          AA_.block(cdim_ * i, cdim_ * (i - 1), cdim_, cdim_) = -1.0 * MatrixXd::Identity(cdim_, cdim_); 
      }
  }

  AAR_ = MatrixXd::Zero(xidim_, xidim_);
  for (int k = 0; k < params_.num_robots; ++k) {
      size_t offset = static_cast<size_t>(k) * nq * cdim_;
      AAR_.block(offset, offset, nq * cdim_, nq * cdim_) = AA_;
  }
  
  AARinv_ = AAR_.inverse();
}

void MultiChompNode::map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(map_mutex_);
  current_map_ = *msg;
  update_distance_map(*msg);
  map_received_ = true;
}

void MultiChompNode::update_distance_map(const nav_msgs::msg::OccupancyGrid& grid) {
  map_resolution_ = grid.info.resolution;
  map_origin_x_ = grid.info.origin.position.x;
  map_origin_y_ = grid.info.origin.position.y;
  map_width_ = grid.info.width;
  map_height_ = grid.info.height;

  cv::Mat raw_cost_map(map_height_, map_width_, CV_64FC1);

  for (int i = 0; i < map_height_; ++i) {
      int ros_row = (map_height_ - 1) - i; 
      for (int j = 0; j < map_width_; ++j) {
          int8_t val = grid.data[ros_row * map_width_ + j];
          if (val == -1) raw_cost_map.at<double>(i, j) = 50.0;
          else raw_cost_map.at<double>(i, j) = static_cast<double>(val);
      }
  }

  cv::GaussianBlur(raw_cost_map, dist_map_, cv::Size(5, 5), 1.0);
  cv::Sobel(dist_map_, dist_grad_x_, CV_64F, 1, 0, 3, 1.0 / (8.0 * map_resolution_));
  cv::Sobel(dist_map_, dist_grad_y_, CV_64F, 0, 1, 3, 1.0 / (8.0 * map_resolution_));
  dist_grad_y_ = -dist_grad_y_;
}

double MultiChompNode::get_environment_cost(double x, double y, Eigen::Vector2d& gradient) const {
  if (dist_map_.empty()) {
      gradient << 0.0, 0.0;
      return 999.0;
  }

  const double min_x = map_origin_x_ + map_resolution_;
  const double max_x = map_origin_x_ + (map_width_ - 2) * map_resolution_;
  const double min_y = map_origin_y_ + map_resolution_;
  const double max_y = map_origin_y_ + (map_height_ - 2) * map_resolution_;

  if (x < min_x || x > max_x || y < min_y || y > max_y) {
      double grad_x = 0.0, grad_y = 0.0;
      double sq_pen_dist = 0.0;

      if (x < min_x) { grad_x = 1.0; sq_pen_dist += (min_x - x) * (min_x - x); }
      else if (x > max_x) { grad_x = -1.0; sq_pen_dist += (x - max_x) * (x - max_x); }
      if (y < min_y) { grad_y = 1.0; sq_pen_dist += (min_y - y) * (min_y - y); }
      else if (y > max_y) { grad_y = -1.0; sq_pen_dist += (y - max_y) * (y - max_y); }

      double norm = std::sqrt(grad_x * grad_x + grad_y * grad_y);
      gradient << grad_x / norm, grad_y / norm;
      return -std::sqrt(sq_pen_dist);
  }

  double gx = (x - map_origin_x_) / map_resolution_;
  double gy = (double)(map_height_ - 1) - (y - map_origin_y_) / map_resolution_;

  int u = static_cast<int>(gx);
  int v = static_cast<int>(gy);

  double dist = dist_map_.at<double>(v, u);
  double dx = - dist_grad_x_.at<double>(v, u);
  double dy = - dist_grad_y_.at<double>(v, u);

  double norm = std::sqrt(dx * dx + dy * dy);
  if (norm > 1e-5) gradient << dx / norm, dy / norm;
  else gradient << 0.0, 0.0;

  return dist;
}

std::vector<nav_msgs::msg::Path> MultiChompNode::get_paths(const std::vector<nav_msgs::msg::Path> & templates) const {
  std::vector<nav_msgs::msg::Path> out;
  out.resize(params_.num_robots);
  const int nq = params_.waypoints_per_robot;

  if (xidim_ == 0 || xi_.size() != static_cast<int>(xidim_)) return out;

  for (int r = 0; r < params_.num_robots; ++r) {
      nav_msgs::msg::Path path;
      path.header.frame_id = "map";
      path.header.stamp = this->now();
      size_t robot_offset = static_cast<size_t>(r) * nq * cdim_;
      path.poses.resize(nq);

      for (int k = 0; k < nq; ++k) {
          size_t idx = robot_offset + static_cast<size_t>(k) * cdim_;
          Eigen::Vector2d p = xi_.block(idx, 0, cdim_, 1);
          double yaw = 0.0;
          Eigen::Vector2d p_next = p, p_prev = p;
          bool found_next = false, found_prev = false;

          for(int step = 1; step <= 3 && (k + step) < nq; ++step) {
              Eigen::Vector2d check = xi_.block(robot_offset + static_cast<size_t>(k + step) * cdim_, 0, cdim_, 1);
              if ((check - p).squaredNorm() > 0.01) { p_next = check; found_next = true; break; }
          }
          for(int step = 1; step <= 3 && (k - step) >= 0; ++step) {
              Eigen::Vector2d check = xi_.block(robot_offset + static_cast<size_t>(k - step) * cdim_, 0, cdim_, 1);
              if ((check - p).squaredNorm() > 0.01) { p_prev = check; found_prev = true; break; }
          }

          if (found_next && found_prev) yaw = std::atan2(p_next.y() - p_prev.y(), p_next.x() - p_prev.x());
          else if (found_next) yaw = std::atan2(p_next.y() - p.y(), p_next.x() - p.x());
          else if (found_prev) yaw = std::atan2(p.y() - p_prev.y(), p.x() - p_prev.x());
          else if (k > 0) {
              const auto& prev_q = path.poses[k-1].pose.orientation;
              yaw = std::atan2(2.0 * (prev_q.w * prev_q.z + prev_q.x * prev_q.y), 1.0 - 2.0 * (prev_q.y * prev_q.y + prev_q.z * prev_q.z));
          }

          geometry_msgs::msg::PoseStamped ps;
          ps.header = path.header;
          ps.pose.position.x = p.x();
          ps.pose.position.y = p.y();
          ps.pose.position.z = 0.0;
          ps.pose.orientation.x = 0.0;
          ps.pose.orientation.y = 0.0;
          ps.pose.orientation.z = std::sin(yaw * 0.5);
          ps.pose.orientation.w = std::cos(yaw * 0.5);
          path.poses[k] = ps;
      }
      out[r] = std::move(path);
  }
  return out;
}

std::vector<Eigen::Vector2d> MultiChompNode::resample_path(const nav_msgs::msg::Path & path, int num_points) const {
  std::vector<Eigen::Vector2d> out;
  out.reserve(num_points);
  const size_t n = path.poses.size();
  if (n == 0 || num_points <= 0) return out;
  if (n == 1) {
      out.assign(num_points, Eigen::Vector2d(path.poses[0].pose.position.x, path.poses[0].pose.position.y));
      return out;
  }

  std::vector<double> s(n, 0.0);
  for (size_t i = 1; i < n; ++i) {
      const auto & p0 = path.poses[i - 1].pose.position;
      const auto & p1 = path.poses[i].pose.position;
      s[i] = s[i - 1] + std::hypot(p1.x - p0.x, p1.y - p0.y);
  }

  double L = s.back();
  if (L < 1e-6) {
      out.assign(num_points, Eigen::Vector2d(path.poses[0].pose.position.x, path.poses[0].pose.position.y));
      return out;
  }

  for (int k = 0; k < num_points; ++k) {
      double target_s = (k == num_points - 1) ? L : (static_cast<double>(k) / (num_points - 1)) * L;
      size_t idx = 0;
      while (idx < n - 1 && s[idx] < target_s) idx++;
      if (idx == 0) { out.push_back(Eigen::Vector2d(path.poses[0].pose.position.x, path.poses[0].pose.position.y)); continue; }

      size_t prev_idx = idx - 1;
      double ds = s[idx] - s[prev_idx];
      double alpha = (ds > 1e-6) ? std::clamp((target_s - s[prev_idx]) / ds, 0.0, 1.0) : 0.0;
      const auto & p0 = path.poses[prev_idx].pose.position;
      const auto & p1 = path.poses[idx].pose.position;
      out.push_back(Eigen::Vector2d(p0.x + alpha * (p1.x - p0.x), p0.y + alpha * (p1.y - p0.y)));
  }
  return out;
}

bool MultiChompNode::set_paths(const std::vector<nav_msgs::msg::Path> & paths) {
    std::lock_guard<std::mutex> lock(trajectory_mutex_);

    int new_num_robots = static_cast<int>(paths.size());
    const int nq = params_.waypoints_per_robot;

    if (nq <= 1 || new_num_robots == 0) return false;

    bool reset_required = (new_num_robots != params_.num_robots) ||
                          (goal_states_.size() != static_cast<size_t>(new_num_robots));

    if (reset_required) {
        optimization_active_ = false; // Freeze solver
        params_.num_robots = new_num_robots;
        xidim_ = params_.num_robots * nq * cdim_;
        xi_ = VectorXd::Zero(xidim_);
        init_matrices();

        start_states_.resize(params_.num_robots, Eigen::Vector2d::Zero());
        goal_states_.resize(params_.num_robots, Eigen::Vector2d::Zero());

        path_pubs_.clear();
        for(int r = 0; r < params_.num_robots; ++r) {
            path_pubs_.push_back(this->create_publisher<nav_msgs::msg::Path>(
                "robot" + std::to_string(r + 1) + "/optimized_path", 10));
        }
    }
    for (int r = 0; r < params_.num_robots; ++r) {
        const auto & path = paths[r];

        if (path.poses.empty()) {
            if (reset_required) return false;
            continue;
        }

        if (path.poses.size() < 2) return false;

        Eigen::Vector2d new_start(path.poses.front().pose.position.x, path.poses.front().pose.position.y);
        Eigen::Vector2d new_goal(path.poses.back().pose.position.x, path.poses.back().pose.position.y);
        size_t robot_offset = static_cast<size_t>(r) * nq * cdim_;

        auto samples = resample_path(path, nq);
        if (samples.size() != static_cast<size_t>(nq)) return false;
        for (int k = 0; k < nq; ++k) xi_.block(robot_offset + k * cdim_, 0, cdim_, 1) = samples[k];
        start_states_[r] = new_start;
        goal_states_[r] = new_goal;
    }

    optimization_active_ = true; // Unlock execution
    return true;
}

void MultiChompNode::update_starts_from_tf() {
  const int nq = params_.waypoints_per_robot;

  for (int r = 0; r < params_.num_robots; ++r) {
      geometry_msgs::msg::TransformStamped tf;
      try {
          tf = tf_buffer_->lookupTransform("map", "robot" + std::to_string(r + 1) + "/base_link", tf2::TimePointZero);
      } catch (const tf2::TransformException & ex) {
          continue; 
      }

      Eigen::Vector2d current_pos(tf.transform.translation.x, tf.transform.translation.y);
      size_t offset = static_cast<size_t>(r) * nq * cdim_;

      int shift_idx = 0;
      double min_dist = std::numeric_limits<double>::infinity();

      int search_horizon = std::min(nq, params_.waypoints_per_robot); 

      for (int k = 0; k < search_horizon; ++k) {
          Eigen::Vector2d pt = xi_.block(offset + k * cdim_, 0, cdim_, 1);
          double dist = (pt - current_pos).norm();
          if (dist < min_dist) { min_dist = dist; shift_idx = k; }
      }

      if (min_dist > 0.6) {
          continue;
      }

      if (shift_idx > 0) {
          for (int k = 0; k < nq - shift_idx; ++k) {
              xi_.block(offset + k * cdim_, 0, cdim_, 1) = xi_.block(offset + (k + shift_idx) * cdim_, 0, cdim_, 1);
          }
          for (int k = nq - shift_idx; k < nq; ++k) {
              xi_.block(offset + k * cdim_, 0, cdim_, 1) = goal_states_[r];
          }
      }

      start_states_[r] = xi_.block(offset, 0, cdim_, 1);
  }
}

void MultiChompNode::timer_callback() {
  std::lock_guard<std::mutex> traj_lock(trajectory_mutex_); 
  if (!optimization_active_ || xidim_ == 0 || xi_.size() == 0) return;

  update_starts_from_tf();
  solve_step();

  iteration_count_++;
  if (iteration_count_ >= 10) {
      iteration_count_ = 0;
      std::vector<nav_msgs::msg::Path> dummy_templates; 
      auto optimized_paths = get_paths(dummy_templates);

      for (size_t r = 0; r < optimized_paths.size(); ++r) {
          if (r < path_pubs_.size() && path_pubs_[r]) {
              path_pubs_[r]->publish(optimized_paths[r]);
          }
      }
  }
}

void MultiChompNode::solve_step() {
  std::lock_guard<std::mutex> lock(map_mutex_);
  if (!map_received_) return;

  const int nq = params_.waypoints_per_robot;
  const double dt = params_.dt;
  VectorXd nabla_smooth = AAR_ * xi_;
  VectorXd nabla_obs = VectorXd::Zero(xidim_);
  VectorXd nabla_inter = VectorXd::Zero(xidim_);

  for (int r = 0; r < params_.num_robots; ++r) {
      size_t offset = static_cast<size_t>(r) * nq * cdim_;
      VectorXd start_p = start_states_[r];
      VectorXd end_p = goal_states_[r];

      for (int i = 0; i < nq; ++i) {
          int idx = offset + i * cdim_;
          VectorXd qq = xi_.block(idx, 0, cdim_, 1);

          VectorXd qd = VectorXd::Zero(cdim_);
          if (i == 0) qd = (xi_.block(idx + cdim_, 0, cdim_, 1) - start_p) / (2.0 * dt);
          else if (i == nq - 1) qd = (end_p - xi_.block(idx - cdim_, 0, cdim_, 1)) / (2.0 * dt);
          else qd = (xi_.block(idx + cdim_, 0, cdim_, 1) - xi_.block(idx - cdim_, 0, cdim_, 1)) / (2.0 * dt);

          double vel = qd.norm();
          if (vel < 1.0e-3) continue;

          VectorXd xdn = qd / vel;
          Eigen::Matrix2d prj = Eigen::Matrix2d::Identity() - xdn * xdn.transpose();
          VectorXd xdd = nabla_smooth.block(idx, 0, cdim_, 1);
          VectorXd kappa = (prj * xdd) / (vel * vel);

          Eigen::Vector2d grad_env;
          double cost_val = get_environment_cost(qq(0), qq(1), grad_env);

          double start_fade = 1.0;
          if (i < 5) start_fade = static_cast<double>(i) / 5.0;

          if (cost_val > 1.0) {
              VectorXd delta = -VectorXd(grad_env);
              nabla_obs.block(idx, 0, cdim_, 1) += start_fade * vel * (prj * delta * params_.dt * cost_val - cost_val * kappa);
          }

          const double safety_dist = 2.0 * params_.robot_radius;
          const double gainR = 1.0;

          for (int r2 = 0; r2 < params_.num_robots; ++r2) {
              if (r == r2) continue;
              size_t offset2 = static_cast<size_t>(r2) * nq * cdim_;
              VectorXd q2 = xi_.block(offset2 + i * cdim_, 0, cdim_, 1);
              VectorXd diff = qq - q2;
              double ddnorm = diff.norm();

              if (ddnorm < safety_dist && ddnorm > 1e-9) {
                  double termR = (1.0 - ddnorm / safety_dist);
                  double costR = gainR * safety_dist * std::pow(termR, 3.0) / 3.0;
                  VectorXd deltaR = -gainR * std::pow(termR, 2.0) * (diff / ddnorm);
                  nabla_inter.block(idx, 0, cdim_, 1) += vel * (prj * deltaR - costR * kappa);
              }
          }
      }
  }

  VectorXd full_grad = nabla_obs + params_.lambda * nabla_smooth + params_.mu * nabla_inter;
  double g_norm = full_grad.norm();
  if (g_norm > 1e6) full_grad *= (1e6 / g_norm);
  VectorXd dxi = AARinv_ * full_grad;

  for (int r = 0; r < params_.num_robots; ++r) {
      size_t offset = static_cast<size_t>(r) * nq * cdim_;
      dxi.block(offset, 0, cdim_, 1).setZero();
      dxi.block(offset + (nq - 1) * cdim_, 0, cdim_, 1).setZero();

      for (int k = nq - 2; k > 0; --k) {
          if ((xi_.block(offset + k * cdim_, 0, cdim_, 1) - goal_states_[r]).norm() < 1e-4) {
              dxi.block(offset + k * cdim_, 0, cdim_, 1).setZero();
          } else break; 
      }
  }

  xi_ -= dxi / params_.eta;

  for (int r = 0; r < params_.num_robots; ++r) {
      size_t offset = static_cast<size_t>(r) * nq * cdim_;
      xi_.block(offset, 0, cdim_, 1) = start_states_[r];
      xi_.block(offset + (nq-1)*cdim_, 0, cdim_, 1) = goal_states_[r];

      for (int k = nq - 2; k > 0; --k) {
          if ((xi_.block(offset + k * cdim_, 0, cdim_, 1) - goal_states_[r]).norm() < 1e-4) {
              xi_.block(offset + k * cdim_, 0, cdim_, 1) = goal_states_[r];
          } else break;
      }
  }
  publish_state();
}

double MultiChompNode::compute_current_cost() const {
  std::lock_guard<std::mutex> lock(map_mutex_);
  if (!map_received_) return 0.0;

  double total_cost = 0.0;
  VectorXd smooth_term = AAR_ * xi_;
  total_cost += params_.lambda * (0.5 * xi_.dot(smooth_term));

  const int nq = params_.waypoints_per_robot;
  double obstacle_cost = 0.0, interference_cost = 0.0;
  const double safety_dist = 2.0 * params_.robot_radius;

  for (int r1 = 0; r1 < params_.num_robots; ++r1) {
      for (int k = 1; k < nq - 1; ++k) {
          int idx1 = (r1 * nq + k) * static_cast<int>(cdim_);
          VectorXd p1 = xi_.block(idx1, 0, cdim_, 1);
          Eigen::Vector2d grad_env;
          double dist = get_environment_cost(p1(0), p1(1), grad_env);
          if (dist > 0.0) obstacle_cost += dist * dist;

          for (int r2 = r1 + 1; r2 < params_.num_robots; ++r2) {
              int idx2 = (r2 * nq + k) * static_cast<int>(cdim_);
              VectorXd p2 = xi_.block(idx2, 0, cdim_, 1);
              double r_dist = (p1 - p2).norm();
              if (r_dist < safety_dist) {
                  double violation = safety_dist - r_dist;
                  interference_cost += violation * violation;
              }
          }
      }
  }
  return total_cost + obstacle_cost + params_.mu * interference_cost;
}

void MultiChompNode::publish_state() {
  visualization_msgs::msg::MarkerArray marker_array;
  const int nq = params_.waypoints_per_robot;

  for (int r = 0; r < params_.num_robots; ++r) {
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = "map";
      marker.header.stamp = this->now();
      marker.ns = "robot_" + std::to_string(r + 1);
      marker.id = r;
      marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.scale.x = 0.05;
      marker.color.a = 1.0;
      marker.color.r = (r % 3 == 0) ? 1.0 : 0.5;
      marker.color.g = (r % 3 == 1) ? 1.0 : 0.5;
      marker.color.b = (r % 3 == 2) ? 1.0 : 0.5;

      size_t robot_offset = static_cast<size_t>(r) * nq * cdim_;
      for (int k = 0; k < nq; ++k) {
          Eigen::Vector2d pt = xi_.block(robot_offset + static_cast<size_t>(k) * cdim_, 0, cdim_, 1);
          geometry_msgs::msg::Point p;
          p.x = pt.x(); p.y = pt.y(); p.z = 0.0;
          marker.points.push_back(p);
      }
      marker_array.markers.push_back(marker);
  }
  marker_pub_->publish(marker_array);
}
