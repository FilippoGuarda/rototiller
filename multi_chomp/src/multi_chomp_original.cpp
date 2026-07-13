#include "multi_chomp/multi_chomp_original.hpp"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using std::placeholders::_1;

// ═══════════════════════════════════════════════════════════════════════════
//  Construction
// ═══════════════════════════════════════════════════════════════════════════

MultiChompOriginalNode::MultiChompOriginalNode()
: Node("multi_chomp_original_server")
{
    load_parameters();

    rclcpp::QoS map_qos(1);
    map_qos.transient_local();
    grid_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
        "robot1/global_costmap/costmap", map_qos,
        std::bind(&MultiChompOriginalNode::map_callback, this, _1));

    RCLCPP_INFO(this->get_logger(),
        "MultiChompOriginal server ready (num_robots=%d, nq=%d)",
        params_.num_robots, params_.waypoints_per_robot);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Parameter loading
// ═══════════════════════════════════════════════════════════════════════════

void MultiChompOriginalNode::load_parameters()
{
    this->declare_parameter("num_robots",           2);
    this->declare_parameter("waypoints_per_robot",  20);
    this->declare_parameter("dt",                   1.0);
    this->declare_parameter("eta",                  100.0);
    this->declare_parameter("lambda",               1.0);
    this->declare_parameter("mu",                   0.4);
    this->declare_parameter("obstacle_gain",        2.0);
    this->declare_parameter("obstacle_max_dist",    4.0);
    this->declare_parameter("inter_gain",           1.0);
    this->declare_parameter("inter_max_dist",       3.0);

    params_.num_robots          = this->get_parameter("num_robots").as_int();
    params_.waypoints_per_robot = this->get_parameter("waypoints_per_robot").as_int();
    params_.dt                  = this->get_parameter("dt").as_double();
    params_.eta                 = this->get_parameter("eta").as_double();
    params_.lambda              = this->get_parameter("lambda").as_double();
    params_.mu                  = this->get_parameter("mu").as_double();
    params_.obstacle_gain       = this->get_parameter("obstacle_gain").as_double();
    params_.obstacle_max_dist   = this->get_parameter("obstacle_max_dist").as_double();
    params_.inter_gain          = this->get_parameter("inter_gain").as_double();
    params_.inter_max_dist      = this->get_parameter("inter_max_dist").as_double();

    xidim_ = static_cast<size_t>(params_.num_robots)
            * static_cast<size_t>(params_.waypoints_per_robot)
            * cdim_;
    xi_ = VectorXd::Zero(xidim_);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Matrix initialisation — mirrors pp2d init_chomp()
//  AAR is a block-diagonal matrix with one AA block per robot.
//  AA encodes the discrete-time acceleration norm (smoothness metric).
// ═══════════════════════════════════════════════════════════════════════════

void MultiChompOriginalNode::init_matrices()
{
    const size_t nq  = static_cast<size_t>(params_.waypoints_per_robot);
    const double dt  = params_.dt;
    const size_t blk = nq * cdim_;

    // Single-robot smoothness matrix AA (tridiagonal, same as pp2d)
    MatrixXd AA = MatrixXd::Zero(blk, blk);
    for (size_t i = 0; i < nq; ++i) {
        AA.block(cdim_ * i, cdim_ * i, cdim_, cdim_) =
            2.0 * MatrixXd::Identity(cdim_, cdim_);
        if (i > 0) {
            AA.block(cdim_ * (i-1), cdim_ * i, cdim_, cdim_) =
                -1.0 * MatrixXd::Identity(cdim_, cdim_);
            AA.block(cdim_ * i, cdim_ * (i-1), cdim_, cdim_) =
                -1.0 * MatrixXd::Identity(cdim_, cdim_);
        }
    }

    // Scale (pp2d: AAR /= dt*dt*(nq+1))
    AA /= dt * dt * static_cast<double>(nq + 1);

    // Build block-diagonal AAR for all robots
    AAR_    = MatrixXd::Zero(xidim_, xidim_);
    AARinv_ = MatrixXd::Zero(xidim_, xidim_);
    for (int r = 0; r < params_.num_robots; ++r) {
        size_t off = static_cast<size_t>(r) * blk;
        AAR_.block(off, off, blk, blk) = AA;
    }
    AARinv_ = AAR_.inverse();

    bbR_ = VectorXd::Zero(xidim_);
}

// ═══════════════════════════════════════════════════════════════════════════
//  set_paths — called once per planning request.
//  Fixes start/goal and initialises xi by linear interpolation (same as pp2d
//  which initialises all waypoints to the start point; linear interp is
//  strictly better as a warm start and is still faithful to the algorithm).
// ═══════════════════════════════════════════════════════════════════════════

bool MultiChompOriginalNode::set_paths(
    const std::vector<nav_msgs::msg::Path> & paths)
{
    const int nq = params_.waypoints_per_robot;

    if (static_cast<int>(paths.size()) != params_.num_robots) {
        RCLCPP_ERROR(this->get_logger(), "set_paths: path count mismatch");
        return false;
    }

    start_states_.resize(params_.num_robots);
    goal_states_.resize(params_.num_robots);
    xidim_ = static_cast<size_t>(params_.num_robots) * nq * cdim_;
    xi_    = VectorXd::Zero(xidim_);

    for (int r = 0; r < params_.num_robots; ++r) {
        const auto & path = paths[r];
        if (path.poses.size() < 2) {
            RCLCPP_ERROR(this->get_logger(),
                "set_paths: robot %d path has < 2 poses", r);
            return false;
        }

        Vector2d qs(path.poses.front().pose.position.x,
                    path.poses.front().pose.position.y);
        Vector2d qe(path.poses.back().pose.position.x,
                    path.poses.back().pose.position.y);

        start_states_[r] = qs;
        goal_states_[r]  = qe;

        // Initialise waypoints by linear interpolation between start and goal
        // (pp2d uses all-start; linear interp reduces initial smoothness cost
        //  but is otherwise algorithmically equivalent)
        size_t off = static_cast<size_t>(r) * nq * cdim_;
        auto samples = resample_path(path, nq);
        if (samples.size() != static_cast<size_t>(nq)) {
            return false;
        }

        for (int k = 0; k < nq; ++k) {
            xi_.block(off + k * cdim_, 0, cdim_, 1) = samples[k];
        }
    }

    init_matrices();
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
//  solve_step — direct port of pp2d cb_idle()
//  All three gradient terms: smoothness, obstacle, interference.
//  Start and goal are FIXED (no sliding window).
// ═══════════════════════════════════════════════════════════════════════════

void MultiChompOriginalNode::solve_step()
{
    std::lock_guard<std::mutex> lock(map_mutex_);
    if (!map_received_) return;

    const int    nq  = params_.waypoints_per_robot;
    const double dt  = params_.dt;
    const int    NR  = params_.num_robots;

    // ── bbR: acceleration bias, recomputed from fixed start/goal ────────────
    //  pp2d: bbR(0)          = start_r / (-dt*dt*(nq+1))
    //        bbR(blk-cdim)   = goal_r  / (-dt*dt*(nq+1))
    bbR_ = VectorXd::Zero(xidim_);
    double scale = -1.0 / (dt * dt * static_cast<double>(nq + 1));
    for (int r = 0; r < NR; ++r) {
        size_t off = static_cast<size_t>(r) * nq * cdim_;
        bbR_.block(off,                  0, cdim_, 1) = scale * start_states_[r];
        bbR_.block(off + (nq-1)*cdim_,   0, cdim_, 1) = scale * goal_states_[r];
    }

    // ── Smoothness gradient: nabla_smooth = AAR * xi + bbR ──────────────────
    VectorXd nabla_smooth = AAR_ * xi_ + bbR_;
    VectorXd const & xidd = nabla_smooth;   // same quantity, pp2d notation

    // ── Obstacle and interference gradients ─────────────────────────────────
    VectorXd nabla_obs   = VectorXd::Zero(xidim_);
    VectorXd nabla_int   = VectorXd::Zero(xidim_);

    const Eigen::Matrix2d JJ = Eigen::Matrix2d::Identity();  // planar: C==W

    for (int r = 0; r < NR; ++r) {
        size_t off = static_cast<size_t>(r) * nq * cdim_;

        for (int iq = 0; iq < nq; ++iq) {
            size_t idx = off + iq * cdim_;
            Vector2d qq = xi_.block(idx, 0, cdim_, 1);

            // ── Velocity qd (central difference, boundary handled) ───────────
            Vector2d qd;
            if (iq == 0) {
                qd = (xi_.block(off + cdim_, 0, cdim_, 1) - start_states_[r])
                     / (2.0 * dt);
            } else if (iq == nq - 1) {
                qd = (goal_states_[r] - xi_.block(off + (nq-2)*cdim_, 0, cdim_, 1))
                     / (2.0 * dt);
            } else {
                qd = (xi_.block(idx + cdim_, 0, cdim_, 1)
                    - xi_.block(idx - cdim_, 0, cdim_, 1))
                     / (2.0 * dt);
            }

            double vel = qd.norm();
            if (vel < 1.0e-3) continue;   // avoid division by zero

            Vector2d xdn = qd / vel;
            Vector2d xdd = JJ * xidd.block(idx, 0, cdim_, 1);
            Eigen::Matrix2d prj = Eigen::Matrix2d::Identity()
                                  - xdn * xdn.transpose();
            Vector2d kappa = prj * xdd / (vel * vel);

            // ── Interference objective (all robot pairs sharing waypoint iq) ─
            // Mirrors pp2d: compare robot r waypoint iq against every other
            // robot s waypoint iq.  nabla_int is antisymmetric: +grad on r,
            // -grad on s.
            for (int s = 0; s < NR; ++s) {
                if (s == r) continue;
                size_t idx_s = static_cast<size_t>(s) * nq * cdim_ + iq * cdim_;
                Vector2d dd = qq - xi_.block(idx_s, 0, cdim_, 1);
                double   dn = dd.norm();

                if (dn < 1.0e-9 || dn >= params_.inter_max_dist) continue;

                double cost_r = params_.inter_gain * params_.inter_max_dist
                                * std::pow(1.0 - dn / params_.inter_max_dist, 3.0)
                                / 3.0;
                Vector2d grad_dd = dd * (-params_.inter_gain
                                         * std::pow(1.0 - dn / params_.inter_max_dist, 2.0)
                                         / dn);

                Vector2d contrib = JJ.transpose() * vel
                                   * (prj * grad_dd - cost_r * kappa);
                nabla_int.block(idx,   0, cdim_, 1) += contrib;
                nabla_int.block(idx_s, 0, cdim_, 1) -= contrib;
            }

            // ── Obstacle objective ──────────────────────────────────────────
            Eigen::Vector2d env_grad;
            double signed_dist = get_environment_cost(qq.x(), qq.y(), env_grad);

            // clearance to obstacle boundary minus robot radius
            double clearance = signed_dist - params_.robot_radius;

            if (clearance < params_.obstacle_max_dist) {
                double s = 1.0 - (clearance / params_.obstacle_max_dist);
                s = std::clamp(s, 0.0, 2.0);

                double c = params_.obstacle_gain * params_.obstacle_max_dist
                        * std::pow(s, 3.0) / 3.0;

                Vector2d delta = env_grad * (-params_.obstacle_gain * std::pow(s, 2.0));

                nabla_obs.block(idx, 0, cdim_, 1) +=
                    JJ.transpose() * vel * (prj * delta - c * kappa);
            }
        }
    }

    // ── Update rule (pp2d: xi -= AARinv*(nabla_obs + lambda*nabla_smooth + mu*nabla_int) / eta) ─
    VectorXd dxi = AARinv_ * (nabla_obs
                              + params_.lambda * nabla_smooth
                              + params_.mu    * nabla_int);
    xi_ -= dxi / params_.eta;

    // ── Clamp endpoints to start/goal (maintain boundary conditions) ─────────
    for (int r = 0; r < NR; ++r) {
        // NOTE: pp2d does NOT hard-clamp; the bbR term enforces boundaries
        // softly through the smoothness gradient.  We match that behaviour.
        (void)r;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  get_paths — convert xi back to nav_msgs::Path
// ═══════════════════════════════════════════════════════════════════════════

std::vector<nav_msgs::msg::Path>
MultiChompOriginalNode::get_paths() const
{
    const int nq = params_.waypoints_per_robot;
    std::vector<nav_msgs::msg::Path> out(params_.num_robots);

    for (int r = 0; r < params_.num_robots; ++r) {
        nav_msgs::msg::Path & path = out[r];
        path.header.frame_id = "map";
        path.header.stamp    = this->now();
        path.poses.resize(nq + 2);

        // Prepend fixed start
        geometry_msgs::msg::PoseStamped ps_start;
        ps_start.header = path.header;
        ps_start.pose.position.x  = start_states_[r].x();
        ps_start.pose.position.y  = start_states_[r].y();
        ps_start.pose.orientation.w = 1.0;
        path.poses[0] = ps_start;

        size_t off = static_cast<size_t>(r) * nq * cdim_;
        for (int k = 0; k < nq; ++k) {
            Vector2d p = xi_.block(off + k * cdim_, 0, cdim_, 1);
            double yaw = 0.0;
            if (k < nq - 1) {
                Vector2d p_next = xi_.block(off + (k+1)*cdim_, 0, cdim_, 1);
                yaw = std::atan2(p_next.y() - p.y(), p_next.x() - p.x());
            } else {
                yaw = std::atan2(goal_states_[r].y() - p.y(),
                                 goal_states_[r].x() - p.x());
            }
            geometry_msgs::msg::PoseStamped ps;
            ps.header = path.header;
            ps.pose.position.x  = p.x();
            ps.pose.position.y  = p.y();
            ps.pose.orientation.z = std::sin(yaw * 0.5);
            ps.pose.orientation.w = std::cos(yaw * 0.5);
            path.poses[k + 1] = ps;
        }

        // Append fixed goal
        geometry_msgs::msg::PoseStamped ps_goal;
        ps_goal.header = path.header;
        ps_goal.pose.position.x  = goal_states_[r].x();
        ps_goal.pose.position.y  = goal_states_[r].y();
        ps_goal.pose.orientation.w = 1.0;
        path.poses[nq + 1] = ps_goal;
    }
    return out;
}

// ═══════════════════════════════════════════════════════════════════════════
//  compute_current_cost — used by action server convergence check
// ═══════════════════════════════════════════════════════════════════════════

double MultiChompOriginalNode::compute_current_cost() const
{
    std::lock_guard<std::mutex> lock(map_mutex_);
    if (!map_received_) return 1e9;

    const int    nq  = params_.waypoints_per_robot;
    const double dt  = params_.dt;
    double total = 0.0;

    for (int r = 0; r < params_.num_robots; ++r) {
        size_t off = static_cast<size_t>(r) * nq * cdim_;
        for (int iq = 0; iq < nq; ++iq) {
            size_t idx = off + iq * cdim_;
            Vector2d qq = xi_.block(idx, 0, cdim_, 1);
            Vector2d qd;
            if (iq == 0) {
                qd = (xi_.block(off + cdim_, 0, cdim_, 1) - start_states_[r]) / (2.0*dt);
            } else if (iq == nq-1) {
                qd = (goal_states_[r] - xi_.block(off+(nq-2)*cdim_, 0, cdim_, 1)) / (2.0*dt);
            } else {
                qd = (xi_.block(idx+cdim_,0,cdim_,1) - xi_.block(idx-cdim_,0,cdim_,1)) / (2.0*dt);
            }
            double vel = qd.norm();
            if (vel < 1e-3) continue;

            Eigen::Vector2d env_grad;
            double dist = get_environment_cost(qq.x(), qq.y(), env_grad);
            double clearance = dist - params_.robot_radius;
            if (clearance < params_.obstacle_max_dist) {
                double s = 1.0 - (clearance / params_.obstacle_max_dist);
                s = std::clamp(s, 0.0, 2.0);
                double c = params_.obstacle_gain * params_.obstacle_max_dist
                        * std::pow(s, 3.0) / 3.0;
                total += vel * c;
            }
        }
    }
    return total;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Costmap callbacks and distance field (identical to multi_chomp.cpp)
// ═══════════════════════════════════════════════════════════════════════════

void MultiChompOriginalNode::map_callback(
    const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(map_mutex_);
    current_map_ = *msg;
    update_distance_map(*msg);
    map_received_ = true;
}

void MultiChompOriginalNode::update_distance_map(
    const nav_msgs::msg::OccupancyGrid & grid)
{
    map_resolution_ = grid.info.resolution;
    map_origin_x_   = grid.info.origin.position.x;
    map_origin_y_   = grid.info.origin.position.y;
    map_width_      = static_cast<int>(grid.info.width);
    map_height_     = static_cast<int>(grid.info.height);

    cv::Mat free_mask(map_height_, map_width_, CV_8UC1);
    cv::Mat occ_mask(map_height_, map_width_, CV_8UC1);

    for (int i = 0; i < map_height_; ++i) {
        int ros_row = (map_height_ - 1) - i;
        for (int j = 0; j < map_width_; ++j) {
            int8_t val = grid.data[ros_row * map_width_ + j];
            bool is_obstacle = (val < 0) || (val >= 50);

            free_mask.at<unsigned char>(i, j) = is_obstacle ? 0 : 255;
            occ_mask.at<unsigned char>(i, j)  = is_obstacle ? 255 : 0;
        }
    }

    cv::Mat dist_out_pixels, dist_in_pixels;
    cv::distanceTransform(free_mask, dist_out_pixels, cv::DIST_L2, 5);
    cv::distanceTransform(occ_mask,  dist_in_pixels,  cv::DIST_L2, 5);

    cv::Mat dist_out_m, dist_in_m;
    dist_out_pixels.convertTo(dist_out_m, CV_64F, map_resolution_);
    dist_in_pixels.convertTo(dist_in_m,   CV_64F, map_resolution_);

    // Positive outside obstacles, negative inside obstacles
    dist_map_ = dist_out_m - dist_in_m;

    cv::Sobel(dist_map_, dist_grad_x_, CV_64F, 1, 0, 3,
              1.0 / (8.0 * map_resolution_));
    cv::Sobel(dist_map_, dist_grad_y_, CV_64F, 0, 1, 3,
              1.0 / (8.0 * map_resolution_));

    dist_grad_y_ = -dist_grad_y_;
}

double MultiChompOriginalNode::get_environment_cost(
    double x, double y, Eigen::Vector2d & gradient) const
{
    if (dist_map_.empty()) {
        gradient << 0.0, 0.0;
        return 0.0;
    }

    const double min_x = map_origin_x_ + map_resolution_;
    const double max_x = map_origin_x_ + (map_width_  - 2) * map_resolution_;
    const double min_y = map_origin_y_ + map_resolution_;
    const double max_y = map_origin_y_ + (map_height_ - 2) * map_resolution_;

    // Outside map: return zero distance and a gradient pushing inward
    if (x < min_x || x > max_x || y < min_y || y > max_y) {
        double gx = 0.0, gy = 0.0;
        if (x < min_x) gx =  1.0;
        else if (x > max_x) gx = -1.0;
        if (y < min_y) gy =  1.0;
        else if (y > max_y) gy = -1.0;

        double n = std::sqrt(gx * gx + gy * gy);
        if (n > 1e-9) gradient << gx / n, gy / n;
        else          gradient << 0.0, 0.0;

        return 0.0;
    }

    double gx_c = (x - map_origin_x_) / map_resolution_;
    double gy_c = static_cast<double>(map_height_ - 1)
                - (y - map_origin_y_) / map_resolution_;

    int u = std::clamp(static_cast<int>(gx_c), 0, map_width_  - 1);
    int v = std::clamp(static_cast<int>(gy_c), 0, map_height_ - 1);

    double dist = dist_map_.at<double>(v, u);
    double dx   = dist_grad_x_.at<double>(v, u);
    double dy   = dist_grad_y_.at<double>(v, u);

    double nm = std::sqrt(dx * dx + dy * dy);
    if (nm > 1e-6) gradient << dx / nm, dy / nm;
    else           gradient << 0.0, 0.0;

    return dist;
}


std::vector<Eigen::Vector2d> MultiChompOriginalNode::resample_path(
    const nav_msgs::msg::Path & path, int num_points) const
{
    std::vector<Eigen::Vector2d> out;
    out.reserve(num_points);

    const size_t n = path.poses.size();
    if (n == 0 || num_points <= 0) {
        return out;
    }

    if (n == 1) {
        Eigen::Vector2d p(path.poses[0].pose.position.x,
                          path.poses[0].pose.position.y);
        out.assign(num_points, p);
        return out;
    }

    std::vector<double> s(n, 0.0);
    for (size_t i = 1; i < n; ++i) {
        const auto & p0 = path.poses[i - 1].pose.position;
        const auto & p1 = path.poses[i].pose.position;
        s[i] = s[i - 1] + std::hypot(p1.x - p0.x, p1.y - p0.y);
    }

    const double L = s.back();
    if (L < 1e-9) {
        Eigen::Vector2d p(path.poses[0].pose.position.x,
                          path.poses[0].pose.position.y);
        out.assign(num_points, p);
        return out;
    }

    for (int k = 0; k < num_points; ++k) {
        double target_s;
        if (num_points == 1) {
            target_s = 0.0;
        } else if (k == num_points - 1) {
            target_s = L;
        } else {
            target_s = (static_cast<double>(k) / static_cast<double>(num_points - 1)) * L;
        }

        size_t idx = 1;
        while (idx < n && s[idx] < target_s) {
            ++idx;
        }

        if (idx >= n) {
            idx = n - 1;
        }

        size_t prev_idx = (idx == 0) ? 0 : idx - 1;
        double ds = s[idx] - s[prev_idx];
        double alpha = (ds > 1e-9)
            ? std::clamp((target_s - s[prev_idx]) / ds, 0.0, 1.0)
            : 0.0;

        const auto & p0 = path.poses[prev_idx].pose.position;
        const auto & p1 = path.poses[idx].pose.position;

        out.emplace_back(
            p0.x + alpha * (p1.x - p0.x),
            p0.y + alpha * (p1.y - p0.y)
        );
    }

    return out;
}