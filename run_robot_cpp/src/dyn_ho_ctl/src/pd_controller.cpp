#include "dyn_ho_ctl/pd_controller.hpp"
#include <iostream>

PDController::PDController(double k_p, double k_d, int dim)
    : k_p_(k_p), k_d_(k_d), dim_(dim), t_prev_(-1.0) // Initialize to invalid time
{
    error_prev_ = Eigen::VectorXd::Zero(dim);
    out_val_ = Eigen::VectorXd::Zero(dim);
}

void PDController::update(double t, const Eigen::VectorXd& error)
{
    if (t_prev_ < 0) {
        // First update - initialize without derivative term
        out_val_ = k_p_ * k_p_ * error;
        error_prev_ = error;
        t_prev_ = t;
        return;
    }

    double dt = t - t_prev_;
    if (dt > 1e-6) {  // prevent division by very small numbers
        Eigen::VectorXd error_derivative = (error - error_prev_) / dt;
        out_val_ = k_p_ * error + k_d_ * error_derivative;

        // update previous values
        error_prev_ = error;
        t_prev_ = t;
    }
} 

Eigen::VectorXd PDController::output() const
{
    return out_val_;
}

void PDController::reset()
{
    t_prev_ = -1.0;
    error_prev_ = Eigen::VectorXd::Zero(dim_);
    out_val_ = Eigen::VectorXd::Zero(dim_);
}