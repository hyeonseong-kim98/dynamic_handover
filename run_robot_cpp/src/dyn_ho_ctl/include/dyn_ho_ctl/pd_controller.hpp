#ifndef PD_CONTROLLER_HPP_
#define PD_CONTROLLER_HPP_

#include <Eigen/Dense>

class PDController {
public:
    PDController(double k_p, double k_d, int dim);

    // update controller with new error
    void update(double t, const Eigen::VectorXd& error);

    // get control output
    Eigen::VectorXd output() const;

    // reset controller
    void reset();

private:
    double k_p_;
    double k_d_;
    int dim_;
    double t_prev_;
    Eigen::VectorXd error_prev_;
    Eigen::VectorXd out_val_;
};

#endif