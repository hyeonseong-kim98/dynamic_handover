#ifndef DYN_HO_CTL_HPP_
#define DYN_HO_CTL_HPP_

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <ur_rtde/rtde_control_interface.h>
#include <ur_rtde/rtde_receive_interface.h>
#include <Eigen/Dense>
#include <Eigen/Geometry> 
#include <memory>
#include <vector>

// local includes
#include "dyn_ho_ctl/pd_controller.hpp"
#include "rg_gripper_client.hpp"

class DynamicHandoverController : public rclcpp::Node {
public:
    DynamicHandoverController(
        const std::string& robot_ip,
        const std::string& gripper_ip,
        int gripper_port,
        double k_p_pos = 100.0,
        double k_d_pos = 10.0,
        double k_p_rot = 10.0,
        double k_d_rot = 1.0,
        double k_p_pos_rel = 50.0,
        double k_d_pos_rel = 10.0,
        double k_p_rot_rel = 2.0,
        double k_d_rot_rel = 0.5,
        double th_rot = 3.0,
        double th_thr = 15.0
    );

    ~DynamicHandoverController();

private:
    // RTDE interfaces
    std::unique_ptr<ur_rtde::RTDEControlInterface> rtde_control_;
    std::unique_ptr<ur_rtde::RTDEReceiveInterface> rtde_receive_;

    // Gripper interface
    std::unique_ptr<RGGripperClient> gripper_;

    // ROS2 communication
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr target_sub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr current_q_pub_;
    rclcpp::TimerBase::SharedPtr control_timer_;

    // PD controllers
    std::unique_ptr<PDController> pd_pos_;
    std::unique_ptr<PDController> pd_rot_;
    std::unique_ptr<PDController> pd_pos_rel_;
    std::unique_ptr<PDController> pd_rot_rel_;

    // robot state 
    std::vector<double> tcp_offset_;
    std::vector<double> current_q_;
    std::vector<double> target_q_;
    std::vector<double> current_ee_pose_;
    std::vector<double> target_ee_pose_;
    std::vector<double> initial_ft_;
    std::vector<double> current_ft_;
    bool release_;

    // force mode parameters
    std::vector<double> task_frame_;
    std::vector<int> selection_vector_;
    std::vector<double> wrench_;
    int force_type_;
    std::vector<double> limits_;

    // parameters
    double k_p_pos_, k_d_pos_;
    double k_p_rot_, k_d_rot_;
    double k_p_pos_rel_, k_d_pos_rel_;
    double k_p_rot_rel_, k_d_rot_rel_;
    double th_rot_, th_thr_;

    // callbacks
    void target_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
    void control_loop();

    // helper functions
    void init_robot();
    void compute_wrench(); 
    void apply_wrench();
    double compute_pulling_force();
    double compute_torque_sum();
    void stop_robot();
};

#endif // DYN_HO_CTL_HPP_