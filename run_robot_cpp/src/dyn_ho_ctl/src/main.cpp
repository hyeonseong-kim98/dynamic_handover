#include "dyn_ho_ctl/dyn_ho_ctl.hpp"
#include <rclcpp/rclcpp.hpp>

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DynamicHandoverController>(
        "192.168.0.100", // robot ip
        "192.168.0.106", // gripper ip
        502, // gripper port
        114.3, 17.1,// 250.3, 14.143,// k_p_pos, k_d_pos
        2.0, 0.5, // k_p_rot, k_d_rot
        100.0, 50.0, // k_p_pos_rel, k_d_pos_rel
        2.0, 0.5, // k_p_rot_rel, k_d_rot_rel
        3.0, 8.143 // th_rot, th_thr
    );

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}