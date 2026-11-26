#include "dyn_ho_ctl/dyn_ho_ctl.hpp"
#include <chrono>

using namespace std::chrono_literals;

DynamicHandoverController::DynamicHandoverController(
    const std::string& robot_ip,
    const std::string& gripper_ip,
    int gripper_port,
    double k_p_pos, double k_d_pos,
    double k_p_rot, double k_d_rot,
    double k_p_pos_rel, double k_d_pos_rel,
    double k_p_rot_rel, double k_d_rot_rel,
    double th_rot, double th_thr)
    : Node("dyn_ho_ctl_node"),
    k_p_pos_(k_p_pos), k_d_pos_(k_d_pos),
    k_p_rot_(k_p_rot), k_d_rot_(k_d_rot),
    k_p_pos_rel_(k_p_pos_rel), k_d_pos_rel_(k_d_pos_rel),
    k_p_rot_rel_(k_p_rot_rel), k_d_rot_rel_(k_d_rot_rel),
    th_rot_(th_rot), th_thr_(th_thr)

{
    // init RTDE interfaces
    rtde_control_ = std::make_unique<ur_rtde::RTDEControlInterface>(robot_ip);
    rtde_receive_ = std::make_unique<ur_rtde::RTDEReceiveInterface>(robot_ip);

    // init rg gripper
    gripper_ = std::make_unique<RGGripperClient>("rg2", gripper_ip, gripper_port);

    // init PD controllers
    RCLCPP_INFO(this->get_logger(), "k_p_pos, k_d_pos: [%f, %f]", k_p_pos_, k_d_pos_);
    RCLCPP_INFO(this->get_logger(), "k_p_rot_rel, k_d_rot_rel: [%f, %f]", k_p_rot_rel_, k_d_rot_rel_);
    RCLCPP_INFO(this->get_logger(), "th_thr: [%f]", th_thr_);
    pd_pos_ = std::make_unique<PDController>(k_p_pos_, k_d_pos_, 3);
    pd_rot_ = std::make_unique<PDController>(k_p_rot_, k_d_rot_, 3);
    pd_pos_rel_ = std::make_unique<PDController>(k_p_pos_rel_, k_d_pos_rel_, 3);
    pd_rot_rel_ = std::make_unique<PDController>(k_p_rot_rel_, k_d_rot_rel_, 3);

    // init force mode parameters
    tcp_offset_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    task_frame_ = {0, 0, 0, 0, 0, 0};
    selection_vector_ = {1, 1, 1, 1, 1, 1};
    wrench_ = {0, 0, 0, 0, 0, 0};
    force_type_ = 2;
    limits_ = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    
    // init robot
    init_robot();
}

DynamicHandoverController::~DynamicHandoverController()
{
    // stop robot
    stop_robot();
}

void DynamicHandoverController::init_robot()
{
    // pre-defined q pose configuration
    std::vector<double> pre_grasp_q = {-0.0008, -1.2887, 2.1003, -0.8569, 1.5611, 0.0679};
    std::vector<double> grasp_pose = {-0.4915, -0.1349, 0.04569, 1.1375, -1.1961, -1.1697};
    std::vector<double> init_q = {-0.9008, -1.6082,  2.1935, -0.5852, 1.1622, 0.0000};

    // move to pre-grasp pose
    gripper_->openGripper();
    std::this_thread::sleep_for(std::chrono::milliseconds(500)); // Wait 0.5 seconds
    rtde_control_->moveJ(pre_grasp_q, 0.5, 0.5);
    std::this_thread::sleep_for(std::chrono::milliseconds(500)); // Wait 0.5 seconds

    // move and grasp
    rtde_control_->moveL(grasp_pose, 0.25, 0.25);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    gripper_->moveGripper(200,150);
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));

    // move to initial pose
    rtde_control_->moveJ(init_q, 0.8, 0.8);
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // get initial pose and force
    current_ee_pose_ = rtde_receive_->getActualTCPPose();
    target_q_ = rtde_receive_->getActualQ();
    target_ee_pose_ = rtde_control_->getForwardKinematics(target_q_,tcp_offset_);
    initial_ft_ = rtde_receive_->getActualTCPForce();
    release_ = false;

    // set force mode
    rtde_control_->forceModeSetDamping(0.0);
    rtde_control_->forceModeSetGainScaling(2.0);

    // init ROS2 communication
    target_sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
        "target_q", 10, 
        std::bind(&DynamicHandoverController::target_callback, this, std::placeholders::_1));
    
    current_q_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(
        "current_q", 10);

    // create control timer (200Hz)
    control_timer_ = this->create_wall_timer(
        5ms, std::bind(&DynamicHandoverController::control_loop, this));
}

void DynamicHandoverController::control_loop()
{
    // get current state
    current_q_ = rtde_receive_->getActualQ();
    current_ee_pose_ = rtde_receive_->getActualTCPPose();
    current_ft_ = rtde_receive_->getActualTCPForce();

    // compute and apply wrench
    compute_wrench();
    apply_wrench();

    // publish state
    auto msg = std_msgs::msg::Float32MultiArray();
    msg.data.assign(current_q_.begin(), current_q_.end());
    current_q_pub_->publish(msg);
}

void DynamicHandoverController::compute_wrench()
{
    // position error
    Eigen::Vector3d pos_error(
        target_ee_pose_[0] - current_ee_pose_[0],
        target_ee_pose_[1] - current_ee_pose_[1],
        target_ee_pose_[2] - current_ee_pose_[2]
    );

    // rotation error
    Eigen::Vector3d target_rotvec(target_ee_pose_[3], target_ee_pose_[4], target_ee_pose_[5]);
    Eigen::Vector3d current_rotvec(current_ee_pose_[3], current_ee_pose_[4], current_ee_pose_[5]);
    
    // convert rotation vectors to rotation matrices
    Eigen::Matrix3d R_target = Eigen::AngleAxisd(target_rotvec.norm(), target_rotvec.normalized()).toRotationMatrix();
    Eigen::Matrix3d R_current = Eigen::AngleAxisd(current_rotvec.norm(), current_rotvec.normalized()).toRotationMatrix();
    
    // compute rotation error matrix
    Eigen::Matrix3d R_error = R_target * R_current.transpose();
    
    // convert rotation error matrix to rotation vector
    Eigen::AngleAxisd angle_axis_error(R_error);
    Eigen::Vector3d rot_error = angle_axis_error.angle() * angle_axis_error.axis();

    // check releasing threshold 
    if (!release_){
        double pulling_force = compute_pulling_force();
        double torque_sum = compute_torque_sum();
        if ((pulling_force > th_thr_) ||  (torque_sum > th_rot_)){ // releasing timing
            release_ = true;
            gripper_->openGripper(300);
            RCLCPP_INFO(this->get_logger(), "Obeject released! [pulling force: %f , torque sum: %f ]", 
                pulling_force, torque_sum);  
        }
    }

    // update PD controllers
    double t = this->now().seconds();
    Eigen::Vector3d wrench_pos, wrench_rot;

    if (!release_){
        pd_pos_->update(t, pos_error);
        pd_rot_->update(t, rot_error);
        wrench_pos = pd_pos_->output();
        wrench_rot = pd_rot_->output();
    }
    else{
        pd_pos_rel_->update(t, pos_error);
        pd_rot_rel_->update(t, rot_error);
        wrench_pos = pd_pos_rel_->output();
        wrench_rot = pd_rot_rel_->output();
    }

    // combine and compensate for initial force/torque
    for (int i = 0; i <3; i++) {
        wrench_[i] = wrench_pos[i] - initial_ft_[i];
        wrench_[i+3] = wrench_rot[i] - initial_ft_[i+3];
    }
}

double DynamicHandoverController::compute_pulling_force()
{   
    // get xyz force
    Eigen::Vector3d a(current_ft_[0] - initial_ft_[0],
                      current_ft_[1] - initial_ft_[1],
                      current_ft_[2] - initial_ft_[2]);

    // calculate direction vector
    Eigen::Vector3d orientation_vector(current_ee_pose_[0], current_ee_pose_[1], 0.0);

    // calculate pulling direction force
    double dot_product = a.dot(orientation_vector);
    double pulling_force = std::abs(dot_product) / orientation_vector.norm();

    return pulling_force;
}

double DynamicHandoverController::compute_torque_sum()
{   
    Eigen::Vector3d torque_vector(current_ft_[3] - initial_ft_[3],
                                  current_ft_[4] - initial_ft_[4],
                                  current_ft_[5] - initial_ft_[5]);
    double torque_sum = torque_vector.norm();

    return torque_sum;
}

void DynamicHandoverController::apply_wrench()
{
    rtde_control_->forceMode(task_frame_, selection_vector_, wrench_, force_type_, limits_);
}

void DynamicHandoverController::target_callback(
    const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
    if (!release_){
        target_q_.assign(msg->data.begin(), msg->data.end());
        target_ee_pose_ = rtde_control_->getForwardKinematics(target_q_,tcp_offset_);
    }
}
void DynamicHandoverController::stop_robot()
{
    // stop robot
    rtde_control_->forceModeStop();
    rtde_control_->stopScript();
}