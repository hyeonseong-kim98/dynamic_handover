# ROS2 Foxy dependancy
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

# UR_RTDE dependancy
import rtde_control
import rtde_receive
from gripper.rg_gripper import RGGripperClient

# Utilities
import time
import numpy as np
import sys, os, pickle
import matplotlib.pyplot as plt
import argparse
from scipy.spatial.transform import Rotation as R
from pid import PID_ControllerClass


# Mujoco Related
sys.path.append('../package/helper/')
sys.path.append('../package/mujoco_usage/')
from mujoco_parser import MuJoCoParserClass
from transformation import *
import mujoco

class URComplianceControlClass(Node):
    def __init__(self, 
                k_p_pos         = 400.,
                k_d_pos         = 700.,
                k_p_rot         = 10.,
                k_d_rot         = 1.,
                k_p_pos_rel     = 50.,
                k_d_pos_rel     = 50.,
                k_p_rot_rel     = 2.,
                k_d_rot_rel     = 0.5,
                th_rot          = 1.,
                th_thr          = 15.,      
                save_data       = False,
                table_up        = True):
        super().__init__('UR_controller_node')

        # Parse inputs
        self.k_p_pos         = k_p_pos,
        self.k_d_pos         = k_d_pos,
        self.k_p_rot         = k_p_rot,
        self.k_d_rot         = k_d_rot,
        self.k_p_pos_rel     = k_p_pos_rel,
        self.k_d_pos_rel     = k_d_pos_rel,
        self.k_p_rot_rel     = k_p_rot_rel,
        self.k_d_rot_rel     = k_d_rot_rel,
        self.th_rot          = th_rot
        self.th_thr          = th_thr
        self.save_data       = save_data

        # Init MuJoCo
        xml_path = './data/assets/ur5e/robot_scene.xml'
        self.env = MuJoCoParserClass(name='handover',rel_xml_path=xml_path,verbose=False)
        print ("XML load done.")
        self.joint_names_ur5e = ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint']
        self.idxs_fwd_ur5e   = self.env.get_idxs_fwd(joint_names=self.joint_names_ur5e)
        self.pre_grap_q = [-0.0008276144610803726, -1.288763241176941, 2.100346390401022, -0.8569260400584717, 1.5611083507537842, 0.06798052787780762]
        self.grasp_pose = [-0.4915619127513904, -0.13498094487386175, -0.03569200506923502, 1.1375891453170213, -1.1961716287018225, -1.1697455043284068]
        self.init_joint_value = [-9.00884475e-01, -1.60826506e+00,  2.19351061e+00, -5.85258489e-01, 1.16221126e+00, 4.98549025e-06]
        if table_up:
            self.grasp_pose[2] += 0.07
        # Init UR_RTDE
        self.rtde_r = rtde_receive.RTDEReceiveInterface("192.168.0.100")
        self.rtde_c = rtde_control.RTDEControlInterface("192.168.0.100")
        # initialize robot to grab object & go to initial pose
        self.init_robot()


    def init_robot(self):
        data_range = {'k_p':[80.,140.], 'k_d':[10.,20], 'theta':[5.,20.]} # [k_p, k_d, theta]
        random_sample = np.random.choice(np.linspace(0,1,8),3)
        self.k_p_pos = data_range['k_p'][0] + (data_range['k_p'][1] - data_range['k_p'][0]) * random_sample[0]
        self.k_d_pos = data_range['k_d'][0] + (data_range['k_d'][1] - data_range['k_d'][0]) * random_sample[1]
        self.th_thr = data_range['theta'][0] + (data_range['theta'][1] - data_range['theta'][0]) * random_sample[2]
        print("Sampled Control Parameter: ",self.k_p_pos, self.k_d_pos, self.th_thr)
    
        self.target_q = np.array(self.init_joint_value).copy()
        self.env.reset()
        self.env.forward(q=self.init_joint_value, joint_idxs=self.idxs_fwd_ur5e)
    
        # Init gripper
        self.rg = RGGripperClient(gripper='rg2', ip="192.168.0.126", port="502")

        # Pregrasp pose
        self.rg.open_gripper(force=200)
        while 95 > self.rg.get_width():
            time.sleep(0.1)
        self.rtde_c.moveJ(self.pre_grap_q, 0.8, 0.8, False)
        time.sleep(0.5)
        self.rtde_c.zeroFtSensor()

        # Grasp
        self.rtde_c.moveL(self.grasp_pose, 0.25, 0.25, False)
        self.grab = True
        if self.grab:
            self.rg.move_gripper(width=200, force=200)
            time.sleep(3)
        
        # Initial pose
        self.rtde_c.moveJ(self.init_joint_value, 0.8, 0.8, False)
        time.sleep(0.5)
        self.initial_pose = np.array(self.rtde_r.getActualTCPPose())   
        self.initial_ft = np.array(self.rtde_r.getActualTCPForce())

        # Init forcemode
        self.rtde_c.forceModeSetDamping(0.0)                         # Damping
        self.rtde_c.forceModeSetGainScaling(2.0)                     # force-acceleration gain
        self.task_frame          = [0, 0, 0, 0, 0, 0]                # Force frame
        self.selection_vector    = [1, 1, 1, 1, 1, 1]                # Compliance activation (x,y,z,rx,ry,rz) / 1 for activate, 0 for deactivate
        self.wrench              = [0, 0, 0, 0, 0, 0]                # Force/torque to apply
        self.force_type          = 2                                 # No transformation of force frame
        self.limits              = [1., 1., 1., 1., 1., 1.]          # Velocity limit

        
        self.current_q = np.array(self.rtde_r.getActualQ()).copy()
        self.q_dot = np.array(self.rtde_r.getActualQd()).copy()
        self.current_pose_6d = np.zeros(6)
        p_curr, R_curr = self.env.get_pR_body("ur_tcp_link")
        rot_vec_curr = R.from_matrix(R_curr).as_rotvec()
        self.current_pose_6d[3:] = rot_vec_curr
        self.current_pose_6d[:3] = p_curr
        self.current_ft = self.rtde_r.getActualTCPForce()
        
        self.target_q = self.current_q.copy()
        self.target_pose_6d = self.current_pose_6d.copy()

        self.real_traj = []
        self.target_traj = []
        self.t_start = 0
        self.t_end = 0
        self.time_tracking = False

        # Init PID Controllers
        self.t_s = time.time()
        self.pid_pos = PID_ControllerClass(k_p=self.k_p_pos,k_d=self.k_d_pos, dim=3)
        self.pid_rot = PID_ControllerClass(k_p=self.k_p_rot, k_d=self.k_d_rot, dim=3)
        self.pid_pos_release = PID_ControllerClass(k_p=self.k_p_pos_rel, k_d=self.k_d_pos_rel, dim=3)
        self.pid_rot_release = PID_ControllerClass(k_p=self.k_p_rot_rel, k_d=self.k_d_rot_rel, dim=3)
        
        # Create Ros Node
        self.target_subscriber = self.create_subscription(Float32MultiArray,'target_q', self.target_q_callback, 10)
        self.q_publisher = self.create_publisher(Float32MultiArray, 'current_q', 10)
        self.handover_start_subscriber = self.create_subscription(Float32MultiArray, 'start', self.start_callback, 10)
        self.control_timer = self.create_timer(1./200.0, self.control_callback)
        print("Start Loop!")

    def start_callback(self, msg):
        if self.time_tracking == False:
            self.time_tracking = True
            self.t_start = time.time()

    def target_q_callback(self, msg: Float32MultiArray):
        if self.grab:
            self.target_q = np.array(msg.data)
            self.env.forward(q=self.target_q, joint_idxs=self.idxs_fwd_ur5e)
            p_trgt, R_trgt = self.env.get_pR_body("ur_tcp_link")
            rot_vec_curr = R.from_matrix(R_trgt).as_rotvec()
            self.target_pose_6d[:3] = p_trgt
            self.target_pose_6d[3:] = rot_vec_curr
        else:
            pass

    def control_callback(self):
        # update states
        self.t = time.time() - self.t_s
        if self.grab == False:
            self.t_release = time.time() - self.t_s_release
        self.current_q = np.array(self.rtde_r.getActualQ()).copy()
        self.q_dot = np.array(self.rtde_r.getActualQd()).copy()
        self.env.forward(q=self.current_q, joint_idxs=self.idxs_fwd_ur5e)
        p_curr, R_curr = self.env.get_pR_body("ur_tcp_link")
        rot_vec_curr = R.from_matrix(R_curr).as_rotvec()
        self.current_pose_6d[:3] = p_curr
        self.current_pose_6d[3:] = rot_vec_curr
        self.current_ft = self.rtde_r.getActualTCPForce()

        # Send current q data
        q_data = self.current_q.copy()
        msg_q = Float32MultiArray()
        msg_q.data = q_data.tolist()
        self.q_publisher.publish(msg_q)
        
        # check releasing
        if self.grab and (self.rot_force_err() > self.th_rot or self.force_err() > self.th_thr) :
            self.grab = False
            self.t_end = time.time()
            current_width = self.rg.get_width_with_offset()*10.
            self.releasing_pose = self.current_pose_6d.copy()
            target_width = int(np.clip(current_width+300, 0, 1000))
            self.rg.move_gripper(width=target_width,force=200)
            print("Gripper Opened!")
            print("Rotation force: ", self.rot_force_err(), " Thrusting force: ", self.force_err())
            print("Handover process time: ", (self.t_end - self.t_start))
            self.t_s_release = time.time()
            self.t_release = 0.
        
        # compute and do wrench
        self.compute_wrench_pid()
        self.do_wrench()
        self.real_traj.append(self.current_pose_6d.copy())
        self.target_traj.append(self.target_pose_6d.copy())

    def compute_wrench_pid(self):
        # Position error
        position_error = np.subtract(self.target_pose_6d[:3], self.current_pose_6d[:3])
        
        # Rotation error using quaternions
        r_current = R.from_rotvec(self.current_pose_6d[3:])
        r_target = R.from_rotvec(self.target_pose_6d[3:])
        r_error = r_target * r_current.inv() # Quaternion error

        # Convert quaternion error to rotation vector (axis-angle)
        rotation_error = r_error.as_rotvec()
        rotation_error = np.array(rotation_error)
        
        # Update pid and get output value
        if self.grab:
            self.pid_pos.update_with_error(self.t,position_error)
            wrench_pos = self.pid_pos.out()
            self.pid_rot.update_with_error(self.t,rotation_error)
            wrench_rot = self.pid_rot.out()
        else: # when gripper opened
            self.pid_pos_release.update_with_error(self.t_release,position_error)
            wrench_pos = self.pid_pos_release.out()
            self.pid_rot_release.update_with_error(self.t_release,rotation_error)
            wrench_rot = self.pid_rot_release.out()
            if self.t_release > 5.:
                self.rtde_c.forceModeStop()
                a = input("Do you want to restart handover? [y/n]")
                if a.lower() == 'y':
                    self.init_robot()
                else:
                    print("Robot Stopped.")
                    self.stop_robot()
                    self.destroy_node()
                    rclpy.shutdown()

        # Concat pose/rot error with gain
        wrench = np.concatenate((wrench_pos, wrench_rot))
        if self.grab:
            wrench -= self.initial_ft
        self.wrench = wrench.tolist()
    
    def do_wrench(self):
        self.rtde_c.forceMode(self.task_frame, self.selection_vector, self.wrench, self.force_type, self.limits)

    def force_err_temp(self) -> float:
        return np.linalg.norm(self.current_ft- self.initial_ft)

    def force_err(self) -> float:
        orientation_vector = self.current_pose_6d[:3] - self.env.get_p_body("ur_base")
        orientation_vector[2] = 0.
        # Convert vectors to numpy arrays
        a = (self.current_ft- self.initial_ft)[:3]
        b = orientation_vector
        
        # Calculate the dot product of a and b
        dot_product = np.dot(a, b)
        
        # Calculate the magnitude squared of vector b
        b_magnitude_squared = np.dot(b, b)
        
        # Calculate the projection of a onto b
        force_error = np.linalg.norm((dot_product / b_magnitude_squared) * b)
        
        return force_error
    
    def rot_force_err(self) -> float:
        return np.linalg.norm(self.current_ft[3:] - self.initial_ft[3:])
    
    def stop_robot(self):
        if self.save_data:
            if not os.path.exists("./data/tracking_handover"):
                os.makedirs("./data/tracking_handover")
            with open("./data/tracking_handover/real_traj.pkl", 'wb') as f:
                pickle.dump(self.real_traj, f)
            with open("./data/tracking_handover/target_traj.pkl", 'wb') as f:
                pickle.dump(self.target_traj, f)
        self.rtde_c.forceModeStop()
        self.rtde_c.stopScript()

def main(args=None):

    data_range = {'k_p':[100.,200.], 'k_d':[10.,40.], 'theta':[5.,15.]} # [k_p, k_d, theta, phi]
    random_sample = np.random.rand(4)
    p_pos = data_range['k_p'][0] + (data_range['k_p'][1] - data_range['k_p'][0]) * random_sample[0]
    d_pos = data_range['k_d'][0] + (data_range['k_d'][1] - data_range['k_d'][0]) * random_sample[0]
    th_thr = data_range['theta'][0] + (data_range['theta'][1] - data_range['theta'][0]) * random_sample[0]
    th_rot = 1.5

    rclpy.init(args=args)
    compliance_controller = URComplianceControlClass(k_p_pos = p_pos,
                                                     k_d_pos = d_pos,
                                                     k_p_rot = 2.,
                                                     k_d_rot = .5,
                                                     k_p_pos_rel = 100,
                                                     k_d_pos_rel = 50,
                                                     k_p_rot_rel = 2.,
                                                     k_d_rot_rel = .5,
                                                     th_rot = th_rot,
                                                     th_thr = th_thr,
                                                     save_data = True)

    try:
        rclpy.spin(compliance_controller)
    except KeyboardInterrupt:
        pass
    
    compliance_controller.stop_robot()
    compliance_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()