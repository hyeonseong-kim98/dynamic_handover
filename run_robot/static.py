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
from mujoco_parser import solve_ik
import mujoco

class URComplianceControlClass(Node):
    def __init__(self, table_up = True):
        super().__init__('UR_controller_node')

        # Init MuJoCo
        xml_path = './data/assets/ur5e/handover_scene.xml'
        self.env = MuJoCoParserClass(name='handover',rel_xml_path=xml_path,verbose=False)
        print ("XML load done.")
        self.joint_names_ur5e = ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint']
        self.idxs_fwd_ur5e   = self.env.get_idxs_fwd(joint_names=self.joint_names_ur5e)
        self.pre_grap_q = [-0.0008276144610803726, -1.288763241176941, 2.100346390401022, -0.8569260400584717, 1.5611083507537842, 0.06798052787780762]
        self.grasp_pose = [-0.4915619127513904, -0.13498094487386175, -0.03569200506923502, 1.1375891453170213, -1.1961716287018225, -1.1697455043284068]
        self.init_joint_value = [-9.00884475e-01, -1.60826506e+00,  2.19351061e+00, -5.85258489e-01, 1.16221126e+00, 4.98549025e-06]

        if table_up:
            self.grasp_pose[2] += 0.07
        # Init human pose
        self.receiver_base_pose_xy = np.array([])
        self.receiver_rwrist_pose = np.array([])
        self.receiver_spine  = np.array([])
        self.count = 0
        # Init UR_RTDE
        self.rtde_r = rtde_receive.RTDEReceiveInterface("192.168.0.100")
        self.rtde_c = rtde_control.RTDEControlInterface("192.168.0.100")
        # initialize robot to grab object & go to initial pose
        self.init_robot()


    def init_robot(self):
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
        
        self.current_q = np.array(self.rtde_r.getActualQ()).copy()
        self.current_pose_6d = np.zeros(6)
        p_curr, R_curr = self.env.get_pR_body("ur_tcp_link")
        rot_vec_curr = R.from_matrix(R_curr).as_rotvec()
        self.current_pose_6d[3:] = rot_vec_curr
        self.current_pose_6d[:3] = p_curr
        self.current_ft = self.rtde_r.getActualTCPForce()
        
        # Init PID Controllers
        self.t_s = time.time()
        
        # Create Ros Node
        self.human_pose_subscriber_ = self.create_subscription(Float32MultiArray, 'human_pose', self.sub_human_pose_callback,10)
        self.control_timer = self.create_timer(1./200.0, self.control_callback)
        
        self.end = False
        
        self.start_handover = False
        self.t_start = 0
        self.t_end = 0
        self.time_tracking = False
        print("Start Loop!")

    def sub_human_pose_callback(self, msg):
        # update reciever_pose
        self.human_pose_processing(msg.data)
        if len(self.receiver_base_pose_xy) != 0:
            # Check x pose in the box
            if self.start_handover == False and self.receiver_base_pose_xy[0] < 2.8:
                self.start_handover = True
            if self.receiver_base_pose_xy[0] < 2.324965 and self.time_tracking == False and self.start_handover:
                self.t_start = time.time()
                self.time_tracking = True
            if self.receiver_base_pose_xy[0] < 0.30 and self.receiver_base_pose_xy[0] > -0.17:
                # Check y pose in the box
                if self.receiver_base_pose_xy[1] < -0.36 and self.receiver_base_pose_xy[1] > -0.88:
                    # Check whether the wrist above the spine
                    if self.receiver_right_wrist[2] > self.receiver_pose_spine[2]:
                        self.count += 1 
                        print(self.count)
                        if self.count > 60: # wait 2 seconds
                            target_pose = self.receiver_right_wrist + np.array([0,-0.3,0.0])
                            orientation_vector = target_pose - self.env.get_p_body("ur_base")
                            R_trgt = rpy2r([0, 0,np.arctan2(orientation_vector[1], orientation_vector[0])-np.pi/2]) @ rpy2r([0,np.pi,0])
                            q_init = self.current_q.copy()
                            target_q, ik_error_stack,_ = solve_ik(
                                                env                = self.env,
                                                joint_names_for_ik = self.joint_names_ur5e,
                                                body_name_trgt     = 'ur_tcp_link',
                                                q_init             = q_init,
                                                p_trgt             = target_pose,
                                                R_trgt             = R_trgt,
                                            )
                            if np.linalg.norm(ik_error_stack) > 0.05: # Stop loop when IK is too unstable
                                print("IK error is too high!")
                            elif not self.end:
                                self.end = True
                                self.rtde_c.moveJ(target_q, 0.5, 0.5, False)

                            # self.rtde_c.moveL(target_pose_6d, 0.25, 0.25, False)
                    else:
                        self.count = 0
                else:
                    self.count = 0
            else:
                self.count = 0
        else:
            self.count = 0


    def human_pose_processing(self, human_pose_list):
        """
            Use base, spine, right wrist
        """
        human_pose_list = np.array(human_pose_list)
        if len(human_pose_list) <= 1: # Clean pose
            self.human_pose_list = np.array([])
            self.receiver_pose = np.array([])
            self.receiver_base_pose_xy = np.array([])
            self.receiver_pose_spine = np.array([])
            self.receiver_right_wrist = np.array([])
        else: 
            self.human_pose_list = human_pose_list.reshape(-1,34,3)
            # Select largest x base pose
            self.receiver_pose = self.human_pose_list[np.argmax(self.human_pose_list[:,0,0])].copy()
            self.receiver_base_pose_xy = self.receiver_pose[0,:2].copy()
            self.receiver_pose_spine = self.receiver_pose[1].copy()
            self.receiver_right_wrist = self.receiver_pose[14].copy()
            
    def control_callback(self):
        # update states
        self.current_ft = self.rtde_r.getActualTCPForce()
        
        # check releasing
        if self.grab and (self.rot_force_err() > 1.5 or self.force_err() > 3.) :
            self.grab = False
            self.t_end = time.time()
            current_width = self.rg.get_width_with_offset()*10.
            self.releasing_pose = self.current_pose_6d.copy()
            target_width = int(np.clip(current_width+300, 0, 1000))
            self.rg.move_gripper(width=target_width,force=200)
            print("Gripper Opened!")
            print("Rotation force: ", self.rot_force_err(), " Thrusting force: ", self.force_err())
            print("Handover process time: ", (self.t_end - self.t_start))
            self.rtde_c.forceModeStop()
            a = input("Do you want to restart handover? [y/n]")
            if a.lower() == 'y':
                self.init_robot()
            else:
                print("Robot Stopped.")
                self.stop_robot()
                self.destroy_node()
                rclpy.shutdown()


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
        self.rtde_c.stopScript()

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--p_pos", action='store', type=float, default=150)         
    parser.add_argument("--d_pos", action='store', type=float, default=25)
    parser.add_argument("--th_rot", action='store', type=float, default=1.5)      
    parser.add_argument("--th_thr", action='store', type=float, default=5.) 
    args_parsed    = parser.parse_args()    
    p_pos = args_parsed.p_pos
    d_pos = args_parsed.d_pos
    th_rot = args_parsed.th_rot
    th_thr = args_parsed.th_thr
    rclpy.init(args=args)
    compliance_controller = URComplianceControlClass()

    try:
        rclpy.spin(compliance_controller)
    except KeyboardInterrupt:
        pass
    
    compliance_controller.stop_robot()
    compliance_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()