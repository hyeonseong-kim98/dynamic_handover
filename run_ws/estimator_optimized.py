import rclpy
import time
import numpy as np
import sys, os
import pickle
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from rclpy.utilities import remove_ros_args
from pathlib import Path
from estimator_options import parse_estimator_args


# Mujoco Related
REPOSITORY_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(REPOSITORY_DIR / 'package' / 'helper'))
sys.path.append(str(REPOSITORY_DIR / 'package' / 'mujoco_usage'))



from mujoco_parser import MuJoCoParserClass
from transformation import *
from mujoco_parser import solve_ik
import mujoco

# SMFM Related
from SparseIAAL.sparse_IAAL_te import SparseIAALClass

class URTargetPublisher(Node):

    def __init__(   self, 
                    train_data_path, 
                    hz                 =   30,
                    hz_obs             =   None,
                    sum_sample         =   10,
                    kernel_data_path   =   None,
                    kappa              =   1e-4,
                    use_gpr_whole      =   False,
                    forecasting_time   =   0.,
                    max_data_num       =   None,
                    sliding_window     =   None,
                    inducing_rate      =   0.4,
                    render             =   True,
                    render_full        =   False,
                    use_te             =   False,
                    chunking_size      =   30,
                    param_te           =   0.1,
                    render_hz          =   30,
                    motion_type        =   "normal"):
        
        super().__init__('UR_Controller')

        self.hz = hz
        self.render_hz = render_hz
        self.human_pose_list = np.array([])
        self.receiver_pose = np.array([])
        self.receiver_base_pose_xy = np.array([])
        self.receiver_base_traj_3d = [] # Observed path
        self.receiver_right_wrist = np.array([])
        self.gripper_pose = np.array([])
        self.start_move_flag = False
        self.stop_move_flag = False

        # Init mujoco
        xml_path = str(REPOSITORY_DIR / 'run_ws/data/assets/ur5e/handover_scene.xml')
        self.env = MuJoCoParserClass(name='handover',rel_xml_path=xml_path,verbose=False)
        print ("XML load done.")
        self.joint_names_ur5e = ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint']
        self.idxs_fwd_ur5e   = self.env.get_idxs_fwd(joint_names=self.joint_names_ur5e)
        self.init_joint_value = [-9.85592696e-01, -1.63374962e+00,  2.22087543e+00, -5.87249344e-01, 1.13764959e+00,  5.30542348e-05]
        self.current_q = np.array(self.init_joint_value).copy()
        self.q_dot = np.zeros(6)
        self.target_q = np.array(self.init_joint_value).copy()
        self.env.reset()
        self.env.forward(q=self.init_joint_value, joint_idxs=self.idxs_fwd_ur5e)
        self.target_pose = self.env.get_p_body("ur_tcp_link")

        # Init IAAL
        print("Now preparing estimator...")
        self.IAAL = SparseIAALClass(filepath            =   train_data_path, 
                                    hz                  =   self.hz,
                                    hz_obs              =   hz_obs,
                                    kernel_data_path    =   kernel_data_path,
                                    sum_sample          =   sum_sample,
                                    kappa               =   kappa,
                                    use_gpr_whole       =   use_gpr_whole,
                                    forecasting_time    =   forecasting_time,
                                    max_data_num        =   max_data_num,
                                    sliding_window      =   sliding_window,
                                    inducing_rate       =   inducing_rate,
                                    use_te              =   use_te,
                                    chunking_size       =   chunking_size,
                                    param_te            =   param_te,
                                    motion_type         =   motion_type)
        self.estimated_pose = self.IAAL.init_pos
        print("Estimator prepared.")
        self.time_tracking = False

        # Init Visualizer
        self.render = render
        self.render_full = render_full
        if render:
            self.env.init_viewer()
        if render_full:
            self.train_object_pose = []
            self.train_receiver_pose = []
            self.motion_length = []
            self.load_data(train_data_path)

        # Create nodes
        self.target_publisher_ = self.create_publisher(Float32MultiArray, 'target_q', 10)
        self.human_pose_subscriber_ = self.create_subscription(Float32MultiArray, 'human_pose', self.sub_human_pose_callback,10)
        self.q_data_subscriber = self.create_subscription(Float32MultiArray, 'current_q', self.get_current_q, 10)
        self.forecasting_subscriber = self.create_subscription(Float32MultiArray, 'forecasting_time', self.sub_forecasting_time_callback, 10)
        self.render_timer = self.create_timer(1./self.render_hz, self.render_callback)
        self.start_publisher = self.create_publisher(Float32MultiArray, 'start', 10)
        self.t_s = time.time()
    


    def get_current_q(self, msg):
        q_data = np.array(msg.data)
        self.current_q = q_data[0:6]
        
    def get_gripper_pose(self, msg):
        # update reciever_pose
        self.gripper_pose = np.array(msg.data)

    def sub_human_pose_callback(self, msg):
        # update reciever_pose
        self.human_pose_processing(msg.data)

        # The robot will move just for one time
        if len(self.receiver_base_pose_xy) > 0:
            if self.receiver_base_pose_xy[0] > 2.8:
                self.start_move_flag = True
            if self.start_move_flag and self.receiver_base_pose_xy[0] < -0.8:
                self.stop_move_flag = True

        if len(self.receiver_base_pose_xy) == 0:
            self.IAAL.reset_obs()
        elif self.start_move_flag and not self.stop_move_flag:
            self.estimated_pose = self.IAAL.estimate_target(self.receiver_base_pose_xy)
            self.publish_target_pose()
            if self.time_tracking == False and self.receiver_base_pose_xy[0] < 2.324965:
                self.time_tracking = True
                msg = Float32MultiArray()
                msg.data = [1.0]
                self.start_publisher.publish(msg)

    def sub_forecasting_time_callback(self, msg):
        self.human_pose_list = np.array([])
        self.receiver_pose = np.array([])
        self.receiver_base_pose_xy = np.array([])
        self.receiver_base_traj_3d = [] # Observed path
        self.receiver_right_wrist = np.array([])
        self.gripper_pose = np.array([])
        self.start_move_flag = False
        self.stop_move_flag = False
        self.IAAL.forecasting_step = int(msg.data[0] * self.IAAL.hz)
        self.current_q = self.init_joint_value.copy()
        print("Forecasting Step: ", self.IAAL.forecasting_step)
        self.IAAL.reset_obs()
        print("Estimator is initialized")

    def publish_target_pose(self):  
        print("Estimated Object Position: ",self.estimated_pose)
        # Get target pose (estimated object position -> robot tcp pose)
        estimated_pose = self.estimated_pose.copy()
        """ forward to object pose & orientation from base vector """
        orientation_vector = estimated_pose - self.env.get_p_body("ur_base")
        R_trgt = rpy2r([0, 0,np.arctan2(orientation_vector[1], orientation_vector[0])-np.pi/2]) @ rpy2r([0,np.pi,0])
        p_trgt = estimated_pose + np.array([0,0,-0.2])
        self.target_pose = p_trgt.copy()
        
        # Clip p_trgt for stable control # max_extension = 0.8, min_extension_xy = 0.7, min_height = 1.06
        extension = np.linalg.norm(p_trgt - self.env.get_p_body("ur_shoulder_link"))
        extension_xy = np.linalg.norm(p_trgt[0:2] - self.env.get_p_body("ur_shoulder_link")[0:2])
        if extension > 0.75:
            p_trgt = self.env.get_p_body("ur_shoulder_link") + (p_trgt - self.env.get_p_body("ur_shoulder_link")) / extension * 0.75
        elif extension_xy < 0.6:
            p_trgt[0:2] = self.env.get_p_body("ur_shoulder_link")[0:2] + (p_trgt - self.env.get_p_body("ur_shoulder_link"))[0:2] / extension_xy * 0.6
        if p_trgt[2] < 1.06:
            p_trgt[2] = 1.06

        # Solve IK    
        # q_init =  self.env.get_qpos_joints(self.joint_names_ur5e)
        q_init = self.current_q.copy()
        self.target_q, ik_error_stack,_ = solve_ik(
                            env                = self.env,
                            joint_names_for_ik = self.joint_names_ur5e,
                            body_name_trgt     = 'ur_tcp_link',
                            q_init             = q_init,
                            p_trgt             = p_trgt,
                            R_trgt             = R_trgt,
                        )
        
        if np.linalg.norm(ik_error_stack) > 0.05: # Stop loop when IK is too unstable
            print("IK error is too high!")
            # self.env.close_viewer()
            # self.destroy_node()
        msg = Float32MultiArray()
        msg.data = self.target_q.tolist()           
        if self.start_move_flag and not self.stop_move_flag:
            self.target_publisher_.publish(msg)
    
    def human_pose_processing(self, human_pose_list):
        """
            Use only base pose and right wrist
        """
        human_pose_list = np.array(human_pose_list)
        if len(human_pose_list) <= 1: # Clean pose
            self.human_pose_list = np.array([])
            self.receiver_pose = np.array([])
            self.receiver_base_pose_xy = np.array([])
            self.receiver_base_traj_3d = []
            self.receiver_right_wrist = np.array([])
        else: 
            self.human_pose_list = human_pose_list.reshape(-1,34,3)
            # Select largest x base pose
            self.receiver_pose = self.human_pose_list[np.argmax(self.human_pose_list[:,0,0])].copy()
            self.receiver_base_pose_xy = self.receiver_pose[0,:2].copy()
            self.receiver_base_traj_3d.append(self.receiver_pose[0].copy())
            self.receiver_right_wrist = self.receiver_pose[14].copy()
    
    def render_callback(self):
        if self.render:
            self.env.data.qpos[:6] = self.current_q.copy()
            self.env.forward(q=self.current_q, joint_idxs=self.idxs_fwd_ur5e)
            # plot estimated pose
            self.env.plot_sphere(self.target_pose, r=0.05, rgba=[1, 0, 0, 1])
            # plot receiver pose
            if not len(self.receiver_pose) == 0:
                for joint in self.receiver_pose:
                    self.env.plot_sphere(p=joint, r=0.02, rgba=(1,0,0,1))
            if not len(self.gripper_pose) < 2:
                self.env.plot_sphere(p=self.gripper_pose, r=0.02, rgba=(0, 0, 1, 1))
            # Render every information
            if self.render_full and len(self.IAAL.traj_obs) > 1:
                sliding_window = self.IAAL.sliding_window
                if sliding_window is None: sliding_window = len(self.IAAL.traj_obs)
                trajectory_idx_selected = self.IAAL.d_ascending_list_obs[-1][:self.IAAL.sum_sample]
                target_pose_idx = self.IAAL.target_idx_list_obs[-1]
                
                # plot obeserve trajectory
                if sliding_window > len(self.receiver_base_traj_3d):
                    self.env.plot_traj(np.array(self.receiver_base_traj_3d[::5]), rgba=(0, 0, 1, 1), plot_line=False, plot_cylinder=True, cylinder_r=0.01)
                else:
                    self.env.plot_traj(np.array(self.receiver_base_traj_3d[-sliding_window::5]), rgba=(0, 0, 1, 1), plot_line=False, plot_cylinder=True, cylinder_r=0.01)

                # plot selected trajectories that is used to weighted summation
                for i, traj_idx in enumerate(trajectory_idx_selected):
                    pose_idx = target_pose_idx[i]
                    self.env.plot_traj(self.train_receiver_pose[traj_idx][pose_idx::5], rgba=(1, 0, 0, 1), plot_line=False, 
                                       plot_cylinder=True, cylinder_r=0.01)
                    if sliding_window > pose_idx:
                        self.env.plot_traj(self.train_receiver_pose[traj_idx][:pose_idx:5], rgba=(0, 1, 0, 1), plot_line=False, 
                                            plot_cylinder=True, cylinder_r=0.01)
                    else:
                        self.env.plot_traj(self.train_receiver_pose[traj_idx][pose_idx-sliding_window:pose_idx:5], rgba=(0, 1, 0, 1), plot_line=False, 
                                            plot_cylinder=True, cylinder_r=0.01)
                    # Choosen pose index
                    self.env.plot_sphere(self.train_object_pose[traj_idx][pose_idx], r=0.02, rgba=[0, 1, 0, 1])
                    self.env.plot_sphere(self.train_receiver_pose[traj_idx][pose_idx], r=0.02, rgba=[0, 1, 0, 1])
                
            self.env.render()
        else:
            pass

    def load_data(self, filepath):

        # Load data
        # Use the same filtered order as the flow models, including max_data_num.
        filenames = self.IAAL.motion_files

        for i in range(len(filenames)):
            with open(self.IAAL.filepath / filenames[i], 'rb') as f:
                motion_data = pickle.load(f)
                self.motion_length.append(len(motion_data['pose_object']))
                self.train_object_pose.append(np.array(motion_data['pose_object']).copy())
                self.train_receiver_pose.append(np.array(motion_data['pose_receiver'])[:,0].copy()) # use only base_pose xy
        self.N = len(self.motion_length)
            
        print('Data ready for visualization'.format(self.N))
        

def main(args=None):
    argv = sys.argv if args is None else [sys.argv[0]] + list(args)
    options = parse_estimator_args(remove_ros_args(args=argv)[1:], render_default=True)
    forecasting_time = 0.14
    rclpy.init(args=args)
    target_pose_publisher = URTargetPublisher(
        train_data_path=options.motion_dir,
        hz=30,
        hz_obs=None,
        kernel_data_path=options.kernel_cache,
        sum_sample=10,
        kappa=1e-4,
        use_gpr_whole=False,
        forecasting_time=forecasting_time,
        max_data_num=options.max_data_num,
        sliding_window=30,
        inducing_rate=0.4,
        use_te=True,
        chunking_size=30,
        param_te=0.1,
        render=options.render,
        render_full=False,
        motion_type=options.motion_type,
    )

    try:
        rclpy.spin(target_pose_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        if options.render:
            target_pose_publisher.env.close_viewer()
        target_pose_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
