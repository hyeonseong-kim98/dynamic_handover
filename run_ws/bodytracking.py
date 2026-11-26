import sys
import cv2
import numpy as np
import argparse
import collections
import pickle, json
from scipy.spatial.transform import Rotation as R

# Zed Related
import pyzed.sl as sl
sys.path.append('../package/zed_cv_viewer/')
import tracking_viewer as cv_viewer
from kalman_filter import KalmanFilter3D
sys.path.append('../package/helper/')
from transformation import *

# ROS2 Foxy
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

def parse_args(init):
    if len(opt.input_svo_file)>0 and opt.input_svo_file.endswith(".svo"):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input: {0}".format(opt.input_svo_file))
    elif len(opt.ip_address)>0 :
        ip_str = opt.ip_address
        if ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4 and len(ip_str.split(':'))==2:
            init.set_from_stream(ip_str.split(':')[0],int(ip_str.split(':')[1]))
            print("[Sample] Using Stream input, IP : ",ip_str)
        elif ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4:
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP : ",ip_str)
        else :
            print("Unvalid IP format. Using live stream")
    if ("HD2K" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD2K
        print("[Sample] Using Camera in resolution HD2K")
    elif ("HD1200" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1200
        print("[Sample] Using Camera in resolution HD1200")
    elif ("HD1080" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1080
        print("[Sample] Using Camera in resolution HD1080")
    elif ("HD720" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD720
        print("[Sample] Using Camera in resolution HD720")
    elif ("SVGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.SVGA
        print("[Sample] Using Camera in resolution SVGA")
    elif ("VGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.VGA
        print("[Sample] Using Camera in resolution VGA")
    elif len(opt.resolution)>0: 
        print("[Sample] No valid resolution entered. Using default")
    else : 
        print("[Sample] Using default resolution")

class HumanPosePublisher(Node):
    def __init__(self):
        super().__init__('human_pose_publisher')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'human_pose', 10)

    def pub_human_pose(self,human_pose_list):
        msg = Float32MultiArray()
        if len(human_pose_list) == 0:
            human_pose_list = [float('nan')]
        else:
            human_pose_list = human_pose_list.reshape(-1)
            human_pose_list = human_pose_list.tolist()
        msg.data = human_pose_list
        self.publisher_.publish(msg)


def main(args=None):

    print("Initializing Human pose Publisher...")
    rclpy.init(args=args)
    human_pose_publisher = HumanPosePublisher()

    print("Running Body Tracking... Press 'q' to quit, or 'm' to pause or restart")

    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    
    parse_args(init_params)

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Enable Positional tracking (mandatory for object detection)
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances
    positional_tracking_parameters.set_as_static = True

    # Read camera extrinsic data
    camera_pose_path = "./data/camera_pose/camera_transformation.json"
    with open(camera_pose_path, 'r') as f:
        camera_pose = json.load(f)
    camera_t_vector = np.array(camera_pose['camera_translation_vector'])
    camera_R_matrix = np.array(camera_pose['camera_rotation_matrix'])
    camera_H = np.eye(4)
    camera_H[:3,:3] = camera_R_matrix; camera_H[:3,3] = camera_t_vector
    print("Extrinsic Matrics: \n", camera_H)

    c_translation = sl.Translation()
    c_translation.init_vector(camera_t_vector[0],camera_t_vector[1],camera_t_vector[2])
    
    c_rotation = sl.Rotation()
    rotation_vector = R.from_matrix(camera_R_matrix).as_rotvec()
    c_rotation.set_rotation_vector(rotation_vector[0],rotation_vector[1],rotation_vector[2])

    c_transform = sl.Transform()
    c_transform.set_translation(c_translation)
    c_transform.set_rotation_matrix(c_rotation)
    camera_pose = sl.Pose()
    positional_tracking_parameters.set_initial_world_transform(c_transform)
    zed.enable_positional_tracking(positional_tracking_parameters)
    
    # Turn on positional tracking
    zed.enable_positional_tracking(positional_tracking_parameters)
    
    body_param = sl.BodyTrackingParameters()
    body_param.enable_tracking = True                # Track people across images flow
    body_param.enable_body_fitting = True            # Smooth skeleton move
    body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE 
    body_param.body_format = sl.BODY_FORMAT.BODY_34  # Choose the BODY_FORMAT you wish to use

    # Enable Object Detection module
    zed.enable_body_tracking(body_param)

    body_runtime_param = sl.BodyTrackingRuntimeParameters()
    body_runtime_param.detection_confidence_threshold = 60

    # Get ZED camera information
    camera_info = zed.get_camera_information()
    # 2D viewer utilities
    display_resolution = sl.Resolution(min(camera_info.camera_configuration.resolution.width, 640), min(camera_info.camera_configuration.resolution.height, 320))
    image_scale = [display_resolution.width / camera_info.camera_configuration.resolution.width
                 , display_resolution.height / camera_info.camera_configuration.resolution.height]
    
    # Create ZED objects filled in the main loop
    bodies = sl.Bodies()
    image = sl.Mat()
    key_wait = 10 

    point_cloud = sl.Mat()
    runtime_param = sl.RuntimeParameters()
    runtime_param.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD


    while True:
        # Grab an image
        if zed.grab(runtime_param) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            # Retrieve bodies
            zed.retrieve_bodies(bodies, body_runtime_param)
            # Retrieve point cloud
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            # Camera_pose from WORLD Coordinate
            zed.get_position(camera_pose, sl.REFERENCE_FRAME.WORLD)
            camera_pose_now = camera_pose.pose_data().m.copy()
            # Update OCV view
            image_left_ocv = image.get_data()
            cv_viewer.render_2D(image_left_ocv,image_scale, bodies.body_list, body_param.enable_tracking, body_param.body_format)

            image_left_ocv_rot = cv2.rotate(image_left_ocv, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imshow("ZED | 2D View", image_left_ocv_rot)

            key = cv2.waitKey(key_wait)
            # get clicked pixel
            if key == 113: # for 'q' key
                print("Exiting...")
                break
            if key == 109: # for 'm' key
                if (key_wait>0):
                    print("Pause")
                    key_wait = 0 
                else : 
                    print("Restart")
                    key_wait = 10 
            
            if len(bodies.body_list)>0:
                human_pose_list = []
                for i in range(len(bodies.body_list)):
                    human_id = np.array(bodies.body_list[i].id)
                    # if bodies.body_list[i].keypoint[0][1] > -0.9:
                    human_pose_list.append(np.array(bodies.body_list[i].keypoint))
                human_pose_list = np.array(human_pose_list)
            else:
                human_pose_list = np.array([])
            
            human_pose_publisher.pub_human_pose(human_pose_list)
    
    image.free(sl.MEM.CPU)
    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()
    cv2.destroyAllWindows()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file, if you want to replay it',default = '')
    parser.add_argument('--ip_address', type=str, help='IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup', default = '')
    parser.add_argument('--resolution', type=str, help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA', default = '')
    opt = parser.parse_args()
    if len(opt.input_svo_file)>0 and len(opt.ip_address)>0:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        exit()
    main() 