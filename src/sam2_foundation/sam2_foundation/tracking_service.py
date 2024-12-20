from pose_estimation_interfaces.msg import TrackItMsg
from pose_estimation_interfaces.msg import ResetTracking
from PIL import Image
from cv_bridge import CvBridge
from rclpy.action import ActionServer


import rclpy
from rclpy.node import Node
import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
segmentation_dir = os.path.join(current_dir, 'segmentation')
sys.path.append(segmentation_dir)

current_dir = os.path.dirname(os.path.abspath(__file__))
segmentation_dir = os.path.join(current_dir, 'FoundationPose')
sys.path.append(segmentation_dir)

from estimater import *
from datareader import *
import argparse
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point, Quaternion, TransformStamped
from std_msgs.msg import Header
import sensor_msgs
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from PIL import Image
import threading
import multiprocessing
from copy import deepcopy
from scipy.optimize import minimize
import message_filters
# from transformers import Owlv2Processor, Owlv2ForObjectDetection
from transformers import AutoProcessor, AutoModelForCausalLM 
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import time

"""
This script defines a ROS2 node node that subscribes to a topic for pose tracking using an initial object pose and depth image data.
It provides real-time tracking of a 3D object, leveraging depth data and mesh-based pose estimation.
1. The node subscribes to a topic for pose tracking.
2. The node receives an initial object pose and depth image data.
3. The node tracks the object in the image data
4. The node publishes the object pose.

"""

class Foundation_Tracking_Subcriber(Node):

    def __init__(self):
        
        super().__init__('pose_tracking_service')
        self.subscription = self.create_subscription(TrackItMsg, 'track_it', self.track, 10)
        self.reset_service = self.create_subscription(ResetTracking, 'reset_tracking', self.reset, 10)
        
        
        self.scorer = ScorePredictor()

        self.refiner = PoseRefinePredictor()
        
        self.mesh = trimesh.load("/home/hoanghuy/master_thesis/FoundationPose/data_real/mesh/LegoBlock.stl")
        self.to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)

        self.bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
        
        diameter = np.linalg.norm(self.mesh.extents * 2)
        # print(diameter)
        # self.mesh.apply_scale(0.01)
        if diameter > 100: # object is in mm
            self.mesh.apply_scale(0.001)
        # print(diameter)
        glctx = dr.RasterizeCudaContext()

        self.debug_dir = "/home/ws/src/sam2_foundation/sam2_foundation/FoundationPose/debug"
        self.est = FoundationPoseMulti(model_pts=self.mesh.vertices, model_normals=self.mesh.vertex_normals, mesh=self.mesh, scorer=self.scorer, refiner=self.refiner, debug_dir=self.debug_dir, debug=0, glctx=glctx)
        self.get_logger().info("Segmentation service initialised")
        self.initial_pose = None
        self.tf_broadcaster = TransformBroadcaster(self)
        torch.autocast(device_type="cuda", dtype=torch.float).__enter__()
        self.done = False
        self.new_frame = False
        self.intrinsics = None
        self.frame_timestamp = None
        self.waiting_for_new = True
        self.bridge = CvBridge()

    def reset(self, msg):
        print("restarted tracking")
        self.initial_pose = None
        self.waiting_for_new = True
        self.new_msg = True
        self.subscription = self.destroy_subscription(self.subscription)
        topic_name = msg.topic_name
        self.subscription = self.create_subscription(TrackItMsg, topic_name, self.track, 10)

    def track(self, msg):
        if self.waiting_for_new:
            
            print("Resetting tracking")
            translation = np.array([0.0, 0.0, 0.0])
            rotation = np.array([0.0, 0.0, 0.0, 1.0])
            translation[0] = msg.initial_pose.transform.translation.x
            translation[1] = msg.initial_pose.transform.translation.y
            translation[2] = msg.initial_pose.transform.translation.z
            
            rotation[0] = msg.initial_pose.transform.rotation.x
            rotation[1] = msg.initial_pose.transform.rotation.y
            rotation[2] = msg.initial_pose.transform.rotation.z
            rotation[3] = msg.initial_pose.transform.rotation.w
            
            # convert to rotation matrix
            rotation_matrix = R.from_quat(rotation).as_matrix()
            self.initial_pose = np.eye(4)
            self.initial_pose[:3, :3] = rotation_matrix
            self.initial_pose[:3, 3] = translation
            self.est.pose_last[0] = torch.from_numpy(self.initial_pose).float().cuda()
            self.waiting_for_new = False
            
        if not self.waiting_for_new: 
            print("Tracking")
            self.frame_timestamp = msg.img.header.stamp
            image = self.bridge.imgmsg_to_cv2(msg.img, 'bgr8')#Image.frombytes('RGB', (msg.img.width, msg.img.height), msg.img.data)
            initial_pose = msg.initial_pose  
            image = np.array(image)
            intrinsics = msg.intrinsics
            self.intrinsics = np.array(intrinsics.data).reshape((3, 3))
            
            depth_array = np.array(CvBridge().imgmsg_to_cv2(msg.depth_img, desired_encoding="passthrough"))
            

            depth_array = np.array(depth_array, dtype=np.float32)
            self.image = np.array(image, dtype=np.float32)
            
            self.depth_array = depth_array / 1000.0
            self.new_frame = True
        else:
            self.new_frame = False

        
    def track_spinner(self):
        start = time.time()
        pose = self.est.track_one_multi(rgb=self.image, depth=self.depth_array, K=self.intrinsics, iteration=10, ob_id=0)
        print(f"Time taken to track: {time.time() - start}")
        
        center_pose = pose@np.linalg.inv(self.to_origin)
        vis = draw_posed_3d_box(self.intrinsics, img=self.image, ob_in_cam=center_pose, bbox=self.bbox)
        vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=self.intrinsics, thickness=2, transparency=0, is_input_rgb=True)
        cv2.imshow('Result', vis) 
        cv2.waitKey(1)
        
        pose_msg = TransformStamped()
        pose_msg.header = Header()
        pose_msg.header.stamp = self.frame_timestamp
        pose_msg.header.frame_id = "camera_color_frame"
        pose_msg.child_frame_id = "object_tracked"
        
        pose_msg.transform.translation.x = float(pose[0, 3])
        pose_msg.transform.translation.y = float(pose[1, 3])
        pose_msg.transform.translation.z = float(pose[2, 3])
        
        q = rotation_matrix_to_quaternion(pose[:3, :3])
        
        pose_msg.transform.rotation.x = q[0]
        pose_msg.transform.rotation.y = q[1]
        pose_msg.transform.rotation.z = q[2]
        pose_msg.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(pose_msg)
            
            

def main():

    rclpy.init()
    
    seg_service = Foundation_Tracking_Subcriber()
    while rclpy.ok():
        rclpy.spin_once(seg_service)
        if not seg_service.waiting_for_new:
            if seg_service.new_frame:
                seg_service.track_spinner()
                seg_service.new_frame = False
            else:
                time.sleep(0.0001)

    rclpy.shutdown()


if __name__ == '__main__':
    main()