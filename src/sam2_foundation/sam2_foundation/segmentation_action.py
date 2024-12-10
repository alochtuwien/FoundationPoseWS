from pose_estimation_interfaces.action import SegmentUsingSamEstimatePoseUsingFoundation
from PIL import Image
from cv_bridge import CvBridge
from rclpy.action import ActionServer


import rclpy
from rclpy.node import Node
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
segmentation_dir = os.path.join(current_dir, 'FoundationPose')
sys.path.append(segmentation_dir)

current_dir = os.path.dirname(os.path.abspath(__file__))
segmentation_dir = os.path.join(current_dir, 'FoundationPose', "segmentation")
sys.path.append(segmentation_dir)

import segmentation_foundationpose as seg

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


"""
This script is the action server for the segmentation and pose estimation of objects using the SAM2 model and the FoundationPose model.
1. The action server receives an image and a prompt from the client.
2. The action server segments the object in the image using the Florence2 +SAM2 model from the segmentation folder.
3. The action server estimates the pose of the object using the FoundationPose model from the FoundationPose folder.
4. The action server returns the segmentation masks and the poses of the object.
5. The action server publishes feedback to the client.
6. The action server sends a result to the client.
7. The action server succeeds.
"""


class Sam2SegmentAndEstimatePoseActionServer(Node):

    def __init__(self):
        
        super().__init__('sam2_segmentation_action_server')
        
        self._action_server = ActionServer(self, SegmentUsingSamEstimatePoseUsingFoundation, 'segmentation_and_pose_estimation_action_server', self.segment)
        
        self.scorer = ScorePredictor()

        self.refiner = PoseRefinePredictor()
        
        self.mesh = trimesh.load("/home/hoanghuy/master_thesis/FoundationPose/data_real/mesh/LegoBlock.stl")
        
        diameter = np.linalg.norm(self.mesh.extents * 2)
        # self.mesh.apply_scale(0.01)
        if diameter > 100: # object is in mm
            self.mesh.apply_scale(0.001)
        glctx = dr.RasterizeCudaContext()

        self.debug_dir = "/home/ws/src/sam2_foundation/sam2_foundation/FoundationPose/debug"
        self.est = FoundationPoseMulti(model_pts=self.mesh.vertices, model_normals=self.mesh.vertex_normals, mesh=self.mesh, scorer=self.scorer, refiner=self.refiner, debug_dir=self.debug_dir, debug=0, glctx=glctx)
        
        self.get_logger().info("Pose estimation models loaded")

    def segment(self, goal_handle):       
        # Get the image from the request and convert it to a PIL image
        image = Image.frombytes('RGB', (goal_handle.request.width, goal_handle.request.height), goal_handle.request.img.data)
        
        feedback_msg = SegmentUsingSamEstimatePoseUsingFoundation.Feedback()
        feedback_msg.done = False
        goal_handle.publish_feedback(feedback_msg)
        text_prompt = goal_handle.request.prompt

        masks, class_names = seg.open_vocabulary_detection_and_segmentation(image_input=image, text_input=text_prompt)

        result = SegmentUsingSamEstimatePoseUsingFoundation.Result()
        
        if len(masks) == 0:
            result.success = True
            
            feedback_msg.done = True
            goal_handle.publish_feedback(feedback_msg)
            goal_handle.succeed()
            return result
        
        # get intrinsics
        pose_msg = None
        with torch.autocast(device_type="cuda", dtype=torch.float):
            intrinsics = goal_handle.request.intrinsics
            intrinsics = np.array(intrinsics.data).reshape((3, 3))
            
            depth_array = np.array(CvBridge().imgmsg_to_cv2(goal_handle.request.depth_img, desired_encoding="passthrough"))
            
    
            depth_array = np.array(depth_array, dtype=np.float32)
            image = np.array(image, dtype=np.float32)
            
            # scale the depth image to meters
            depth_array = depth_array / 1000.0
            
            
            # torch.autocast(device_type="cuda", dtype=torch.float).__enter__()
            mask = masks[0].astype(bool)
            
            image = np.array(image)
            pose, pose_not_transformed = self.est.register_multi_return_tmp(K=intrinsics, rgb=image, depth=depth_array, ob_mask=mask, iteration=5)
            pose_msg = TransformStamped()
            pose_not_transformed_msg = TransformStamped()
            pose_not_transformed = pose_not_transformed.cpu().numpy()
            if image is not None and depth_array is not None and not np.array_equal(pose, np.zeros((4,4))):
                pose_msg.transform.translation.x = float(pose[0, 3])
                pose_msg.transform.translation.y = float(pose[1, 3])
                pose_msg.transform.translation.z = float(pose[2, 3])
                
                

                # Assuming you have a function to convert the rotation matrix to a quaternion
                q = rotation_matrix_to_quaternion(pose[:3, :3])
                
                # pose_msg.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
                pose_msg.transform.rotation.x = q[0]
                pose_msg.transform.rotation.y = q[1]
                pose_msg.transform.rotation.z = q[2]
                pose_msg.transform.rotation.w = q[3]

                pose_msg.header = Header()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.header.frame_id = "camera_color_frame"  
                pose_msg.child_frame_id = "object"
                
                
                
                pose_not_transformed_msg.transform.translation.x = float(pose_not_transformed[0, 3])
                pose_not_transformed_msg.transform.translation.y = float(pose_not_transformed[1, 3])
                pose_not_transformed_msg.transform.translation.z = float(pose_not_transformed[2, 3])
                
                q = rotation_matrix_to_quaternion(pose_not_transformed[:3, :3])
                
                # pose_msg.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
                pose_not_transformed_msg.transform.rotation.x = q[0]
                pose_not_transformed_msg.transform.rotation.y = q[1]
                pose_not_transformed_msg.transform.rotation.z = q[2]
                pose_not_transformed_msg.transform.rotation.w = q[3]
                
                pose_not_transformed_msg.header = Header()
                pose_not_transformed_msg.header.stamp = goal_handle.request.img.header.stamp
                pose_not_transformed_msg.header.frame_id = "camera_color_frame"
                pose_not_transformed_msg.child_frame_id = "object_not_transformed"

                
                
        masks = [CvBridge().cv2_to_imgmsg(mask) for mask in masks]
        result.masks = masks
        if pose_msg is not None:
            result.poses = [pose_msg, pose_not_transformed_msg]
            result.success = True
        else:
            result.success = False
            result.poses = []
            
        feedback_msg.done = True
        goal_handle.publish_feedback(feedback_msg)
        goal_handle.succeed()
        return result


def main():

    rclpy.init()
    
    action_serv = Sam2SegmentAndEstimatePoseActionServer()

    rclpy.spin(action_serv)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
