Always 

"""
source /opt/ros/humble/setup.bash 
"""

Link Foundation pose int o the src/sam2_foundation/sam2_foundation service

Change model path in sam2_foundation action and service

Change prompt in pose realtime pipeline
build with
colcon build --symlink-install

# in one terminal
ros2 launch realsense2_camera rs_launch.py depth_module.depth_profile:=848,480,15 rgb_camera.color_profile:=848,480,15 align_depth.enable:=true pointcloud.enable:=true enable_sync:=true

# another one
ros2 run sam2_foundation segmentation_pose_estimation_action_server 
# main loop
ros2 run pose_realtime_pipeline client 

# another one
ros2 run sam2_foundation tracking_service


