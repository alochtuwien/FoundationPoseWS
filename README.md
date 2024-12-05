Always 

"""
source /opt/ros/humble/setup.bash 
"""

Than 

# in one terminal
ros2 launch realsense2_camera rs_launch.py depth_module.depth_profile:=848,480,15 rgb_camera.color_profile:=848,480,15 align_depth.enable:=true pointcloud.enable:=true enable_sync:=true

# another one
ros2 run sam2_foundation action 
# main loop
ros2 run pose_realtime_pipeline client 

# another one
ros2 run sam2_foundation service 


