#include "rclcpp/rclcpp.hpp"
#include "pose_estimation_interfaces/srv/extract_seg_map_with_prompt.hpp"
#include "pose_estimation_interfaces/action/segment_using_sam.hpp"
#include "pose_estimation_interfaces/msg/track_it_msg.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_ros/transform_broadcaster.h"

#include <chrono>
#include <cstdlib>
#include <memory>
using std::placeholders::_1;

using namespace std::chrono_literals;

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::CameraInfo> ImagesSyncPolicy;
typedef message_filters::Synchronizer<ImagesSyncPolicy> ImagesSync;


static const rmw_qos_profile_t rmw_qos_profile_latch =
{
  RMW_QOS_POLICY_HISTORY_KEEP_LAST,
  10,
  RMW_QOS_POLICY_RELIABILITY_RELIABLE,
  RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL,
  RMW_QOS_DEADLINE_DEFAULT,
  RMW_QOS_LIFESPAN_DEFAULT,
  RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
  RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
  false
};

class PipelineExecutionNode : public rclcpp::Node
{
public:
  PipelineExecutionNode()
  : Node("pose_estimation_pipeline")
  {
    client_ = this->create_client<pose_estimation_interfaces::srv::ExtractSegMapWithPrompt>("SegmentWPrompt");
    this->client_ptr_ = rclcpp_action::create_client<pose_estimation_interfaces::action::SegmentUsingSam>(
      this,
      "segmentation_action_server");

    image_sub = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this, "/camera/camera/color/image_raw", rmw_qos_profile_latch);
    depth_sub = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this, "/camera/camera/aligned_depth_to_color/image_raw", rmw_qos_profile_default);
    camera_info_sub = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::CameraInfo>>(this, "/camera/camera/color/camera_info", rmw_qos_profile_default);

    sync = std::make_shared<ImagesSync>(ImagesSyncPolicy(10), *image_sub, *depth_sub, *camera_info_sub);
    sync->registerCallback(&PipelineExecutionNode::runtime_loop, this);

    segmentation_map_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/segmentation_map", 10);

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

    track_it_publisher_ = this->create_publisher<pose_estimation_interfaces::msg::TrackItMsg>("/track_it", 10);   
    

  }

private:
  rclcpp_action::Client<pose_estimation_interfaces::action::SegmentUsingSam>::SharedPtr client_ptr_;
  // rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  // camera info subscriber
  // rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_subscription_;
  rclcpp::Client<pose_estimation_interfaces::srv::ExtractSegMapWithPrompt>::SharedPtr client_;
  // Create publisher for the segmentation map
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr segmentation_map_publisher_;
  // Create publisher for the tf
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  
  // create approximate time synchronizer
  std::shared_ptr<ImagesSync> sync;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> image_sub;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> depth_sub;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::CameraInfo>>camera_info_sub;
  geometry_msgs::msg::TransformStamped last_transformStamped_not_transformed;
  // create publisher for TrackItMsg
  rclcpp::Publisher<pose_estimation_interfaces::msg::TrackItMsg>::SharedPtr track_it_publisher_;


  bool _segmentation_started = false;
  bool _new_masks_available = false;
  bool _intrinsics_acquired = false;
  bool _tracking_started = false;
  bool _new_pose_estimation_available = false;

  std::vector<pose_estimation_interfaces::msg::TrackItMsg> buffor_msgs;
  
  double _intrinsics[9];
  double _distortion[5];

  // current segmentation maps
  std::vector<sensor_msgs::msg::Image> _segmentation_maps;
  // add handle to the segmentation goal
  rclcpp_action::ClientGoalHandle<pose_estimation_interfaces::action::SegmentUsingSam>::SharedPtr goal_handle_;

  void runtime_loop(const sensor_msgs::msg::Image::SharedPtr msg,
                    const sensor_msgs::msg::Image::SharedPtr depth_msg,
                    const sensor_msgs::msg::CameraInfo::SharedPtr camera_info_msg){
    std::uint64_t header_info_sec = msg->header.stamp.sec;
    std::uint64_t header_info_nsec = msg->header.stamp.nanosec/1e6;
    std::cout << "Received image with timestamp " << header_info_sec << "," << header_info_nsec << std::endl;
    if ((_new_pose_estimation_available && !_tracking_started) || _tracking_started){

      _tracking_started = true;
      auto msg_to_track = pose_estimation_interfaces::msg::TrackItMsg();
      std::array<double, 9> _intrinsics;
      std::array<double, 5> _distortion;

      for (int i = 0; i < 9; i++){
        _intrinsics[i] = camera_info_msg->k[i];
      }
      for (int i = 0; i < 5; i++){
        _distortion[i] = camera_info_msg->d[i];
      }

      msg_to_track.intrinsics = _intrinsics;
      msg_to_track.distortion = _distortion;
      msg_to_track.img = *msg;
      msg_to_track.depth_img = *depth_msg;
      msg_to_track.initial_pose = last_transformStamped_not_transformed;

      for (auto &msg_from_buffor : buffor_msgs){
        msg_from_buffor.initial_pose = last_transformStamped_not_transformed;
        track_it_publisher_->publish(msg_from_buffor);
      }

      track_it_publisher_->publish(msg_to_track);
      buffor_msgs.clear();
    }

    if (!_segmentation_started){
      buffor_msgs.clear();
      _segmentation_started = true;
      auto goal = pose_estimation_interfaces::action::SegmentUsingSam::Goal();
      goal.img = *msg;
      goal.height = msg->height;
      goal.width = msg->width;
      goal.prompt = "Please provide the segmentation map of the black clamp with the orange tip visible on the image";

      // get the camera intrinsics
      std::array<double, 9> intrinsics;
      std::array<double, 5> distortion;

      for (int i = 0; i < 9; i++){
        intrinsics[i] = camera_info_msg->k[i];
      }
      for (int i = 0; i < 5; i++){
        distortion[i] = camera_info_msg->d[i];
      }
      goal.intrinsics = intrinsics;
      goal.distortion = distortion;

      goal.depth_img = *depth_msg;

      auto send_goal_options = rclcpp_action::Client<pose_estimation_interfaces::action::SegmentUsingSam>::SendGoalOptions();
      send_goal_options.result_callback = std::bind(&PipelineExecutionNode::result_callback, this, _1);

      this->client_ptr_->async_send_goal(goal, send_goal_options);

    }
    
  }


  void result_callback(const rclcpp_action::ClientGoalHandle<pose_estimation_interfaces::action::SegmentUsingSam>::WrappedResult & result){
    RCLCPP_INFO(this->get_logger(), "Result received");
    if (result.code == rclcpp_action::ResultCode::SUCCEEDED){
      RCLCPP_INFO(this->get_logger(), "Segmentation succeeded");
      auto data = result.result;
      _segmentation_maps = data->masks;
      _new_masks_available = true;
      _segmentation_started = false;
      auto poses_estimated = data->poses;
      if (_segmentation_maps.size() > 0){
        segmentation_map_publisher_->publish(_segmentation_maps[0]);
      }
      if (poses_estimated.size() > 0){
        last_transformStamped_not_transformed = poses_estimated[1];
        tf_broadcaster_->sendTransform(poses_estimated[0]);
        _new_pose_estimation_available = true;
      }
    }
    else{
      RCLCPP_ERROR(this->get_logger(), "Segmentation failed");
    }
  }

};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  //wait for messages from the camera
  rclcpp::executors::MultiThreadedExecutor executor;
  auto node = std::make_shared<PipelineExecutionNode>();
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}