#include "rclcpp/rclcpp.hpp"
#include "pose_estimation_interfaces/srv/extract_seg_map_with_prompt.hpp"
#include "pose_estimation_interfaces/action/segment_using_sam_estimate_pose_using_foundation.hpp"
#include "pose_estimation_interfaces/msg/reset_tracking.hpp"
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
    this->client_ptr_ = rclcpp_action::create_client<pose_estimation_interfaces::action::SegmentUsingSamEstimatePoseUsingFoundation>(
      this,
      "segmentation_and_pose_estimation_action_server");

    image_sub = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this, "/camera/camera/color/image_raw", rmw_qos_profile_latch);
    depth_sub = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this, "/camera/camera/aligned_depth_to_color/image_raw", rmw_qos_profile_default);
    camera_info_sub = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::CameraInfo>>(this, "/camera/camera/color/camera_info", rmw_qos_profile_default);

    sync = std::make_shared<ImagesSync>(ImagesSyncPolicy(10), *image_sub, *depth_sub, *camera_info_sub);
    sync->registerCallback(&PipelineExecutionNode::runtime_loop, this);

    segmentation_map_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/segmentation_map", 10);

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

    track_it_publisher_ = this->create_publisher<pose_estimation_interfaces::msg::TrackItMsg>("/track_it", 10);   
    
    reset_tracking_publisher_ = this->create_publisher<pose_estimation_interfaces::msg::ResetTracking>("/reset_tracking", 10);

  }

private:
  rclcpp_action::Client<pose_estimation_interfaces::action::SegmentUsingSamEstimatePoseUsingFoundation>::SharedPtr client_ptr_;
  // rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  // camera info subscriber
  // rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_subscription_;
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
  // create client for reset tracking service
  rclcpp::Publisher<pose_estimation_interfaces::msg::ResetTracking>::SharedPtr reset_tracking_publisher_;


  bool _segmentation_started = false;
  bool _new_masks_available = false;
  bool _intrinsics_acquired = false;
  bool _tracking_started = false;
  bool _new_pose_estimation_available = false;
  bool _acquiring_buffer = false;
  bool _segmentation_requested = true;
  uint _id = 0;

  // segmentation request time stamp
  std::time_t _segmentation_last_request_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());  

  std::vector<pose_estimation_interfaces::msg::TrackItMsg> buffor_msgs;
  
  double _intrinsics[9];
  double _distortion[5];

  // current segmentation maps
  std::vector<sensor_msgs::msg::Image> _segmentation_maps;
  // add handle to the segmentation goal
  rclcpp_action::ClientGoalHandle<pose_estimation_interfaces::action::SegmentUsingSamEstimatePoseUsingFoundation>::SharedPtr goal_handle_;

  void runtime_loop(const sensor_msgs::msg::Image::SharedPtr msg,
                    const sensor_msgs::msg::Image::SharedPtr depth_msg,
                    const sensor_msgs::msg::CameraInfo::SharedPtr camera_info_msg){
    // std::uint64_t header_info_sec = msg->header.stamp.sec;
    // std::uint64_t header_info_nsec = msg->header.stamp.nanosec/1e6;
    // std::cout << "Received image with timestamp " << header_info_sec << "," << header_info_nsec << std::endl;

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
    msg_to_track.reset_tracking = false;

    if (_tracking_started){
      track_it_publisher_->publish(msg_to_track);
    }

    if ((_new_pose_estimation_available)){

      _tracking_started = true;
      _new_pose_estimation_available = false;

      if (buffor_msgs.size() > 0){
        buffor_msgs.front().reset_tracking = true;
      }
      else
      {
        msg_to_track.reset_tracking = true;
      }

      // publish reset tracking message
      auto msg_reset_tracking = pose_estimation_interfaces::msg::ResetTracking();
      msg_reset_tracking.reset = true;
      // generate _id
      _id += 1;
      msg_reset_tracking.topic_name = "/track_it" + std::to_string(_id);
      // msg_reset_tracking.topic_name = "/track_it";
      reset_tracking_publisher_->publish(msg_reset_tracking);

      track_it_publisher_ = this->create_publisher<pose_estimation_interfaces::msg::TrackItMsg>(msg_reset_tracking.topic_name, 10);

      for (auto &msg_from_buffor : buffor_msgs){
        if (msg_from_buffor.reset_tracking){
          RCLCPP_INFO(this->get_logger(), "Uploading new pose estimation to the buffer");
        }
        msg_from_buffor.initial_pose = last_transformStamped_not_transformed;
        track_it_publisher_->publish(msg_from_buffor);
      }
      _acquiring_buffer = false;
      track_it_publisher_->publish(msg_to_track);
      buffor_msgs.clear();
      RCLCPP_INFO(this->get_logger(), "Tracking restarted");
    }




    if (!_segmentation_started && _segmentation_requested){
      buffor_msgs.clear();
      _segmentation_requested = false;
      _segmentation_started = true;
      _acquiring_buffer = true;
      auto goal = pose_estimation_interfaces::action::SegmentUsingSamEstimatePoseUsingFoundation::Goal();
      goal.img = *msg;
      goal.height = msg->height;
      goal.width = msg->width;
      goal.prompt = "Blocks";

     
      goal.intrinsics = _intrinsics;
      goal.distortion = _distortion;

      goal.depth_img = *depth_msg;

      auto send_goal_options = rclcpp_action::Client<pose_estimation_interfaces::action::SegmentUsingSamEstimatePoseUsingFoundation>::SendGoalOptions();
      send_goal_options.result_callback = std::bind(&PipelineExecutionNode::result_callback, this, _1);

      this->client_ptr_->async_send_goal(goal, send_goal_options);
      RCLCPP_INFO(this->get_logger(), "New request to estimate pose");


    }

    if (_acquiring_buffer){
      buffor_msgs.push_back(msg_to_track);
    }    

    auto current_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    // check if the last segmentation request was more than 5 seconds ago
    if (current_time - _segmentation_last_request_time > 10){
      _segmentation_requested = true;
      _segmentation_last_request_time = current_time;
    }


  }


  void result_callback(const rclcpp_action::ClientGoalHandle<pose_estimation_interfaces::action::SegmentUsingSamEstimatePoseUsingFoundation>::WrappedResult & result){
    RCLCPP_INFO(this->get_logger(), "Result received");
    if (result.code == rclcpp_action::ResultCode::SUCCEEDED){
      RCLCPP_INFO(this->get_logger(), "Segmentation and pose estimation succeeded");
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
      RCLCPP_ERROR(this->get_logger(), "Segmentation and pose estimation failed");
      _new_masks_available = false;
      _segmentation_started = false;
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