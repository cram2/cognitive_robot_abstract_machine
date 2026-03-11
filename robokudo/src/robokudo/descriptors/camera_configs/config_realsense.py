from dataclasses import dataclass

from typing_extensions import Tuple


# Camera config for Realsense Camera devices.
# Tested with a D435
# Please start the ROS Realsense driver like this:
#   roslaunch realsense2_camera rs_aligned_depth.launch
@dataclass
class CameraConfig:
    """Configuration class for Intel RealSense cameras.

    This class defines the configuration parameters for RealSense cameras,
    particularly tested with the D435 model. It assumes the use of aligned depth
    data and requires the RealSense ROS driver to be running.

    :ivar interface_type: Type of camera interface, set to "Kinect" for compatibility
    :ivar depthOffset: Offset value for depth measurements
    :ivar filterBlurredImages: Flag to enable/disable filtering of blurred images
    :ivar color2depth_ratio: Ratio between color and depth image resolution (x, y)
    :ivar hi_res_mode: Enable high resolution mode (disabled for RealSense)
    :ivar topic_depth: ROS topic for aligned depth image data
    :ivar topic_color: ROS topic for color image data
    :ivar topic_cam_info: ROS topic for camera information
    :ivar depth_hints: Transport hints for depth image subscription
    :ivar color_hints: Transport hints for color image subscription
    :ivar tf_from: Frame ID of the camera's color optical frame
    :ivar tf_to: Target frame ID for transformations
    :ivar lookup_viewpoint: Flag to enable viewpoint lookup (disabled by default)
    :ivar only_stable_viewpoints: Flag to use only stable viewpoints
    :ivar max_viewpoint_distance: Maximum allowed distance for viewpoint changes
    :ivar max_viewpoint_rotation: Maximum allowed rotation for viewpoint changes
    :ivar semantic_map: Filename of the semantic map configuration

    .. note::
        Requires the RealSense ROS driver to be running with aligned depth:
        roslaunch realsense2_camera rs_aligned_depth.launch
    """

    # camera
    interface_type: str = "Kinect"
    depthOffset: int = 0
    filterBlurredImages: bool = True
    # We currently assume that the Color and Depth topics run at the same resolution
    color2depth_ratio: Tuple[float, float] = (1.0, 1.0)
    hi_res_mode: bool = False

    # camera topics
    topic_depth: str = "/camera/aligned_depth_to_color/image_raw/compressedDepth"
    topic_color: str = "/camera/color/image_raw/compressed"
    topic_cam_info: str = "/camera/color/camera_info"
    depth_hints: str = "compressedDepth"
    color_hints: str = "compressed"

    # tf
    tf_from: str = "/camera_color_optical_frame"
    tf_to: str = "/map"
    lookup_viewpoint: bool = False
    only_stable_viewpoints: bool = True
    max_viewpoint_distance: float = 0.01
    max_viewpoint_rotation: float = 1.0
    semantic_map: str = "semantic_map_iai_kitchen.yaml"
