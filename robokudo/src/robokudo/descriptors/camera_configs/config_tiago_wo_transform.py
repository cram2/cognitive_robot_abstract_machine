from dataclasses import dataclass

from typing_extensions import Tuple


@dataclass
class CameraConfig:
    """Configuration class for the TIAGo robot's Xtion camera without transform lookup.

    This class defines the configuration parameters for the Xtion RGB-D camera
    mounted on the TIAGo robot, similar to config_tiago.py but with viewpoint
    lookup disabled. It includes settings for camera interface and topic names.

    :ivar interface_type: Type of camera interface, set to "Kinect" for compatibility
    :ivar depthOffset: Offset value for depth measurements
    :ivar filterBlurredImages: Flag to enable/disable filtering of blurred images
    :ivar color2depth_ratio: Ratio between color and depth image resolution (x, y)
    :ivar hi_res_mode: Enable high resolution mode (disabled for Xtion)
    :ivar topic_depth: ROS topic for depth image data
    :ivar topic_color: ROS topic for color image data
    :ivar topic_cam_info: ROS topic for camera information
    :ivar depth_hints: Transport hints for depth image subscription
    :ivar color_hints: Transport hints for color image subscription
    :ivar tf_from: Frame ID of the Xtion's RGB optical frame
    :ivar tf_to: Target frame ID for transformations (odom frame)
    :ivar lookup_viewpoint: Flag to enable viewpoint lookup (disabled in this config)
    :ivar only_stable_viewpoints: Flag to use only stable viewpoints
    :ivar max_viewpoint_distance: Maximum allowed distance for viewpoint changes
    :ivar max_viewpoint_rotation: Maximum allowed rotation for viewpoint changes
    :ivar semantic_map: Filename of the semantic map configuration

    .. note::
        This configuration is identical to config_tiago.py except that
        lookup_viewpoint is set to False by default.
    """

    # camera
    interface_type: str = "Kinect"
    depthOffset: int = 0
    filterBlurredImages: bool = True
    # If the resolution of the depth image differs from the color image, we need to define the factor for (x, y).
    # Example: (0.5,0.5) for a 640x480 depth image compared to a 1280x960 rgb image Otherwise, just put (1,1) here
    color2depth_ratio: Tuple[float, float] = (1.0, 1.0)
    hi_res_mode: bool = False

    # camera topics
    topic_depth: str = "/xtion/depth_registered/image_raw/compressedDepth"
    topic_color: str = "/xtion/rgb/image_raw/compressed"
    topic_cam_info: str = "/xtion/rgb/camera_info"
    depth_hints: str = "compressedDepth"
    color_hints: str = "compressed"

    # tf
    tf_from: str = "/xtion_rgb_optical_frame"
    tf_to: str = "/odom"
    lookup_viewpoint: bool = False
    only_stable_viewpoints: bool = True
    max_viewpoint_distance: float = 0.01
    max_viewpoint_rotation: float = 1.0
    semantic_map: str = "semantic_map_iai_kitchen.yaml"
