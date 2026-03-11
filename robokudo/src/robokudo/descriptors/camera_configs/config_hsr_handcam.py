from dataclasses import dataclass

"""
Camera config for the camera-in-hand on the Toyota HSR robot.
"""


@dataclass
class CameraConfig:
    """Configuration class for the hand-mounted camera on the Toyota HSR robot.

    This class defines the configuration parameters for the HSR's hand camera,
    including interface settings, topic names, and transformation settings. The camera
    provides color images without depth information.

    :ivar interface_type: Type of camera interface, set to "ROSCameraWithoutDepthInterface"
    :ivar depthOffset: Offset value for depth measurements (unused in this config)
    :ivar rotate_image: Image rotation setting ('90_ccw', '90_cw', '180', or None)
    :ivar topic_color: ROS topic for color image data
    :ivar topic_cam_info: ROS topic for camera information
    :ivar tf_from: Frame ID of the hand camera
    :ivar tf_to: Target frame ID for transformations
    :ivar lookup_viewpoint: Flag to enable viewpoint lookup (currently disabled)
    :ivar only_stable_viewpoints: Flag to use only stable viewpoints
    :ivar max_viewpoint_distance: Maximum allowed distance for viewpoint changes
    :ivar max_viewpoint_rotation: Maximum allowed rotation for viewpoint changes
    :ivar semantic_map: Filename of the semantic map configuration
    """

    # camera
    interface_type: str = "ROSCameraWithoutDepthInterface"
    depthOffset: int = 0
    rotate_image: str = (
        "90_ccw"  # Possible values: None, '90_ccw', '90_cw', '180' with ccw = counter-clockwise, cw = clockwise
    )
    # filterBlurredImages: bool = True

    # camera topics
    # topic_color: str = "hsrb/hand_camera/image_rect_color"
    topic_color: str = "/hsrb/hand_camera/image_raw"
    topic_cam_info: str = "/hsrb/hand_camera/camera_info"
    # color_hints: str = "compressed"

    # tf
    tf_from: str = "/hand_camera_frame"
    tf_to: str = "/map"
    lookup_viewpoint: bool = (
        False  # Change this to True to lookup transform between cam and world
    )
    only_stable_viewpoints: bool = True
    max_viewpoint_distance: float = 0.01
    max_viewpoint_rotation: float = 1.0
    semantic_map: str = "semantic_map_iai_kitchen.yaml"
