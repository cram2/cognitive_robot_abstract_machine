"""
This is the camera config for the HSR robot.
"""

from dataclasses import dataclass

from typing_extensions import Tuple


@dataclass
class CameraConfig:
    """A config"""

    interface_type: str = "Kinect"
    """Camera interface type."""

    depthOffset: int = 0
    """Depth offset."""

    filterBlurredImages: bool = True

    color2depth_ratio: Tuple[float, float] = (1.0, 1.0)
    """
    If the resolution of the depth image differs from the color image,
    we need to define the factor for (x, y).
    Example: (0.5,0.5) for a 640x480 depth image compared to a
    1280x960 rgb image Otherwise, just put (1,1) here
    """

    hi_res_mode: bool = False
    """ Setting this to true will apply some workarounds to match the depth data to RGB on the Kinect """

    # topic_depth: str = "hsrb/head_rgbd_sensor/depth_registered/image_raw"
    topic_depth: str = "hsrb/head_rgbd_sensor/depth_registered/image/compressedDepth"
    topic_color: str = "hsrb/head_rgbd_sensor/rgb/image_raw/compressed"
    topic_cam_info: str = "hsrb/head_rgbd_sensor/rgb/camera_info"
    # depth_hints = "raw"
    depth_hints: str = "compressedDepth"
    color_hints: str = "compressed"

    tf_from: str = "head_rgbd_sensor_rgb_frame"
    """Frame ID of the camera's optical frame."""

    tf_to: str = "map"
    """World frame ID to transform the camera's optical frame to."""

    lookup_viewpoint: bool = True
    """Whether to lookup the viewpoint between the camera and the world."""

    only_stable_viewpoints: bool = True
    max_viewpoint_distance: float = 0.01
    max_viewpoint_rotation: float = 1.0
    semantic_map: str = "semantic_map_iai_kitchen.yaml"
