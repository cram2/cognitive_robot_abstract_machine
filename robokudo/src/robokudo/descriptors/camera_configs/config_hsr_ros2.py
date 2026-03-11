"""
This is the camera config for the HSR robot.
"""

from dataclasses import dataclass

from typing_extensions import Tuple


@dataclass
class CameraConfig:
    """A config"""

    # camera
    interface_type: str = "Kinect"
    depthOffset: str = 0
    filterBlurredImages: bool = True
    # If the resolution of the depth image differs from the color image,
    # we need to define the factor for (x, y).
    # Example: (0.5,0.5) for a 640x480 depth image compared to a
    # 1280x960 rgb image Otherwise, just put (1,1) here
    color2depth_ratio: Tuple[float, float] = (1.0, 1.0)
    # Setting this to true will apply some workarounds
    # to match the depth data to RGB on the Kinect
    hi_res_mode: bool = False

    # camera topics
    topic_depth: str = "/head_rgbd_sensor/depth_registered/image/compressedDepth"
    topic_color: str = "/head_rgbd_sensor/rgb/image_rect_color/compressed"
    topic_cam_info: str = "/head_rgbd_sensor/rgb/camera_info"
    # depth_hints: str = "raw"
    depth_hints: str = "compressedDepth"
    color_hints: str = "compressed"

    # tf
    tf_from: str = "head_rgbd_sensor_rgb_frame"
    tf_to: str = "map"
    lookup_viewpoint: bool = (
        True  # Change this to True to lookup transform between cam and world
    )
    only_stable_viewpoints: bool = True
    max_viewpoint_distance: float = 0.01
    max_viewpoint_rotation: float = 1.0
    semantic_map: str = "semantic_map_iai_kitchen.yaml"
