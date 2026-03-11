from dataclasses import dataclass

from typing_extensions import Tuple


@dataclass
class CameraConfig:
    # camera
    interface_type: str = "Kinect"
    depthOffset: int = 0
    filterBlurredImages: bool = False
    # If the resolution of the depth image differs from the color image, we need to define the factor for (x, y).
    # Example: (0.5,0.5) for a 640x480 depth image compared to a 1280x960 rgb image Otherwise, just put (1,1) here
    color2depth_ratio: Tuple[float, float] = (1.0, 1.0)
    hi_res_mode: bool = (
        False  # Setting this to true will apply some workarounds to match the depth data to RGB on the Kinect
    )

    # camera topics
    # topic_depth = "/camera/depth/image_raw/compressedDepth"
    topic_depth: str = "/camera/depth/image_raw"
    topic_color: str = "/camera/color/image_raw/compressed"
    topic_cam_info: str = "/camera/color/camera_info"
    depth_hints: str = "raw"
    color_hints: str = "compressed"

    # tf
    lookup_viewpoint: bool = False
    tf_from: str = "camera_color_optical_frame"
    tf_to: str = "map"
    only_stable_viewpoints: bool = True
    max_viewpoint_distance: float = 0.01
    max_viewpoint_rotation: float = 1.0
    semantic_map: str = "semantic_map_iai_kitchen.yaml"
