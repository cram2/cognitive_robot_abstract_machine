from dataclasses import dataclass

from typing_extensions import Tuple, ClassVar

from robokudo.descriptors.camera_configs.components import (
    RGBDComponent,
    TfComponent,
    StableViewpointComponent,
    WorldDescriptorComponent,
)
from robokudo.descriptors.camera_configs.base_camera_config import BaseCameraConfig


@dataclass(slots=True)
class KinectCameraConfig(
    BaseCameraConfig,
    RGBDComponent,
    TfComponent,
    StableViewpointComponent,
    WorldDescriptorComponent,
):
    """Configuration class for a Kinect camera setup in a robotic environment.

    This class defines the configuration parameters for a Kinect camera, including
    interface settings, topic names, and transformation settings. It is designed to
    work with ROS topics and TF transformations.
    """

    registry_name: ClassVar[str] = "kinect"
    """Overwrite BaseCameraConfig"""

    interface_type: str = "Kinect"
    """Overwrite BaseCameraConfig"""

    topic_depth: str = "/kinect_head/depth_registered/image_raw/compressedDepth"
    """Overwrite RGBDComponent"""

    topic_color: str = "/kinect_head/rgb/image_color/compressed"
    """Overwrite RGBDComponent"""

    topic_cam_info: str = "/kinect_head/rgb/camera_info"
    """Overwrite RGBDComponent"""

    color2depth_ratio: Tuple[float, float] = (0.5, 0.5)
    """Overwrite RGBDComponent"""

    hi_res_mode: bool = True
    """Overwrite RGBDComponent"""

    tf_from: str = "head_mount_kinect_rgb_optical_frame"
    """Overwrite TFComponent"""

    tf_to: str = "map"
    """Overwrite TFComponent"""
