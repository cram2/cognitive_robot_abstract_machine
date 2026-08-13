from dataclasses import dataclass

from typing_extensions import ClassVar

from robokudo.descriptors.camera_configs.components import (
    RGBDComponent,
    TfComponent,
    StableViewpointComponent,
    WorldDescriptorComponent,
)
from robokudo.descriptors.camera_configs.base_camera_config import BaseCameraConfig


@dataclass(slots=True)
class HsrCameraConfig(
    BaseCameraConfig,
    RGBDComponent,
    TfComponent,
    StableViewpointComponent,
    WorldDescriptorComponent,
):
    """Camera config for the HSR robot."""

    registry_name: ClassVar[str] = "hsr"

    interface_type: str = "Kinect"

    topic_depth: str = "hsrb/head_rgbd_sensor/depth_registered/image/compressedDepth"

    topic_color: str = "hsrb/head_rgbd_sensor/rgb/image_raw/compressed"

    topic_cam_info: str = "hsrb/head_rgbd_sensor/rgb/camera_info"

    tf_from: str = "head_rgbd_sensor_rgb_optical_frame"

    tf_to: str = "map"
