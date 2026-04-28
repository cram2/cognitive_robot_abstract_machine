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
class OrbbecCameraConfig(
    BaseCameraConfig,
    RGBDComponent,
    TfComponent,
    StableViewpointComponent,
    WorldDescriptorComponent,
):
    """Configuration class for the OrbBec Astra camera."""

    registry_name: ClassVar[str] = "orbbec"

    interface_type = "Kinect"

    topic_depth: str = "/camera/depth/image_raw"

    topic_color: str = "/camera/color/image_raw/compressed"

    topic_cam_info: str = "/camera/color/camera_info"

    tf_from: str = "camera_color_optical_frame"

    tf_to: str = "map"

    depth_hints: str = "raw"

    filterBlurredImages: bool = False
