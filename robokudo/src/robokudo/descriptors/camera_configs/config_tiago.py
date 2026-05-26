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
class TiagoCameraConfig(
    BaseCameraConfig,
    RGBDComponent,
    TfComponent,
    StableViewpointComponent,
    WorldDescriptorComponent,
):
    """Configuration class for the TIAGo robot's Xtion camera.

    This class defines the configuration parameters for the Xtion RGB-D camera
    mounted on the TIAGo robot. It includes settings for camera interface,
    topic names, and transformation settings.

    .. note::
        The configuration uses the Kinect interface type for compatibility with
        the Xtion camera, as both cameras use similar RGB-D data formats.
    """

    registry_name: ClassVar[str] = "tiago"

    interface_type: str = "Kinect"

    topic_depth: str = "/xtion/depth_registered/image_raw/compressedDepth"

    topic_cam_info: str = "/xtion/rgb/camera_info"

    topic_color: str = "/xtion/rgb/image_raw/compressed"

    tf_from: str = "/xtion_rgb_optical_frame"

    tf_to: str = "/odom"
