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
class RealsenseCameraConfig(
    BaseCameraConfig,
    RGBDComponent,
    TfComponent,
    StableViewpointComponent,
    WorldDescriptorComponent,
):
    """Configuration class for Intel RealSense cameras.

    This class defines the configuration parameters for RealSense cameras,
    particularly tested with the D435 model. It assumes the use of aligned depth
    data and requires the RealSense ROS driver to be running.

    .. note::
        Requires the RealSense ROS driver to be running with aligned depth:
        roslaunch realsense2_camera rs_aligned_depth.launch
    """

    registry_name: ClassVar[str] = "realsense"

    interface_type: str = "Kinect"

    topic_depth: str = "/camera/aligned_depth_to_color/image_raw/compressedDepth"

    topic_color: str = "/camera/color/image_raw/compressed"

    topic_cam_info: str = "/camera/color/camera_info"

    tf_from: str = "/camera_color_optical_frame"

    tf_to: str = "/map"

    filterBlurredImages: bool = False

    lookup_viewpoint: bool = False
