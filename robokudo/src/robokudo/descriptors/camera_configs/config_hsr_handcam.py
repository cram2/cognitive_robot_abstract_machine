from dataclasses import dataclass

from typing_extensions import ClassVar

from robokudo.descriptors.camera_configs.base_camera_config import BaseCameraConfig
from robokudo.descriptors.camera_configs.components import (
    ColorComponent,
    TfComponent,
    StableViewpointComponent,
    WorldDescriptorComponent,
)


@dataclass(slots=True)
class HsrHandCameraConfig(
    BaseCameraConfig,
    ColorComponent,
    TfComponent,
    StableViewpointComponent,
    WorldDescriptorComponent,
):
    """Camera config for the camera-in-hand on the Toyota HSR robot."""

    registry_name: ClassVar[str] = "hsr_handcam"

    rotate_image: str = "90_ccw"
    """Rotate the image cw = clockwise, ccw = counter-clockwise. Possible values: None, 90_{cw, ccw}, 180_{cw, ccw}."""

    interface_type: str = "ROSCameraWithoutDepthInterface"

    topic_color: str = "hsrb/head_rgbd_sensor/rgb/image_raw/compressed"

    topic_cam_info: str = "hsrb/head_rgbd_sensor/rgb/camera_info"

    tf_from: str = "hand_camera_frame"

    tf_to: str = "map"
