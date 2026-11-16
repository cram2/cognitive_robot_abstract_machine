from __future__ import annotations

# %% Imports
from krrood.ormatic.utils import classproperty
from dataclasses import dataclass


from robokudo.types.image import ImageRotation

from robokudo.descriptors.camera_configs.base_camera_config import BaseCameraConfig
from robokudo.descriptors.camera_configs.components import (
    ColorComponent,
    TfComponent,
    StableViewpointComponent,
    WorldDescriptorComponent,
)


# %% Hand camera configuration
@dataclass(slots=True)
class HsrHandCameraConfig(
    BaseCameraConfig,
    ColorComponent,
    TfComponent,
    StableViewpointComponent,
    WorldDescriptorComponent,
):
    """
    Camera config for the camera-in-hand on the Toyota HSR robot.
    """

    @classproperty
    def registry_name(cls) -> str:
        """
        Identifier selecting this camera configuration.
        """
        return "hsr_handcam"

    rotate_image: ImageRotation | None = ImageRotation.COUNTERCLOCKWISE_90
    """
    Optional rotation applied to the delivered image and calibration.
    """
    interface_type: str = "ROSCameraWithoutDepthInterface"

    topic_color: str = "hsrb/head_rgbd_sensor/rgb/image_raw/compressed"

    topic_camera_info: str = "hsrb/head_rgbd_sensor/rgb/camera_info"

    tf_from: str = "hand_camera_frame"

    tf_to: str = "map"
