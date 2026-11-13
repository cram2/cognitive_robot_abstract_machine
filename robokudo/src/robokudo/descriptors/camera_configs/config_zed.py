"""
Topic defaults shared by ZED Mini and ZED 2i ROS 2 wrappers.
"""

from __future__ import annotations

# %% Imports

from dataclasses import dataclass
from enum import StrEnum

from krrood.ormatic.utils import classproperty
from robokudo.descriptors.camera_configs.config_stereo import StereoCameraConfig

# %% ZED source


class ZedSource(StrEnum):
    """
    Registered name of the ZED topic preset.
    """

    ZED = "zed"


@dataclass(kw_only=True)
class ZedCameraConfig(StereoCameraConfig):
    """
    Use :class:`StereoCameraConfig` with ZED wrapper 5.1+ topic names.
    """

    topic_color: str = "/zed/zed_node/left/color/rect/image"
    """
    Rectified left color stream in the default wrapper namespace.
    """

    topic_camera_info: str = "/zed/zed_node/left/color/rect/camera_info"
    """
    Per-acquisition calibration of the rectified left pixel grid.
    """

    topic_depth: str = "/zed/zed_node/depth/depth_registered"
    """
    SDK depth aligned to the left image; configure float32 meters.
    """

    tf_from: str = "zed_left_camera_optical_frame"
    """
    Left optical frame for the default camera name; override when renamed.
    """

    @classproperty
    def registry_name(cls) -> str:
        """
        Select the ZED preset in the camera registry.
        """
        return ZedSource.ZED
