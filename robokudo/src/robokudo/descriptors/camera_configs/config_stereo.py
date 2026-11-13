"""
Configuration for rectified ROS stereo depth from any upstream processor.
"""

from __future__ import annotations

# %% imports

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from rclpy.qos import ReliabilityPolicy

from krrood.ormatic.utils import classproperty
from robokudo.descriptors.camera_configs.base_camera_config import BaseCameraConfig
from robokudo.descriptors.camera_configs.components import (
    StaticCameraTransformComponent,
    TfComponent,
)
from robokudo.descriptors.camera_configs.exceptions import (
    InvalidStereoCalibrationTolerance,
    InvalidStereoHistoryDepth,
    InvalidStereoMaximumDepth,
    InvalidStereoSynchronizationQueueSize,
    StereoOpticalFrameMissing,
    StereoPoseSourcesConflict,
)

# %% source identity


class DepthCameraSource(StrEnum):
    """
    Registered source and interface identifiers for rectified depth.
    """

    STEREO = "stereo"
    RECTIFIED_DEPTH = "rectified_depth"


# %% rectified depth configuration


@dataclass(kw_only=True)
class StereoCameraConfig(BaseCameraConfig, TfComponent, StaticCameraTransformComponent):
    """
    Read aligned metric depth and a rectified left image from a ROS producer.
    """

    interface_type: str = DepthCameraSource.RECTIFIED_DEPTH
    """
    Interface identifier for rectified metric depth.
    """

    topic_color: str = "/left/image_rect_color"
    """
    Rectified left image topic.
    """

    topic_depth: str = "/stereo/depth"
    """
    Float32 meter depth topic aligned to the left image.
    """

    topic_camera_info: str = "/left/camera_info"
    """
    Per-frame left calibration topic.
    """

    tf_from: str = "left_optical_frame"
    """
    Rectified optical frame shared by all input messages.
    """

    tf_to: str = "map"
    """
    Global frame to transform the rectified optical frame to.
    """

    lookup_viewpoint: bool = True
    """
    Whether to require acquisition-time TF before consuming a frame.
    """

    color_reliability: ReliabilityPolicy = ReliabilityPolicy.RELIABLE
    """
    Requested reliability for camera images.
    """

    depth_reliability: ReliabilityPolicy = ReliabilityPolicy.RELIABLE
    """
    Requested reliability for inferred depth.
    """

    calibration_reliability: ReliabilityPolicy = ReliabilityPolicy.RELIABLE
    """
    Requested reliability for calibration messages.
    """

    history_depth: int = 2
    """
    DDS history length for each subscription.
    """

    synchronization_queue_size: int = 30
    """
    Maximum messages per stream retained for delayed depth matching.
    """

    calibration_tolerance: float = 0.03
    """
    Maximum calibration timestamp difference in seconds.
    """

    max_depth: float = 10.0
    """
    Maximum valid depth in meters.
    """

    @classproperty
    def registry_name(cls: type[StereoCameraConfig]) -> str:
        """
        The camera source name recognized by the descriptor factory.
        """
        return DepthCameraSource.STEREO

    def __post_init__(self) -> None:
        """
        Reject unbounded buffering and contradictory transform settings.
        """
        if self.history_depth <= 0:
            raise InvalidStereoHistoryDepth(history_depth=self.history_depth)
        if self.synchronization_queue_size <= 0:
            raise InvalidStereoSynchronizationQueueSize(
                synchronization_queue_size=self.synchronization_queue_size
            )
        if not isfinite(self.max_depth) or self.max_depth <= 0:
            raise InvalidStereoMaximumDepth(max_depth=self.max_depth)
        if not isfinite(self.calibration_tolerance) or self.calibration_tolerance < 0:
            raise InvalidStereoCalibrationTolerance(
                calibration_tolerance=self.calibration_tolerance
            )
        if not self.tf_from:
            raise StereoOpticalFrameMissing()
        if self.lookup_viewpoint and self.static_camera_transform_enabled:
            raise StereoPoseSourcesConflict()
