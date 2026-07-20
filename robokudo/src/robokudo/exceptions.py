"""RoboKudo-specific exception types."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import StrEnum
from typing_extensions import Any

from krrood.exceptions import DataclassException


@dataclass
class RoboKudoError(DataclassException, ABC):
    """Base class for RoboKudo-specific errors."""


@dataclass
class ColorToDepthRatioMissing(RoboKudoError, RuntimeError):
    """Raised when COLOR2DEPTH_RATIO is required but missing."""

    operation: str = "continue"

    def error_message(self) -> str:
        return f"COLOR2DEPTH_RATIO is missing from the CAS. Can't {self.operation}."

    def suggest_correction(self) -> str:
        return (
            "make sure the camera driver or reader stores COLOR2DEPTH_RATIO in the CAS "
            "before this operation."
        )


@dataclass
class UnknownMode(RoboKudoError, ValueError):
    """Raised when a configured mode value is not supported."""

    mode: Any
    context: str = "RoboKudo component"

    def error_message(self) -> str:
        return f"{self.context} received an unknown mode: {self.mode!r}."

    def suggest_correction(self) -> str:
        return "use one of the modes supported by this component."


@dataclass
class CameraDataMissing(RoboKudoError):
    """Raised when a camera callback requires a message that was not provided."""

    data_name: str
    context: str

    def error_message(self) -> str:
        return f"{self.data_name} is required for {self.context}, but was not provided."

    def suggest_correction(self) -> str:
        return "check the camera subscriptions and synchronization setup."


@dataclass
class CVBridgeImageConversionError(RoboKudoError, ValueError):
    """Raised when image conversion through the cv_bridge workaround fails."""

    source_encoding: str
    """Source ROS image encoding."""

    target_encoding: str
    """Requested target image encoding."""

    reason: str
    """Reason why the image conversion cannot be performed."""

    def error_message(self) -> str:
        return (
            f"Cannot convert ROS image encoding '{self.source_encoding}' to "
            f"'{self.target_encoding}': {self.reason}."
        )

    def suggest_correction(self) -> str:
        return "request a compatible target encoding or provide an image with matching channels."


@dataclass
class CVBridgeImageShapeError(RoboKudoError, ValueError):
    """Raised when an image array shape cannot represent a ROS image."""

    shape: tuple[int, ...]
    """Image array shape."""

    dimensions: int
    """Number of image array dimensions."""

    def error_message(self) -> str:
        return (
            f"Expected 2D or 3D image array, got shape {self.shape} "
            f"with ndim={self.dimensions}."
        )

    def suggest_correction(self) -> str:
        return "provide a single-channel or multi-channel image array."


@dataclass
class CVBridgeROSImageShapeError(RoboKudoError, ValueError):
    """Raised when a ROS image message has invalid dimensions."""

    height: int
    """ROS image height."""

    width: int
    """ROS image width."""

    def error_message(self) -> str:
        return f"Invalid ROS image shape height={self.height}, width={self.width}."

    def suggest_correction(self) -> str:
        return "provide a ROS image with positive height and width."


@dataclass
class CVBridgeROSImageStepError(RoboKudoError, ValueError):
    """Raised when a ROS image row step cannot contain one pixel row."""

    row_bytes: int
    """Configured ROS image row size in bytes."""

    pixel_row_bytes: int
    """Minimum row size required by width, channels, and dtype."""

    def error_message(self) -> str:
        return (
            f"ROS image step ({self.row_bytes}) is smaller than pixel row bytes "
            f"({self.pixel_row_bytes})."
        )

    def suggest_correction(self) -> str:
        return "provide a ROS image step that can contain one full pixel row."


@dataclass
class CVBridgeROSImagePayloadError(RoboKudoError, ValueError):
    """Raised when a ROS image payload is too small for its metadata."""

    actual_bytes: int
    """Number of bytes available in the ROS image payload."""

    required_bytes: int
    """Minimum number of bytes required by height and row step."""

    def error_message(self) -> str:
        return (
            f"ROS image payload too small: got {self.actual_bytes} bytes, "
            f"expected at least {self.required_bytes}."
        )

    def suggest_correction(self) -> str:
        return "provide image data that matches the ROS image height and step."


@dataclass
class CVBridgeUnsupportedImageData(RoboKudoError, ValueError):
    """Raised when image data cannot be mapped to a ROS image encoding."""

    dtype: str
    """Image array data type."""

    channel_count: int
    """Number of image channels."""

    def error_message(self) -> str:
        return (
            f"Unsupported dtype for ROS image conversion: dtype={self.dtype}, "
            f"channels={self.channel_count}."
        )

    def suggest_correction(self) -> str:
        return (
            "provide an image dtype and channel count supported by ROS image encodings."
        )


@dataclass
class CVBridgeUnsupportedEncoding(RoboKudoError, ValueError):
    """Raised when the cv_bridge workaround receives an unknown encoding."""

    encoding: str
    """Unsupported ROS image encoding."""

    def error_message(self) -> str:
        return f"Unsupported ROS image encoding '{self.encoding}'."

    def suggest_correction(self) -> str:
        return "use one of the ROS image encodings supported by CVBridgeWorkaround."


@dataclass
class CVBridgeUnsupportedTargetEncoding(RoboKudoError, ValueError):
    """Raised when the cv_bridge workaround cannot produce an encoding."""

    target_encoding: str
    """Requested target image encoding."""

    def error_message(self) -> str:
        return f"Unsupported desired encoding '{self.target_encoding}'."

    def suggest_correction(self) -> str:
        return "request passthrough, bgr8, 32FC1, or add support for the desired conversion."


@dataclass
class StoredCameraTransformFrameMetadataMissing(RoboKudoError):
    """Raised when stored camera transform frame metadata is missing."""

    def error_message(self) -> str:
        return "Stored CAMERA_TO_WORLD_TRANSFORM is missing frame-name metadata."

    def suggest_correction(self) -> str:
        return "Recreate the recording with the current storage format."


@dataclass
class AnalysisPreconditionError(RoboKudoError, ABC):
    """Base class for unmet analysis preconditions."""


@dataclass
class CASCheckConfigurationError(AnalysisPreconditionError, ValueError):
    """Raised when a CAS check annotator is not configured correctly."""

    component_name: str

    def error_message(self) -> str:
        return (
            f"{self.component_name} needs a check function to work properly, "
            "but no function was provided."
        )

    def suggest_correction(self) -> str:
        return "pass a callable via the 'func' parameter."


@dataclass
class CASCheckFailed(AnalysisPreconditionError):
    """Raised when a configured CAS check condition fails."""

    reason: str

    def error_message(self) -> str:
        return f"CAS check failed: {self.reason}"

    def suggest_correction(self) -> str:
        return "inspect the CAS contents and the condition configured for this check."


class PointCloudThresholdRelation(StrEnum):
    """Relative position of a point cloud size to a configured threshold."""

    BELOW = "below"
    ABOVE = "above"


@dataclass
class PointCloudThresholdError(AnalysisPreconditionError):
    """Raised when point cloud size violates a configured threshold."""

    point_count: int
    threshold: int
    relation: PointCloudThresholdRelation

    def error_message(self) -> str:
        return (
            f"Scene point cloud size ({self.point_count}) is {self.relation.value} "
            f"the configured threshold ({self.threshold})."
        )

    def suggest_correction(self) -> str:
        return (
            "adjust the threshold or verify that the expected point cloud was written "
            "to the CAS."
        )


@dataclass
class PlaneModelMissing(AnalysisPreconditionError):
    """Raised when an algorithm requires a plane model in the CAS."""

    context: str

    def error_message(self) -> str:
        return f"{self.context} requires a plane model in the CAS, but none was found."

    def suggest_correction(self) -> str:
        return (
            "run a plane annotator before this algorithm or provide a plane annotation."
        )


@dataclass
class PointCloudTooSmallForClustering(AnalysisPreconditionError):
    """Raised when too few points are available for point-cloud clustering."""

    point_count: int
    minimum_point_count: int
    context: str = "point cloud clustering"

    def error_message(self) -> str:
        return (
            f"{self.context} requires at least {self.minimum_point_count} points, "
            f"but only {self.point_count} points are available."
        )

    def suggest_correction(self) -> str:
        return "adjust the clustering setup or provide a denser segmented point cloud."


@dataclass
class EmptyPointCloud(AnalysisPreconditionError):
    """Raised when an algorithm requires a non-empty point cloud."""

    context: str

    def error_message(self) -> str:
        return f"{self.context} requires a non-empty point cloud."

    def suggest_correction(self) -> str:
        return "verify that the upstream point cloud producer wrote points to the CAS."


@dataclass
class ImageContourMissing(AnalysisPreconditionError):
    """Raised when an algorithm requires an image contour but none was found."""

    context: str

    def error_message(self) -> str:
        return (
            f"{self.context} requires at least one image contour, but none was found."
        )

    def suggest_correction(self) -> str:
        return "check the image input and contour extraction thresholds."


@dataclass
class WorldDescriptorError(RoboKudoError, ABC):
    """Base class for world descriptor related errors."""


@dataclass
class WorldDescriptorLoadError(WorldDescriptorError, RuntimeError):
    """Raised when loading a world descriptor fails."""

    ros_package: str
    module_name: str

    def error_message(self) -> str:
        return (
            f"Failed to load world descriptor "
            f"'{self.ros_package}.{self.module_name}'."
        )

    def suggest_correction(self) -> str:
        return (
            "check that the ROS package exists and that the descriptor module can be "
            "loaded by RoboKudo's ModuleLoader."
        )


@dataclass
class WorldDescriptorBootstrapError(WorldDescriptorError, RuntimeError):
    """Raised when merging or removing world descriptor content fails."""

    operation: str

    def error_message(self) -> str:
        return f"Failed to {self.operation}."

    def suggest_correction(self) -> str:
        return (
            "inspect the original chained exception for the underlying world model "
            "operation that failed."
        )
