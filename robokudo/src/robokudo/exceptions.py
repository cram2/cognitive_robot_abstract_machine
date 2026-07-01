"""RoboKudo-specific exception types."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
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


@dataclass
class PointCloudThresholdError(AnalysisPreconditionError):
    """Raised when point cloud size violates a configured threshold."""

    point_count: int
    threshold: int
    relation: str

    def error_message(self) -> str:
        return (
            f"Scene point cloud size ({self.point_count}) is {self.relation} "
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
