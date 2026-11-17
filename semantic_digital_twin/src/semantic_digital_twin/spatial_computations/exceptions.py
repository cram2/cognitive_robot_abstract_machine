"""
Errors in spatial ray queries.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass
from krrood.exceptions import DataclassException


# %% Ray query inputs
@dataclass
class RayPointShapeMismatch(DataclassException):
    """
    Ray origins and targets describe different numbers of spatial points.
    """

    origin_shape: tuple[int, ...]
    """
    Shape of the ray-origin point array.
    """
    target_shape: tuple[int, ...]
    """
    Shape of the ray-target point array.
    """

    def error_message(self) -> str:
        """
        Identify the incompatible ray-point arrays.
        """
        return f"Ray origin shape {self.origin_shape} differs from target shape {self.target_shape}."

    def suggest_correction(self) -> str:
        """
        Describe matching ray query inputs.
        """
        return "Supply one target point for every origin point."


# %% Camera rendering inputs
@dataclass
class UnsupportedCameraProjection(DataclassException):
    """
    The requested projection cannot be represented by the rendering mode.
    """

    reason: str
    """
    Projection constraint violated by the camera calibration.
    """

    def error_message(self) -> str:
        return f"Unsupported camera projection: {self.reason}."

    def suggest_correction(self) -> str:
        return "Use a calibration supported by the selected rendering mode."
