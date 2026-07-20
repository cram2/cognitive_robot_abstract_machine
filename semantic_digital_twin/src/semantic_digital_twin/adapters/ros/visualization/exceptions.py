from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

from krrood.exceptions import DataclassException
from semantic_digital_twin.spatial_types.spatial_types import SpatialType


@dataclass
class CannotRenderSpatialTypeError(DataclassException):
    """
    Raised when no renderer can turn a spatial type into RViz markers.
    """

    spatial_type_type: Type = field(kw_only=True)
    """
    The type of the spatial type that could not be rendered.
    """

    def error_message(self) -> str:
        return (
            f"No renderer available for spatial type {self.spatial_type_type.__name__}."
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class WorldNotResolvableError(DataclassException):
    """
    Raised when no world can be resolved for a spatial type that should be published.
    """

    spatial_type: SpatialType = field(kw_only=True)
    """
    The spatial type whose world could not be resolved.
    """

    def error_message(self) -> str:
        return (
            "Cannot resolve a world: no world was given and the spatial type "
            f"{self.spatial_type} has no reference frame to derive one from."
        )

    def suggest_correction(self) -> str:
        return "Pass the world explicitly or give the spatial type a reference frame."
