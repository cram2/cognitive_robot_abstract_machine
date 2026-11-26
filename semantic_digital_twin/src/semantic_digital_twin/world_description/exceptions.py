"""
Exceptions raised while describing the bodies of a world and their properties.
"""

from __future__ import annotations

# %% Imports

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from krrood.exceptions import DataclassException
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import UsageError

if TYPE_CHECKING:
    from semantic_digital_twin.world_description.world_entity import (
        KinematicStructureEntity,
    )

# %% Color channels


@dataclass
class InvalidColorChannelCount(DataclassException, ValueError):
    """
    A channel sequence cannot describe an RGB or RGBA color.
    """

    channel_count: int
    """Number of supplied color channels."""

    def error_message(self) -> str:
        return f"A color cannot be constructed from {self.channel_count} channels."

    def suggest_correction(self) -> str:
        return "Supply three RGB channels or four RGBA channels."


# %% Inertial properties


@dataclass(kw_only=True)
class ArrayShapeError(DataclassException, ValueError):
    """
    An array does not have the shape of the vector or matrix it is meant to be.
    """

    expected_shape: tuple[int, ...]
    """
    The shape the array must have.
    """

    shape: tuple[int, ...]
    """
    The shape the array has.
    """

    def error_message(self) -> str:
        return f"An array of shape {self.shape} was given where {self.expected_shape} is required."

    def suggest_correction(self) -> str:
        return f"Give an array of shape {self.expected_shape}."


@dataclass(kw_only=True)
class NegativeMomentOfInertiaError(DataclassException, ValueError):
    """
    An inertia tensor has a negative moment of inertia on its diagonal.
    """

    moments: NDArray
    """
    The moments of inertia about the x, y and z axes.
    """

    def error_message(self) -> str:
        return f"The moments of inertia {self.moments} include a negative one."

    def suggest_correction(self) -> str:
        return "Give non-negative moments of inertia about each axis."


@dataclass(kw_only=True)
class AsymmetricInertiaTensorError(DataclassException, ValueError):
    """
    The matrix of an inertia tensor is not symmetric.
    """

    matrix: NDArray
    """
    The matrix of the tensor.
    """

    def error_message(self) -> str:
        return f"The inertia tensor {self.matrix} is not symmetric."

    def suggest_correction(self) -> str:
        return "Give equal products of inertia above and below the diagonal."


# %% Geometry


@dataclass(kw_only=True)
class TooFewRegionPointsError(UsageError):
    """
    Raised when a region is built from fewer unique points than span an area.
    """

    unique_point_count: int
    """
    How many unique points were given.
    """

    minimum_point_count: int
    """
    How many unique points a region needs at least.
    """

    def error_message(self) -> str:
        return f"A region needs at least {self.minimum_point_count} unique points, got {self.unique_point_count}."

    def suggest_correction(self) -> str:
        return f"Give at least {self.minimum_point_count} distinct points."


@dataclass(kw_only=True)
class BoundingBoxCornerFramesDifferError(UsageError):
    """
    Raised when the corners of a bounding box are expressed in different frames.
    """

    min_point_frame: KinematicStructureEntity | None
    """
    The frame of the minimum corner.
    """

    max_point_frame: KinematicStructureEntity | None
    """
    The frame of the maximum corner.
    """

    def error_message(self) -> str:
        return f"The minimum corner is in {self.min_point_frame}, the maximum corner in {self.max_point_frame}."

    def suggest_correction(self) -> str:
        return "Express both corners in the same frame."


# %% Fill levels


@dataclass(kw_only=True)
class MissingFillLevelLimitsError(UsageError):
    """
    Raised when a fill-level connection's fill degree of freedom has no position limits,
    so the integrated fill level cannot be clamped.
    """

    connection_name: PrefixedName
    """
    The name of the fill-level connection whose fill degree of freedom lacks position
    limits.
    """

    def error_message(self) -> str:
        return f"The fill degree of freedom of {self.connection_name} has no position limits to clamp the fill level to."

    def suggest_correction(self) -> str:
        return "create the connection via initialize_fill_level, or give its degree of freedom lower and upper position limits."


# %% Mesh files


@dataclass(kw_only=True)
class MeshFileNotPlaceableByAPose(DataclassException, ValueError):
    """
    A mesh turned within its file and scaled unevenly cannot be shown by placing its
    scaled file at a pose, because the scale stretches the mesh along the turned axes.
    """

    file_name: str
    """
    The mesh's file.
    """

    scale: NDArray[np.float64]
    """
    The scale along the mesh's x, y and z axes.
    """

    def error_message(self) -> str:
        return (
            f"The mesh in {self.file_name} is turned within its file and scaled "
            f"unevenly by {self.scale}, so no pose places its scaled file where the "
            f"mesh lies."
        )

    def suggest_correction(self) -> str:
        return "Place the mesh's own vertices, its unscaled mesh, instead of its file."
