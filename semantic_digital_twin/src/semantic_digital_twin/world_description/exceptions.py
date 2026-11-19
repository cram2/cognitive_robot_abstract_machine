"""
Exceptions raised while describing the bodies of a world and their properties.
"""

from __future__ import annotations

# %% Imports

from dataclasses import dataclass

from numpy.typing import NDArray

from krrood.exceptions import DataclassException

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
