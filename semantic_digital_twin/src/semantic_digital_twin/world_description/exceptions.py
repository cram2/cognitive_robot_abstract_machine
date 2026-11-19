from __future__ import annotations

# %% Imports

from dataclasses import dataclass

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


# %% Inertia tensors


@dataclass(kw_only=True)
class InertiaTensorElementCountError(DataclassException, ValueError):
    """
    A sequence of numbers cannot be the elements of a 3x3 inertia tensor.
    """

    element_count: int
    """
    The number of elements that were given.
    """

    def error_message(self) -> str:
        return f"An inertia tensor has 9 elements, not {self.element_count}."

    def suggest_correction(self) -> str:
        return "Give the nine elements of the 3x3 matrix, row after row."


@dataclass(kw_only=True)
class NegativeMomentOfInertiaError(DataclassException, ValueError):
    """
    An inertia tensor has a negative moment of inertia on its diagonal.
    """

    moments: list[float]
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

    elements: list[float]
    """
    The elements of the matrix, row after row.
    """

    def error_message(self) -> str:
        return f"The inertia tensor with elements {self.elements} is not symmetric."

    def suggest_correction(self) -> str:
        return "Give equal products of inertia above and below the diagonal."
