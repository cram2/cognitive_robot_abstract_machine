"""
Rigid motions given as a pair of vectors along the axes of one frame, such as twists and
screw axes, which a rigid change of frame moves by its adjoint action.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from typing_extensions import Generic, Self

from krrood_spatial_math.coordinate_system import (
    CoordinateRepresentation,
    RigidCoordinateChange,
)
from krrood_spatial_math.vector import GenericCrossProductVector

# %% a rigid motion as how it turns and how the origin moves


class MotionVectorPair(
    CoordinateRepresentation, Generic[GenericCrossProductVector], ABC
):
    """
    A rigid motion of a body given by two vectors along the axes of one frame: how the
    body turns, and how the point of the body at the frame's origin moves.

    A change of frame moves the pair by its adjoint action: both vectors turn, and the
    moved origin's motion gains the turn crossed with the shift between the origins.
    """

    @property
    @abstractmethod
    def rotational_part(self) -> GenericCrossProductVector:
        """
        How the body turns.
        """

    @property
    @abstractmethod
    def translational_part(self) -> GenericCrossProductVector:
        """
        How the point of the body at the frame's origin moves.
        """

    @classmethod
    @abstractmethod
    def from_parts(
        cls,
        *,
        rotational_part: GenericCrossProductVector,
        translational_part: GenericCrossProductVector,
    ) -> Self:
        """
        :param rotational_part: How the body turns.
        :param translational_part: How the point of the body at the origin of the frame
            both parts are given in moves.
        :return: The motion with these parts, given in their frame.
        """

    def apply_transformation(self, *, transformation: RigidCoordinateChange) -> Self:
        rotational_part = self.rotational_part.apply_transformation(
            transformation=transformation
        )
        translational_part = self.translational_part.apply_transformation(
            transformation=transformation
        ) + transformation.translation.cross(other=rotational_part)
        return self.from_parts(
            rotational_part=rotational_part, translational_part=translational_part
        )
