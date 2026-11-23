"""
Planes that lie in a known three-dimensional Euclidean space.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

from krrood_spatial_math.cartesian_frame_3_dimensional import (
    CartesianFrame3Dimensional,
)
from krrood_spatial_math.space import EuclideanSpace

if TYPE_CHECKING:
    from krrood_spatial_math.vector import Vector

# %% planes in three-dimensional space


@dataclass(frozen=True, kw_only=True, eq=False)
class EmbeddedPlane(EuclideanSpace):
    """
    An oriented plane in a three-dimensional Euclidean space: the x-y plane of
    :attr:`frame_3_dimensional`, whose :attr:`normal` is that frame's z-axis.

    It is the :attr:`~CartesianFrame3Dimensional.embedded_plane` of every frame that
    lies on it in every state, so two planes are equal when their frames are one frame.
    """

    frame_3_dimensional: CartesianFrame3Dimensional
    """
    The frame whose origin lies on this plane and whose x- and y-axes span it.
    """

    @property
    def normal(self) -> Vector:
        """
        The unit vector perpendicular to this plane: the z-axis of
        :attr:`frame_3_dimensional`, given in that frame.

        A plane with another normal is the x-y plane of a frame whose z-axis is that
        normal.
        """
        return self.frame_3_dimensional.z_axis

    def __eq__(self, other: object) -> bool:
        """
        :return: Whether ``other`` is the x-y plane of the same frame;
            :data:`NotImplemented` for anything that is not a plane.
        """
        if not isinstance(other, EmbeddedPlane):
            return NotImplemented
        return other.frame_3_dimensional is self.frame_3_dimensional

    def __hash__(self) -> int:
        """
        :return: The hash of the identity of :attr:`frame_3_dimensional`, which equal
            planes share.
        """
        return id(self.frame_3_dimensional)
