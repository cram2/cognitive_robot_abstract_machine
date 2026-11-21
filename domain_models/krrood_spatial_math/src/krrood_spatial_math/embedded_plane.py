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
    A plane lying in a known three-dimensional Euclidean space: the x-y plane of
    :attr:`frame_3_dimensional`, whose :attr:`normal` is that frame's z-axis.

    Every plane is the x-y plane of some frame, so a vertical or tilted plane is the x-y
    plane of a frame turned to lie in it. Two planes are equal when they lie in one
    space and share their points, whichever side their frames' z-axes point to.
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
        :return: Whether ``other`` is a plane known to lie in this plane's space and share
            its points; :data:`NotImplemented` for anything that is not a plane.
        """
        if not isinstance(other, EmbeddedPlane):
            return NotImplemented
        return self.frame_3_dimensional.compare_embedded_plane(
            frame=other.frame_3_dimensional
        ).is_same_plane

    def __hash__(self) -> int:
        space = self.frame_3_dimensional.space
        if space is None:
            return hash(self.frame_3_dimensional)
        return id(space)
