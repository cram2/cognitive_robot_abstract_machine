"""
Cartesian frames on a plane that lies in a three-dimensional Euclidean space.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid5

from krrood.patterns.identifier_and_name import NameT
from krrood_spatial_math.cartesian_frame_3_dimensional import (
    CartesianFrame3Dimensional,
)
from krrood_spatial_math.coordinate_system import CartesianFrame
from krrood_spatial_math.embedded_plane import EmbeddedPlane

# %% frames that values on a plane are given in


@dataclass(unsafe_hash=True, kw_only=True)
class CartesianFrame2Dimensional(CartesianFrame[EmbeddedPlane, NameT]):
    """
    The right-handed Cartesian frame, in metres, on the x-y plane of
    :attr:`frame_3_dimensional`, with that frame's origin, x-axis and y-axis.

    Seen from the side the z-axis of :attr:`frame_3_dimensional` points to, the y-axis
    is a quarter turn counterclockwise from the x-axis, so frames on one plane are
    related by a rigid change only when their z-axes point the same way.
    """

    frame_3_dimensional: CartesianFrame3Dimensional
    """
    The frame whose origin, x-axis and y-axis this frame has.
    """

    @property
    def space(self) -> EmbeddedPlane:
        """
        The x-y plane of :attr:`frame_3_dimensional`.
        """
        return EmbeddedPlane(frame_3_dimensional=self.frame_3_dimensional)

    @property
    def name(self) -> NameT:
        """
        The name of :attr:`frame_3_dimensional`, which this frame shares.
        """
        return self.frame_3_dimensional.name

    @property
    def id(self) -> UUID:
        """
        The identifier derived from that of :attr:`frame_3_dimensional`, so the frames on
        the plane of one frame share it.
        """
        return uuid5(self.frame_3_dimensional.id, type(self).__name__)
