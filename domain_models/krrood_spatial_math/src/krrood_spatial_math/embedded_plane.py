"""
Planes that lie in a known three-dimensional Euclidean space.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood_spatial_math.cartesian_frame_3_dimensional import (
    CartesianFrame3Dimensional,
)
from krrood_spatial_math.space import EuclideanSpace2Dimensional

# %% planes in three-dimensional space


@dataclass(frozen=True, kw_only=True)
class EmbeddedPlane(EuclideanSpace2Dimensional):
    """
    A plane lying in a known three-dimensional Euclidean space: the x-y plane of
    :attr:`frame_3_dimensional`, whose normal is that frame's z-axis.

    Every plane is the x-y plane of some frame, so a vertical or tilted plane is the x-y
    plane of a frame turned to lie in it. Two planes are equal when their frames are.
    """

    frame_3_dimensional: CartesianFrame3Dimensional
    """
    The frame whose origin lies on this plane and whose x- and y-axes span it.
    """
