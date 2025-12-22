from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import List, TYPE_CHECKING

from ..spatial_types import Point3

if TYPE_CHECKING:
    from ..world import Body


@dataclass
class ClosestPoint:
    """
    Data class to store the closest point between two bodies the minimum distance between these two bodies.
    """

    body_a: Body
    """
    The body the closest points belong to.
    """
    body_b: Body
    """
    The other body the closest points are measured to.
    """
    point_on_a: Point3
    """
    Closest point on the first body.
    """
    point_on_b: Point3
    """
    Closest point on the second body.
    """
    min_distance: float
    """
    The minimum distance between the two bodies.
    """
