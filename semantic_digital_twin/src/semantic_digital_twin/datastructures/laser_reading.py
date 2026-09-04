from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import List

from semantic_digital_twin.spatial_types import Vector3


@dataclass
class LaserReading:
    """
    One sweep of a laser.

    Both lists hold one entry per beam and share their order, so ``direction[i]`` is the
    beam that measured ``distance[i]``.
    """

    direction: List[Vector3] = field(default_factory=list)
    """
    The direction of each beam, as a unit vector in the laser's frame.
    """

    distance: List[float] = field(default_factory=list)
    """
    The distance each beam travelled before it hit a surface, in meters.

    A beam that hit nothing within the scan pattern's range measures ``math.inf``.
    """
