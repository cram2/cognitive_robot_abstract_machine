from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from typing_extensions import TYPE_CHECKING

from semantic_digital_twin.datastructures.scan_pattern import ScanPattern

if TYPE_CHECKING:
    from semantic_digital_twin.world_description.world_entity import (
        KinematicStructureEntity,
    )


@dataclass
class LidarReading:
    """
    One sweep of a lidar, laid out like ``sensor_msgs/LaserScan``.
    """

    reference_frame: KinematicStructureEntity
    """
    The frame the sweep was measured in.
    """

    scan_pattern: ScanPattern
    """
    The directions the sweep covered and the distances it could measure.
    """

    ranges: np.ndarray
    """
    The distance each beam travelled before it hit a surface, in meters, ordered like the
    beams of :attr:`scan_pattern`.

    A beam that hit nothing within the scan pattern's range measures ``math.inf``.
    """
