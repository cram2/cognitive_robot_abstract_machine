from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from typing_extensions import TYPE_CHECKING, List, Optional

from semantic_digital_twin.exceptions import InvalidScanPattern
from semantic_digital_twin.spatial_types.spatial_types import Vector3

if TYPE_CHECKING:
    from semantic_digital_twin.world_description.world_entity import (
        KinematicStructureEntity,
    )


@dataclass
class ScanPattern:
    """
    The directions a laser scanner sweeps and the distances it can measure.

    The beams lie in the xy plane of the scanner's frame, with the first beam at
    :attr:`minimum_angle` measured from the x axis and rotating about the z axis, as
    ``sensor_msgs/LaserScan`` defines it.
    """

    minimum_angle: float
    """
    The angle of the first beam, in radians.
    """

    maximum_angle: float
    """
    The angle of the last beam, in radians.
    """

    angle_increment: float
    """
    The angle between two neighbouring beams, in radians.
    """

    minimum_range: float
    """
    The closest distance the scanner can measure, in meters.
    """

    maximum_range: float
    """
    The farthest distance the scanner can measure, in meters.
    """

    def __post_init__(self):
        if self.angle_increment <= 0:
            raise InvalidScanPattern(
                pattern=self, reason="the angle increment must be positive"
            )
        if self.maximum_angle < self.minimum_angle:
            raise InvalidScanPattern(
                pattern=self,
                reason="the maximum angle must not be smaller than the minimum angle",
            )
        if self.minimum_range < 0:
            raise InvalidScanPattern(
                pattern=self, reason="the minimum range must not be negative"
            )
        if self.maximum_range <= self.minimum_range:
            raise InvalidScanPattern(
                pattern=self,
                reason="the maximum range must be larger than the minimum range",
            )

    @property
    def beam_count(self) -> int:
        """
        :return: How many beams one scan holds.
        """
        return (
                int(round((self.maximum_angle - self.minimum_angle) / self.angle_increment))
                + 1
        )

    @property
    def beam_angles(self) -> np.ndarray:
        """
        :return: The angle of every beam, in radians, ordered from
            :attr:`minimum_angle` outwards.
        """
        return self.minimum_angle + np.arange(self.beam_count) * self.angle_increment

    @property
    def beam_directions(self) -> np.ndarray:
        """
        :return: A unit vector along every beam, one per row, in the scanner's frame.
        """
        angles = self.beam_angles
        return np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)))

    def beam_directions_in_frame(
            self, reference_frame: Optional[KinematicStructureEntity]
    ) -> List[Vector3]:
        """
        :param reference_frame: The frame the returned vectors are expressed in.
        :return: A unit vector along every beam.
        """
        return [
            Vector3(*direction, reference_frame=reference_frame)
            for direction in self.beam_directions
        ]
