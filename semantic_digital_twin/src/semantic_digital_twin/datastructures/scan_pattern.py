from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from semantic_digital_twin.exceptions import InvalidScanPattern


@dataclass
class ScanPattern:
    """
    The directions a lidar sweeps and the distances it can measure.

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
    def beam_angles(self) -> npt.NDArray[np.float64]:
        """
        :return: The angle of every beam, in radians, ordered from
            :attr:`minimum_angle` outwards.
        """
        return self.minimum_angle + np.arange(self.beam_count) * self.angle_increment

    @property
    def beam_directions(self) -> npt.NDArray[np.float64]:
        """
        :return: A unit vector along every beam, one per row, in the scanner's frame.
        """
        angles = self.beam_angles
        return np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)))
