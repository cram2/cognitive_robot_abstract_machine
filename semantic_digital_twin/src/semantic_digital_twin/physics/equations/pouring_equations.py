"""
Fill-level dynamics of a container that pours liquid out as it tilts.
"""

from __future__ import annotations

from dataclasses import dataclass

import krrood.symbolic_math.symbolic_math as sm
from krrood.patterns.classproperty import classproperty
from krrood.symbolic_math.symbolic_math import Scalar

from semantic_digital_twin.physics.exceptions import (
    NonPositiveContainerGeometryError,
)

# %% pouring equation


@dataclass(kw_only=True)
class PouringEquation:
    """
    Drain of a rectangular container that tilts about its vertical centre axis and
    pours over the lip of its wall::

        dry_height     = container_height · (1 − fill)
        lip_distance   = √(dry_height² + lip_offset²)
        onset_tilt     = atan2(dry_height, lip_offset) + repose_angle
        head_above_lip = max(0, lip_distance · sin(tilt − onset_tilt))
        fill_velocity  = −outflow_rate_constant · head_above_lip / container_height

    ``fill`` is the normalized fill level in ``[0, 1]`` and ``tilt`` the angle of the
    container's up axis from the vertical, in radians.
    """

    container_height: float
    """
    Inner height of the container, in metres.
    """

    container_width: float
    """
    Inner width of the container, in metres.
    """

    outflow_rate_constant: float = 1.0
    """
    Proportionality constant scaling the head above the lip to the normalized drain
    rate.
    """

    discharge_coefficient: float = 0.3
    """
    Coefficient scaling the Torricelli exit speed to a rim pour, whose thin film over
    the lip leaves far slower than an orifice's ``0.6``-``1.0``.
    """

    repose_angle: float = 0.0
    """
    Angle the contents' surface holds before it slumps, in radians; zero for a liquid,
    positive for solids that stand in a heap.
    """

    minimum_head: float = 0.01
    """
    Head the exit speed is computed from at least, in metres, which keeps its gradient
    bounded as the pour starts.
    """

    def __post_init__(self) -> None:
        """
        :raises NonPositiveContainerGeometryError: If the height or width is not
            positive.
        """
        if self.container_height <= 0.0 or self.container_width <= 0.0:
            raise NonPositiveContainerGeometryError(
                container_height=self.container_height,
                container_width=self.container_width,
            )

    @classproperty
    def standard_gravity(cls) -> float:
        """
        Gravitational acceleration liquid falls with, in metres per second squared.
        """
        return 9.81

    @property
    def lip_offset(self) -> float:
        """
        Horizontal distance of the pouring lip, the container's wall, from the tilt
        axis, in metres.
        """
        return self.container_width / 2

    def onset_tilt(self, *, fill: sm.ScalarData) -> Scalar:
        """
        :param fill: Normalized fill level in ``[0, 1]``.
        :return: Symbolic tilt at which a container this full starts to pour, in
            radians; later by the repose angle for heaped contents.
        """
        dry_height = self.container_height * (1 - fill)
        return sm.atan2(dry_height, self.lip_offset) + self.repose_angle

    def head_above_lip(self, *, tilt: sm.ScalarData, fill: sm.ScalarData) -> Scalar:
        """
        :param tilt: Tilt of the container from the vertical, in radians.
        :param fill: Normalized fill level in ``[0, 1]``.
        :return: Symbolic height of the contents' surface above the lip, in metres; zero
            while the container does not pour.
        """
        dry_height = self.container_height * (1 - fill)
        lip_distance = sm.sqrt(dry_height**2 + self.lip_offset**2)
        return sm.max(
            sm.Scalar(0.0),
            lip_distance * sm.sin(tilt - self.onset_tilt(fill=fill)),
        )

    def exit_speed(self, *, tilt: sm.ScalarData, fill: sm.ScalarData) -> Scalar:
        """
        :param tilt: Tilt of the container from the vertical, in radians.
        :param fill: Normalized fill level in ``[0, 1]``.
        :return: Symbolic speed of the stream leaving the lip, ``C_d · √(2 g h)``, in
            metres per second, with the head ``h`` at least :attr:`minimum_head`.
        """
        head = sm.max(
            sm.Scalar(self.minimum_head), self.head_above_lip(tilt=tilt, fill=fill)
        )
        return self.discharge_coefficient * sm.sqrt(2 * self.standard_gravity * head)

    def fill_velocity(self, *, tilt: sm.ScalarData, fill: sm.ScalarData) -> Scalar:
        """
        :param tilt: Tilt of the container from the vertical, in radians.
        :param fill: Normalized fill level in ``[0, 1]``.
        :return: Symbolic rate of the normalized fill level, negative while pouring.
        """
        return (
            -self.outflow_rate_constant
            * self.head_above_lip(tilt=tilt, fill=fill)
            / self.container_height
        )
