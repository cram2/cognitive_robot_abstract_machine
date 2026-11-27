"""
Fill-level dynamics of a container that pours liquid out as it tilts.
"""

from __future__ import annotations

from dataclasses import dataclass

import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.symbolic_math import Scalar

from semantic_digital_twin.physics.equations.projectile_motion import ProjectileMotion
from semantic_digital_twin.physics.exceptions import (
    NonPositiveContainerGeometryError,
)
from semantic_digital_twin.world_description.geometry import Scale

# %% pouring equation


@dataclass(kw_only=True)
class PouringEquation:
    r"""
    Drain of a rectangular container that tilts about its vertical centre axis and
    pours over the lip of its wall:

    .. math::

        d &= H (1 - f) \\
        r &= \sqrt{d^2 + l^2} \\
        \theta_0 &= \operatorname{atan2}(d, l) + \theta_\mathrm{r} \\
        h &= \max\bigl(0, r \sin(\theta - \theta_0)\bigr) \\
        \dot{f} &= -k \, h / H

    where

    :math:`f` : normalized fill level in :math:`[0, 1]`

    :math:`\theta` : tilt of the container's up axis from the vertical

    :math:`H` : inner height of the container

    :math:`l` : :attr:`lip_offset`, the lip's horizontal distance from the tilt axis

    :math:`\theta_\mathrm{r}` : :attr:`repose_angle`

    :math:`k` : :attr:`outflow_rate_constant`

    :math:`d` : height of the container left dry above the contents

    :math:`r` : distance from the contents' surface centre to the lip

    :math:`\theta_0` : :meth:`onset_tilt`, the tilt at which the pour starts

    :math:`h` : :meth:`head_above_lip`
    """

    container_scale: Scale
    """
    Inner extents of the container, in metres: ``y`` across the pouring lip and ``z``
    its height; ``x``, along the lip, does not change the normalized drain.
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
        :raises NonPositiveContainerGeometryError: If the width across the lip or the
            height is not positive.
        """
        if self.container_scale.y <= 0.0 or self.container_scale.z <= 0.0:
            raise NonPositiveContainerGeometryError(
                container_scale=self.container_scale
            )

    @property
    def lip_offset(self) -> float:
        """
        Horizontal distance of the pouring lip, the container's wall, from the tilt
        axis, in metres.
        """
        return self.container_scale.y / 2

    def onset_tilt(self, *, fill: sm.ScalarData) -> Scalar:
        """
        :param fill: Normalized fill level in ``[0, 1]``.
        :return: Symbolic tilt at which a container this full starts to pour, in
            radians; later by the repose angle for heaped contents.
        """
        dry_height = self.container_scale.z * (1 - fill)
        return sm.atan2(dry_height, self.lip_offset) + self.repose_angle

    def head_above_lip(self, *, tilt: sm.ScalarData, fill: sm.ScalarData) -> Scalar:
        """
        :param tilt: Tilt of the container from the vertical, in radians.
        :param fill: Normalized fill level in ``[0, 1]``.
        :return: Symbolic height of the contents' surface above the lip, in metres; zero
            while the container does not pour.
        """
        dry_height = self.container_scale.z * (1 - fill)
        lip_distance = sm.sqrt(dry_height**2 + self.lip_offset**2)
        return sm.max(
            sm.Scalar(0.0),
            lip_distance * sm.sin(tilt - self.onset_tilt(fill=fill)),
        )

    def exit_speed(self, *, tilt: sm.ScalarData, fill: sm.ScalarData) -> Scalar:
        r"""
        :param tilt: Tilt of the container from the vertical, in radians.
        :param fill: Normalized fill level in ``[0, 1]``.
        :return: Symbolic speed of the stream leaving the lip,
            :math:`C_\mathrm{d} \sqrt{2 g h}` by Torricelli's law, in metres per second,
            with the head :math:`h` at least :attr:`minimum_head`.
        """
        head = sm.max(
            sm.Scalar(self.minimum_head), self.head_above_lip(tilt=tilt, fill=fill)
        )
        return self.discharge_coefficient * ProjectileMotion.speed_after_drop(
            height=head
        )

    def fill_velocity(self, *, tilt: sm.ScalarData, fill: sm.ScalarData) -> Scalar:
        """
        :param tilt: Tilt of the container from the vertical, in radians.
        :param fill: Normalized fill level in ``[0, 1]``.
        :return: Symbolic rate of the normalized fill level, negative while pouring and
            zero once the container is empty.
        """
        drain = (
            -self.outflow_rate_constant
            * self.head_above_lip(tilt=tilt, fill=fill)
            / self.container_scale.z
        )
        return sm.if_greater_zero(fill, drain, sm.Scalar(0.0))
