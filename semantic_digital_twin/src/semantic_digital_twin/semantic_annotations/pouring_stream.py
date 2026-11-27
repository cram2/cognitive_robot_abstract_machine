"""
The stream a pouring container sends into the opening of the container it pours into.
"""

from __future__ import annotations

from dataclasses import dataclass

import krrood.symbolic_math.symbolic_math as sm
from krrood_spatial_math.spatial_types import Point3
from typing_extensions import TYPE_CHECKING

from semantic_digital_twin.physics.equations.projectile_motion import ProjectileMotion
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

if TYPE_CHECKING:
    from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel

# %% pouring stream


@dataclass(eq=False, kw_only=True)
class PouringStream(SemanticAnnotation):
    """
    The stream from :attr:`source` into :attr:`receiver`: it carries the source's
    outflow only while it lands in the receiver's opening, so neither container flows
    while it would spill.
    """

    source: HasFillLevel
    """
    The container pouring.
    """

    receiver: HasFillLevel
    """
    The container the source pours into.
    """

    def landing_point(self) -> Point3:
        """
        :return: Symbolic world-frame point where the stream, thrown from the source's
            lip at its exit speed, crosses the receiver's opening plane.
        """
        return ProjectileMotion(
            launch_point=self.source.liquid_exit_point(),
            launch_velocity=self.source.liquid_exit_velocity(
                exit_speed=self.source.exit_speed()
            ),
        ).landing_point(plane_height=self.receiver.opening_point().z)

    def gate(self) -> sm.Scalar:
        """
        :return: Symbolic gate in ``[0, 1]``, open while the source's lip is above the
            receiver's opening and the stream lands within the opening.
        """
        return self.height_gate() * self.overlap_gate()

    def height_gate(self) -> sm.Scalar:
        """
        :return: Symbolic factor of the gate closing as the source's lip sinks to the
            receiver's opening, measured in receiver heights.
        """
        clearance = self.source.liquid_exit_point().z - self.receiver.opening_point().z
        return self._logistic(
            value=clearance / self.receiver.root.collision.scale.z,
            sharpness=self.receiver.height_gate_sharpness,
        )

    def overlap_gate(self) -> sm.Scalar:
        """
        :return: Symbolic factor of the gate closing as the stream lands further from
            the receiver's opening, measured in opening radii.
        """
        landing_point = self.landing_point()
        opening = self.receiver.opening_point()
        landing_distance = sm.sqrt(
            (landing_point.x - opening.x) ** 2 + (landing_point.y - opening.y) ** 2
        )
        opening_radius = self.receiver.opening_radius
        return self._logistic(
            value=(opening_radius - landing_distance) / opening_radius,
            sharpness=self.receiver.overlap_gate_sharpness,
        )

    def inflow_velocity(self) -> sm.Scalar:
        """
        :return: Symbolic rate of the receiver's normalized fill level from the gated
            volume the source loses.
        """
        return self.gate() * self.source.outflow_volume_rate() / self.receiver.capacity

    @staticmethod
    def _logistic(*, value: sm.Scalar, sharpness: float) -> sm.Scalar:
        """
        :param value: The symbolic margin being gated.
        :param sharpness: Steepness of the transition around ``value == 0``.
        :return: Smooth step in ``[0, 1]`` approaching ``1`` as ``value`` grows
            positive.
        """
        return sm.Scalar(1.0) / (sm.Scalar(1.0) + sm.exp(-sharpness * value))
