"""
Motion of a body that moves under gravity alone, without drag.
"""

from __future__ import annotations

from dataclasses import dataclass

import krrood.symbolic_math.symbolic_math as sm
from krrood.patterns.classproperty import classproperty
from krrood.symbolic_math.symbolic_math import Scalar
from krrood_spatial_math.spatial_types import Point3, Vector3

# %% projectile motion


@dataclass(kw_only=True)
class ProjectileMotion:
    r"""
    Flight of a body accelerated by gravity alone:

    .. math::

        p(t) = p_0 + v_0 t - \tfrac{1}{2} g t^2 e_z

    from :attr:`launch_point` :math:`p_0` with :attr:`launch_velocity` :math:`v_0`.
    """

    launch_point: Point3
    """
    Where the body leaves, expressed in a frame whose z axis points up.
    """

    launch_velocity: Vector3
    """
    Velocity the body leaves with, in the frame of :attr:`launch_point`, in metres per
    second.
    """

    @classproperty
    def standard_gravity(cls) -> float:
        """
        Standard gravitational acceleration, in metres per second squared.
        """
        return 9.80665

    @classmethod
    def speed_after_drop(cls, *, height: sm.ScalarData) -> Scalar:
        r"""
        :param height: Height the body falls from rest, in metres.
        :return: Symbolic speed the body reaches, :math:`\sqrt{2 g h}`, in metres per
            second.
        """
        return sm.sqrt(2 * cls.standard_gravity * height)

    def landing_point(
        self, *, plane_height: sm.ScalarData, minimum_drop_height: float = 0.01
    ) -> Point3:
        """
        :param plane_height: Height of the horizontal plane the body lands on, in the
            frame of :attr:`launch_point`.
        :param minimum_drop_height: Drop the flight time is computed from at least, in
            metres, which keeps its gradient bounded as the launch point nears the
            plane.
        :return: Symbolic point where the body, on its way down, crosses the plane.
        """
        gravity = self.standard_gravity
        drop_height = sm.max(
            sm.Scalar(minimum_drop_height), self.launch_point.z - plane_height
        )
        rising_speed = self.launch_velocity.z
        flight_time = (
            rising_speed + sm.sqrt(rising_speed**2 + 2 * gravity * drop_height)
        ) / gravity
        return Point3(
            x=self.launch_point.x + self.launch_velocity.x * flight_time,
            y=self.launch_point.y + self.launch_velocity.y * flight_time,
            z=plane_height,
            reference_frame=self.launch_point.reference_frame,
        )
