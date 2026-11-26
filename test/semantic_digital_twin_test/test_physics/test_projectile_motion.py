"""
Tests of the flight of a body under gravity alone.
"""

from __future__ import annotations

import math

import pytest
from krrood_spatial_math.spatial_types import Point3, Vector3
from semantic_digital_twin.physics.equations.projectile_motion import ProjectileMotion

# %% landing point


def test_a_horizontal_throw_lands_after_the_fall_time() -> None:
    """
    A body thrown horizontally lands on the plane after the time a fall through the drop
    takes, carried along by its launch velocity meanwhile.
    """
    launch_x, launch_y, launch_z = 0.1, -0.2, 0.5
    speed_x, speed_y = 0.3, 0.1
    plane_height = 0.2
    flight = ProjectileMotion(
        launch_point=Point3(x=launch_x, y=launch_y, z=launch_z),
        launch_velocity=Vector3(x=speed_x, y=speed_y, z=0.0),
    )

    landing = flight.landing_point(plane_height=plane_height).evaluate()[:3]

    fall_time = math.sqrt(
        2 * (launch_z - plane_height) / ProjectileMotion.standard_gravity
    )
    assert landing == pytest.approx(
        [
            launch_x + speed_x * fall_time,
            launch_y + speed_y * fall_time,
            plane_height,
        ]
    )
