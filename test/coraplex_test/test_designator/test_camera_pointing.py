"""
Camera pointing follows the optical frame's viewing direction.
"""

from __future__ import annotations

# %% Imports
from unittest.mock import Mock

import numpy as np
import pytest

import coraplex.robot_plans.motions.robot_body as robot_body
from coraplex.robot_plans.motions.robot_body import LookingMotion
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.spatial_types import Pose, Vector3
from ...semantic_digital_twin_test.conftest import table_world
from semantic_digital_twin.world import World

# %% Optical pointing


def test_looking_motion_uses_optical_forward_direction(
    table_world: World,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Pointing uses the optical forward axis expressed in the optical camera root.
    """
    camera = Camera(
        name=PrefixedName(name="camera"),
        root=table_world.root,
        camera_model=PinholeCameraModel.from_field_of_view(
            resolution=CameraResolution(), field_of_view=FieldOfView()
        ),
    )
    robot = Mock()
    robot.get_torso.return_value.root = table_world.root
    monkeypatch.setattr(LookingMotion, "robot", property(lambda self: robot))
    pointing = Mock()
    monkeypatch.setattr(robot_body, "Pointing", pointing)
    motion = LookingMotion(target=Pose(), camera=camera)
    motion._motion_chart
    expected = Vector3.Z(reference_frame=camera.root)
    np.testing.assert_allclose(
        pointing.call_args.kwargs["pointing_axis"].to_np(),
        expected.to_np(),
        atol=1e-12,
    )
