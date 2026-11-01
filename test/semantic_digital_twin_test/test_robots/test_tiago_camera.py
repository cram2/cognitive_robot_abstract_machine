"""
Optical camera mounting for the MuJoCo Tiago.
"""

from __future__ import annotations

# %% Imports
import numpy as np
import pytest

from semantic_digital_twin.robots.robot_parts import RobotCamera
from semantic_digital_twin.robots.tiago import TiagoMujocoBody, TiagoMujocoCamera
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)

# %% Optical mount


def test_camera_optical_forward_follows_head_forward(
    table_world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Optical forward points along the head's positive X axis.
    """

    def find_head(root: KinematicStructureEntity, body_name: str) -> Body:
        """
        Return the mounting body requested by the camera.
        """
        assert root is table_world.root
        assert body_name is TiagoMujocoBody.HEAD
        return table_world.root

    monkeypatch.setattr(table_world, "get_body_in_branch_by_name", find_head)
    camera = TiagoMujocoCamera.setup_default_configuration_in_world_below_robot_root(
        table_world.root
    )

    assert isinstance(camera, RobotCamera)
    assert camera.root.parent_connection.parent is table_world.root
    np.testing.assert_allclose(
        camera.root.parent_connection.origin.to_np() @ camera.viewing_direction.to_np(),
        Vector3.X().to_np(),
    )
