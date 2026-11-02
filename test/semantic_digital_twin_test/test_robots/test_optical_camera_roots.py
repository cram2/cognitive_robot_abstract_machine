"""
Optical camera roots and robot sensor selection.
"""

from __future__ import annotations

# %% Imports
import numpy as np
import pytest

from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.garmi import GarmiCamera
from semantic_digital_twin.robots.justin import JustinCamera
from semantic_digital_twin.robots.icub3 import ICub3Camera
from semantic_digital_twin.robots.daisy import DAiSyCamera
from semantic_digital_twin.robots.armar7 import AzureKinectRGB
from semantic_digital_twin.robots.unitree_g1 import D435
from semantic_digital_twin.robots.robot_parts import Camera, RobotCamera
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.testing import ray_test_world
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world import World

# %% Optical contract


def test_camera_root_defines_optical_forward_direction(table_world: World) -> None:
    """
    A camera needs no separate viewing axis or optical offset.
    """
    camera = Camera(
        name=PrefixedName("camera"),
        root=table_world.root,
        camera_model=PinholeCameraModel.from_field_of_view(
            resolution=CameraResolution(), field_of_view=FieldOfView()
        ),
    )
    np.testing.assert_array_equal(camera.viewing_direction.to_np(), Vector3.Z().to_np())
    assert camera.viewing_direction.reference_frame is camera.root


# %% Mounted roots


def test_optical_root_preserves_mounting_pose(table_world: World) -> None:
    """
    An optical body stores its full fixed mounting pose in the world graph.
    """
    parent_T_optical = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=0.1,
        y=0.2,
        z=0.3,
        roll=0.4,
        pitch=0.5,
        yaw=0.6,
        reference_frame=table_world.root,
    )
    optical_root = RobotCamera.create_optical_body(table_world.root, parent_T_optical)
    np.testing.assert_allclose(
        optical_root.global_transform.to_np(),
        table_world.root.global_transform.to_np() @ parent_T_optical.to_np(),
        atol=1e-12,
    )
    assert optical_root.parent_connection.parent is table_world.root
    assert optical_root.collision.shapes == []


# %% Approximate sensor mounts
@pytest.mark.parametrize(
    "camera_type",
    [GarmiCamera, JustinCamera, ICub3Camera, DAiSyCamera, AzureKinectRGB, D435],
)
def test_approximate_camera_keeps_its_modeled_mount(
    camera_type: type[RobotCamera],
    table_world: World,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Cameras without sensor frames retain the existing mount with fixed optical axes.
    """
    monkeypatch.setattr(
        table_world, "get_body_in_branch_by_name", lambda root, name: table_world.root
    )
    camera = camera_type.setup_default_configuration_in_world_below_robot_root(
        table_world.root
    )
    assert camera.root.parent_connection.parent is table_world.root
    expected = np.eye(4)
    if camera_type is GarmiCamera:
        expected[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    np.testing.assert_allclose(
        camera.root.global_transform.to_np(),
        table_world.root.global_transform.to_np() @ expected,
        atol=1e-12,
    )


# %% Parent motion


def test_optical_mount_follows_parent_motion(
    ray_test_world: tuple[World, Body, Body, Body, Body],
) -> None:
    """
    Moving the mounting body carries its fixed optical frame without changing roll.
    """
    world, parent, *_ = ray_test_world
    parent_T_optical = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=0.2, roll=0.3, yaw=0.4
    )
    optical_root = RobotCamera.create_optical_body(parent, parent_T_optical)
    with world.modify_world():
        parent.parent_connection.origin = HomogeneousTransformationMatrix(
            data=HomogeneousTransformationMatrix.from_xyz_rpy(
                y=0.5, roll=0.6, pitch=0.7
            ),
            reference_frame=parent.parent_connection.parent,
            child_frame=parent,
        )
    np.testing.assert_allclose(
        optical_root.global_transform.to_np(),
        parent.global_transform.to_np() @ parent_T_optical.to_np(),
        atol=1e-12,
    )
