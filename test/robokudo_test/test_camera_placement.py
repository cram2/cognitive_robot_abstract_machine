"""
World placement of directly configured semantic cameras.
"""

from __future__ import annotations

# %% Imports
import numpy as np
import pytest

from robokudo.world_descriptor import BaseWorldDescriptor, CameraPlacement
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body
from ..semantic_digital_twin_test.conftest import table_world

# %% Camera placement fixture


@pytest.fixture
def camera_placement() -> CameraPlacement:
    """
    Provide a camera with a distinct annotation name and a detached optical body.
    """
    camera = Camera(
        name=PrefixedName(name="table_camera"),
        root=Body(name=PrefixedName(name="custom_optical_frame")),
        camera_model=PinholeCameraModel.from_field_of_view(
            resolution=CameraResolution(), field_of_view=FieldOfView()
        ),
    )
    return CameraPlacement(
        camera=camera,
        parent_T_camera=HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0.2, y=0.3, z=0.4, roll=0.5, pitch=0.6, yaw=0.7
        ),
    )


# %% Camera identity


def test_camera_placement_registers_supplied_annotation_and_root(
    table_world: World,
    camera_placement: CameraPlacement,
) -> None:
    """
    Building a camera preserves the supplied annotation and optical body's identity.
    """
    root = camera_placement.camera.root
    camera = BaseWorldDescriptor(world=table_world).build_camera(
        table_world.root, camera_placement
    )
    assert camera is camera_placement.camera
    assert camera.root is root
    assert table_world.get_semantic_annotation_by_id(camera.id) is camera


# %% Optical poses


def test_camera_placement_uses_parent_relative_optical_pose(
    table_world: World,
    camera_placement: CameraPlacement,
) -> None:
    """
    The supplied placement directly defines the camera's optical pose.
    """
    camera = BaseWorldDescriptor(world=table_world).build_camera(
        table_world.root, camera_placement
    )
    np.testing.assert_allclose(
        camera.root.global_transform.to_np(),
        table_world.root.global_transform.to_np()
        @ camera_placement.parent_T_camera.to_np(),
        atol=1e-12,
    )
    assert camera.root.parent_connection.parent is table_world.root


def test_camera_placement_follows_parent_motion(
    table_world: World,
    camera_placement: CameraPlacement,
) -> None:
    """
    A mounted camera follows its parent while preserving the configured optical offset.
    """
    parent = Body(name=PrefixedName(name="mount"))
    with table_world.modify_world():
        connection = Connection6DoF.create_with_dofs(
            parent=table_world.root, child=parent, world=table_world
        )
        table_world.add_connection(connection)
    camera = BaseWorldDescriptor(world=table_world).build_camera(
        parent, camera_placement
    )
    with table_world.modify_world():
        connection.origin = HomogeneousTransformationMatrix(
            data=HomogeneousTransformationMatrix.from_xyz_rpy(y=0.8, pitch=0.7),
            reference_frame=table_world.root,
            child_frame=parent,
        )
    np.testing.assert_allclose(
        camera.root.global_transform.to_np(),
        parent.global_transform.to_np() @ camera_placement.parent_T_camera.to_np(),
        atol=1e-12,
    )
    assert camera.root.parent_connection.parent is parent
