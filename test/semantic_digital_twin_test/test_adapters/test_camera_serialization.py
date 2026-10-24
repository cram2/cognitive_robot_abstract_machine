"""
Camera serialization preserves each sensor's root-relative viewing direction.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from typing_extensions import TYPE_CHECKING

from krrood.adapters.json_serializer import from_json
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.garmi import GarmiCamera
from semantic_digital_twin.robots.pr2 import PR2KinectV1
from semantic_digital_twin.spatial_types import Vector3

if TYPE_CHECKING:
    from semantic_digital_twin.robots.robot_parts import Camera
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.world_entity import Body


# %% camera direction and object roundtrip
@pytest.fixture(params=(PR2KinectV1, GarmiCamera), ids=("pr2", "garmi"))
def camera(
    request: pytest.FixtureRequest,
    world_with_two_bodies: tuple[World, Body, Body],
) -> Camera:
    """
    Place a concrete camera annotation on an existing world's named root body.

    :param request: Selects a camera class with its own viewing direction.
    :param world_with_two_bodies: Fresh world and bodies from the shared fixture.
    :return: Camera whose direction and reference frame are provided by its class.
    """
    world, root, _ = world_with_two_bodies
    root.name = PrefixedName("eye", prefix="robot")
    with world.modify_world():
        world.add_kinematic_structure_entity(root)
    camera_type: type[Camera] = request.param
    return camera_type(
        root=root,
        name=PrefixedName("inspection", prefix="sensors"),
        field_of_view=FieldOfView(horizontal_angle=0.99, vertical_angle=0.75),
        default_camera=True,
    )


def test_camera_json_contains_its_direction_and_reference_frame(camera: Camera) -> None:
    """
    Publish the camera's computed axis through the existing vector serializer.

    :param camera: Concrete camera annotation with a root-relative viewing direction.
    """
    payload = camera.to_json()
    direction = camera.forward_facing_axis

    assert payload["forward_facing_axis"] == direction.to_json()
    tracker = WorldEntityWithIDKwargsTracker.from_world(camera.root._world)
    restored_direction = Vector3.from_json(
        payload["forward_facing_axis"], **tracker.create_kwargs()
    )
    assert restored_direction.reference_frame is camera.root
    np.testing.assert_array_equal(restored_direction.to_np(), direction.to_np())


def test_camera_json_roundtrip_preserves_its_class_and_properties(
    camera: Camera,
) -> None:
    """
    Restore the concrete camera while retaining its world-body reference.

    :param camera: Camera whose fields and computed direction must survive JSON.
    """
    payload = json.loads(json.dumps(camera.to_json()))
    tracker = WorldEntityWithIDKwargsTracker.from_world(camera.root._world)

    restored = from_json(payload, **tracker.create_kwargs())

    assert type(restored) is type(camera)
    assert restored.root is camera.root
    assert restored.name == camera.name
    assert restored.field_of_view == camera.field_of_view
    assert restored.default_camera == camera.default_camera
    np.testing.assert_array_equal(
        restored.forward_facing_axis.to_np(), camera.forward_facing_axis.to_np()
    )
