import os
from copy import deepcopy

import numpy as np
import pytest
import trimesh.boolean

from krrood.symbolic_math.symbolic_math import FloatVariable
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import (
    SpatialTypeNotJsonSerializable,
    WorldEntityWithIDNotInKwargs,
)
from semantic_digital_twin.spatial_types import (
    Point3,
    Vector3,
    Quaternion,
    RotationMatrix,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


def test_body_json_serialization():
    world = World()
    body = Body(name=PrefixedName("body"))
    collision = [
        Box(origin=HomogeneousTransformationMatrix.from_xyz_rpy(0, 1, 0, 0, 0, 1, body))
    ]
    body.collision = ShapeCollection(collision, reference_frame=body)
    body.collision_config.max_avoided_bodies = 69
    body.collision_config.disabled = False
    body.collision_config.buffer_zone_distance = 1.227
    body.collision_config.violated_distance = 0.23

    with world.modify_world():
        world.add_kinematic_structure_entity(body)

    other_world = deepcopy(world)

    json_data = body.to_json()
    tracker = WorldEntityWithIDKwargsTracker.from_world(other_world)
    body2 = Body.from_json(json_data, **tracker.create_kwargs())

    assert body2.index is not None
    assert body2 is other_world.get_world_entity_with_id_by_id(body2.id)

    for c1 in body.collision:
        for c2 in body2.collision:
            assert c1 == c2

    assert (
        body.collision.shapes[0].origin.reference_frame
        == body2.collision.shapes[0].origin.reference_frame
    )

    assert (
        body.collision.shapes[0].origin.child_frame
        == body2.collision.shapes[0].origin.child_frame
    )

    assert id(body.collision.shapes[0].origin.reference_frame) != id(
        body2.collision.shapes[0].origin.reference_frame
    )

    assert body == body2

    assert body.collision_config == body2.collision_config


def test_transformation_matrix_json_serialization():
    body = Body(name=PrefixedName("body"))
    body2 = Body(name=PrefixedName("body2"))
    transform = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=1, y=2, z=3, roll=1, pitch=2, yaw=3, reference_frame=body, child_frame=body2
    )
    json_data = transform.to_json()
    kwargs = {}
    tracker = WorldEntityWithIDKwargsTracker.from_kwargs(kwargs)
    tracker.add_world_entity_with_id(body)
    tracker.add_world_entity_with_id(body2)
    transform_copy = HomogeneousTransformationMatrix.from_json(json_data, **kwargs)
    assert transform.reference_frame == transform_copy.reference_frame
    assert id(transform.reference_frame) == id(transform_copy.reference_frame)
    assert np.allclose(transform.to_np(), transform_copy.to_np())


def test_point3_json_serialization():
    body = Body(name=PrefixedName("body"))
    point = Point3(1, 2, 3, reference_frame=body)
    json_data = point.to_json()
    kwargs = {}
    tracker = WorldEntityWithIDKwargsTracker.from_kwargs(kwargs)
    tracker.add_world_entity_with_id(body)
    point_copy = Point3.from_json(json_data, **kwargs)
    assert point.reference_frame == point_copy.reference_frame
    assert id(point.reference_frame) == id(point_copy.reference_frame)
    assert np.allclose(point.to_np(), point_copy.to_np())


def test_point3_json_serialization_with_expression():
    body = Body(name=PrefixedName("body"))
    point = Point3(FloatVariable(name="muh"), reference_frame=body)
    with pytest.raises(SpatialTypeNotJsonSerializable):
        point.to_json()


def test_KinematicStructureEntityNotInKwargs():
    body = Body(name=PrefixedName("body"))
    point = Point3(1, 2, 3, reference_frame=body)
    json_data = point.to_json()
    kwargs = {}
    with pytest.raises(WorldEntityWithIDNotInKwargs):
        Point3.from_json(json_data, **kwargs)


def test_KinematicStructureEntityNotInKwargs2():
    body = Body(name=PrefixedName("body"))
    point = Point3(1, 2, 3, reference_frame=body)
    json_data = point.to_json()
    tracker = WorldEntityWithIDKwargsTracker.from_world(World())
    with pytest.raises(WorldEntityWithIDNotInKwargs):
        Point3.from_json(json_data, **tracker.create_kwargs())


def test_vector3_json_serialization_with_expression():
    body = Body(name=PrefixedName("body"))
    vector = Vector3(FloatVariable(name="muh"), reference_frame=body)
    with pytest.raises(SpatialTypeNotJsonSerializable):
        vector.to_json()


def test_quaternion_json_serialization_with_expression():
    body = Body(name=PrefixedName("body"))
    quaternion = Quaternion(FloatVariable(name="muh"), reference_frame=body)
    with pytest.raises(SpatialTypeNotJsonSerializable):
        quaternion.to_json()


def test_rotation_matrix_json_serialization_with_expression():
    body = Body(name=PrefixedName("body"))
    rotation = RotationMatrix.from_rpy(
        roll=FloatVariable(name="muh"), reference_frame=body
    )
    with pytest.raises(SpatialTypeNotJsonSerializable):
        rotation.to_json()


def test_transformation_matrix_json_serialization_with_expression():
    body = Body(name=PrefixedName("body"))
    transform = HomogeneousTransformationMatrix.from_xyz_rpy(
        FloatVariable(name="muh"), reference_frame=body
    )
    with pytest.raises(SpatialTypeNotJsonSerializable):
        transform.to_json()


def test_vector3_json_serialization():
    body = Body(name=PrefixedName("body"))
    vector = Vector3(1, 2, 3, reference_frame=body)
    json_data = vector.to_json()
    kwargs = {}
    tracker = WorldEntityWithIDKwargsTracker.from_kwargs(kwargs)
    tracker.add_world_entity_with_id(body)
    vector_copy = Vector3.from_json(json_data, **kwargs)
    assert vector.reference_frame == vector_copy.reference_frame
    assert id(vector.reference_frame) == id(vector_copy.reference_frame)
    assert np.allclose(vector.to_np(), vector_copy.to_np())


def test_quaternion_json_serialization():
    body = Body(name=PrefixedName("body"))
    quaternion = Quaternion(1, 0, 0, 0, reference_frame=body)
    json_data = quaternion.to_json()
    kwargs = {}
    tracker = WorldEntityWithIDKwargsTracker.from_kwargs(kwargs)
    tracker.add_world_entity_with_id(body)
    quaternion_copy = Quaternion.from_json(json_data, **kwargs)
    assert quaternion.reference_frame == quaternion_copy.reference_frame
    assert id(quaternion.reference_frame) == id(quaternion_copy.reference_frame)
    assert np.allclose(quaternion.to_np(), quaternion_copy.to_np())


def test_rotation_matrix_json_serialization():
    body = Body(name=PrefixedName("body"))
    rotation = RotationMatrix.from_rpy(roll=1, pitch=2, yaw=3, reference_frame=body)
    json_data = rotation.to_json()
    kwargs = {}
    tracker = WorldEntityWithIDKwargsTracker.from_kwargs(kwargs)
    tracker.add_world_entity_with_id(body)
    rotation_copy = RotationMatrix.from_json(json_data, **kwargs)
    assert rotation.reference_frame == rotation_copy.reference_frame
    assert id(rotation.reference_frame) == id(rotation_copy.reference_frame)
    assert np.allclose(rotation.to_np(), rotation_copy.to_np())


def test_pose_json_serialization():
    body = Body(name=PrefixedName("body"))
    pose = Pose.from_xyz_rpy(
        x=4, y=5, z=7, roll=1, pitch=2, yaw=3, reference_frame=body
    )
    json_data = pose.to_json()
    kwargs = {}
    tracker = WorldEntityWithIDKwargsTracker.from_kwargs(kwargs)
    tracker.add_world_entity_with_id(body)
    pose_copy = Pose.from_json(json_data, **kwargs)
    assert pose.reference_frame == pose_copy.reference_frame
    assert id(pose.reference_frame) == id(pose_copy.reference_frame)
    assert np.allclose(pose, pose_copy)


def test_connection_json_serialization_with_world():
    world = World()
    body = Body(name=PrefixedName("body"))
    body2 = Body(name=PrefixedName("body2"))
    with world.modify_world():
        world.add_kinematic_structure_entity(body)
        world.add_kinematic_structure_entity(body2)
        c = FixedConnection(
            parent=body,
            child=body2,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=1, roll=2, reference_frame=body, child_frame=body2
            ),
        )
        world.add_connection(c)
    json_data = c.to_json()
    tracker = WorldEntityWithIDKwargsTracker.from_world(world)
    c2 = FixedConnection.from_json(json_data, **tracker.create_kwargs())
    assert c == c2
    assert c._world != c2._world
    assert c.parent.name == c2.parent.name
    assert c.child.name == c2.child.name
    assert np.allclose(
        c.parent_T_connection_expression.to_np(),
        c2.parent_T_connection_expression.to_np(),
    )
    assert (
        c.parent_T_connection_expression.reference_frame
        == c2.parent_T_connection_expression.reference_frame
    )
    assert (
        c.parent_T_connection_expression.child_frame
        == c2.parent_T_connection_expression.child_frame
    )


def test_transformation_matrix_json_serialization_with_world_in_kwargs():
    world = World()
    body = Body(name=PrefixedName("body"))
    body2 = Body(name=PrefixedName("body2"))
    with world.modify_world():
        world.add_kinematic_structure_entity(body)
        world.add_kinematic_structure_entity(body2)
        c = FixedConnection(
            parent=body,
            child=body2,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=1, roll=2, reference_frame=body, child_frame=body2
            ),
        )
        world.add_connection(c)
    json_data = c.to_json()
    tracker = WorldEntityWithIDKwargsTracker.from_world(world)
    c2 = FixedConnection.from_json(json_data, **tracker.create_kwargs())
    assert c == c2
    assert c._world != c2._world
    assert c.parent.name == c2.parent.name
    assert c.child.name == c2.child.name
    assert np.allclose(
        c.parent_T_connection_expression.to_np(),
        c2.parent_T_connection_expression.to_np(),
    )
    assert (
        c.parent_T_connection_expression.reference_frame
        == c2.parent_T_connection_expression.reference_frame
    )
    assert (
        c.parent_T_connection_expression.child_frame
        == c2.parent_T_connection_expression.child_frame
    )


def test_json_serialization_with_mesh():
    body: Body = (
        STLParser(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "semantic_digital_twin",
                "resources",
                "stl",
                "milk.stl",
            )
        )
        .parse()
        .root
    )

    json_data = body.to_json()
    body2 = Body.from_json(json_data)

    for c1 in body.collision:
        for c2 in body2.collision:
            assert (trimesh.boolean.difference([c1.mesh, c2.mesh])).is_empty
