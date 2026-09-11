import json
import os
from copy import deepcopy

import numpy as np
import pytest
import trimesh.boolean

from krrood.adapters.exceptions import UntrackedObjectError
from krrood.adapters.json_serializer import from_json, to_json
from krrood.symbolic_math.exceptions import SymbolicMathNotJsonSerializableError
from krrood.symbolic_math.symbolic_math import FloatVariable
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import (
    SpatialTypeNotJsonSerializable,
    WorldEntityWithIDNotInKwargs,
    MissingWorldError,
)
from semantic_digital_twin.spatial_types import (
    Point3,
    Vector3,
    Quaternion,
    RotationMatrix,
)
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point2,
    Pose,
    Pose2D,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    ScrewConnection,
    RevoluteConnection,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Box
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    Body,
    ReferencedWorldEntity,
    WorldEntityReferenceWriter,
)


def test_body_json_serialization():
    world = World()
    body = Body(name=PrefixedName("body"))
    collision = [
        Box(origin=HomogeneousTransformationMatrix.from_xyz_rpy(0, 1, 0, 0, 0, 1, body))
    ]
    body.collision = ShapeCollection(collision, reference_frame=body)

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


def test_dof_hardware_interface_serialization():
    dof = DegreeOfFreedom(has_hardware_interface=True)
    rebuilt_dof = from_json(to_json(dof))

    assert dof.has_hardware_interface == rebuilt_dof.has_hardware_interface
    assert dof == rebuilt_dof


def test_transformation_matrix_json_serialization():
    body = Body(name=PrefixedName("body"))
    body2 = Body(name=PrefixedName("body2"))
    transform = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=1, y=2, z=3, roll=1, pitch=2, yaw=3, reference_frame=body, child_frame=body2
    )
    json_data = transform.to_json()
    kwargs = {}
    tracker = WorldEntityWithIDKwargsTracker.from_kwargs(kwargs)
    tracker.add(body.id, body)
    tracker.add(body2.id, body2)
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
    tracker.add(body.id, body)
    point_copy = Point3.from_json(json_data, **kwargs)
    assert point.reference_frame == point_copy.reference_frame
    assert id(point.reference_frame) == id(point_copy.reference_frame)
    assert np.allclose(point.to_np(), point_copy.to_np())


def test_point3_json_serialization_with_expression():
    body = Body(name=PrefixedName("body"))
    point = Point3(f := FloatVariable(name="muh"), reference_frame=body)
    with pytest.raises(SpatialTypeNotJsonSerializable):
        point.to_json()


def test_spatial_type_refuses_an_expression_like_any_symbolic_math_value():
    body = Body(name=PrefixedName("body"))
    point = Point3(FloatVariable(name="muh"), reference_frame=body)

    with pytest.raises(SymbolicMathNotJsonSerializableError) as error:
        point.to_json()

    assert error.value.expression is point


def test_KinematicStructureEntityNotInKwargs():
    body = Body(name=PrefixedName("body"))
    point = Point3(1, 2, 3, reference_frame=body)
    json_data = point.to_json()
    kwargs = {}
    with pytest.raises(MissingWorldError):
        Point3.from_json(json_data, **kwargs)


def test_KinematicStructureEntityNotInKwargs2():
    body = Body(name=PrefixedName("body"))
    point = Point3(1, 2, 3, reference_frame=body)
    json_data = point.to_json()
    tracker = WorldEntityWithIDKwargsTracker.from_world(World())
    with pytest.raises(WorldEntityWithIDNotInKwargs) as raised:
        Point3.from_json(json_data, **tracker.create_kwargs())

    assert raised.value.key == body.id
    assert raised.value.world_entity_name == body.name


def test_an_entity_a_reference_cannot_be_resolved_to_is_named():
    """
    A reference says which entity it means, so that a world missing that entity reports
    more than an id nobody can look up.
    """
    parent = Body(name=PrefixedName("cable_post"))
    child = Body(name=PrefixedName("cable_hanger"))
    connection = FixedConnection(parent=parent, child=child)
    json_data = connection.to_json()

    tracker = WorldEntityWithIDKwargsTracker.from_world(World())
    tracker.add(child.id, child)
    with pytest.raises(WorldEntityWithIDNotInKwargs) as raised:
        FixedConnection.from_json(json_data, **tracker.create_kwargs())

    assert raised.value.key == parent.id
    assert raised.value.world_entity_name == parent.name


def test_a_reference_resolves_to_the_entity_it_names():
    """
    The name a reference carries is context for a reader; the entity itself is still
    found through the id it was written with.
    """
    parent = Body(name=PrefixedName("cable_post"))
    child = Body(name=PrefixedName("cable_hanger"))
    connection = FixedConnection(parent=parent, child=child)
    json_data = connection.to_json()

    tracker = WorldEntityWithIDKwargsTracker.from_world(World())
    tracker.add(parent.id, parent)
    tracker.add(child.id, child)
    parsed_connection = FixedConnection.from_json(json_data, **tracker.create_kwargs())

    assert parsed_connection.parent is parent
    assert parsed_connection.child is child


def test_an_entity_written_as_a_reference_carries_only_its_id_and_name():
    body = Body(name=PrefixedName("cable_post"))

    json_data = to_json(body, **WorldEntityReferenceWriter().create_kwargs())

    assert json_data == to_json(ReferencedWorldEntity(id=body.id, name=body.name))


def test_an_entity_written_as_a_reference_is_read_as_the_entity_of_the_world():
    world = World()
    body = Body(name=PrefixedName("cable_post"))
    with world.modify_world():
        world.add_kinematic_structure_entity(body)
    json_data = json.loads(
        json.dumps(to_json(body, **WorldEntityReferenceWriter().create_kwargs()))
    )

    tracker = WorldEntityWithIDKwargsTracker.from_world(world)

    assert from_json(json_data, **tracker.create_kwargs()) is body


def test_a_reference_to_an_entity_cannot_be_read_without_a_world():
    body = Body(name=PrefixedName("cable_post"))
    json_data = to_json(body, **WorldEntityReferenceWriter().create_kwargs())

    with pytest.raises(MissingWorldError):
        from_json(json_data)


def test_a_reference_to_an_entity_the_world_lacks_names_the_entity():
    body = Body(name=PrefixedName("cable_post"))
    json_data = to_json(body, **WorldEntityReferenceWriter().create_kwargs())

    tracker = WorldEntityWithIDKwargsTracker.from_world(World())
    with pytest.raises(WorldEntityWithIDNotInKwargs) as raised:
        from_json(json_data, **tracker.create_kwargs())

    assert raised.value.key == body.id
    assert raised.value.world_entity_name == body.name


def test_a_connection_written_as_a_reference_carries_only_its_id_and_name():
    world = World()
    parent = Body(name=PrefixedName("cable_post"))
    child = Body(name=PrefixedName("cable"))
    with world.modify_world():
        world.add_connection(connection := FixedConnection(parent=parent, child=child))

    json_data = to_json(connection, **WorldEntityReferenceWriter().create_kwargs())

    assert json_data == WorldEntityReferenceWriter().write_reference(connection)


def test_a_connection_written_as_a_reference_is_read_as_the_connection_of_the_world():
    world = World()
    parent = Body(name=PrefixedName("cable_post"))
    child = Body(name=PrefixedName("cable"))
    with world.modify_world():
        world.add_connection(connection := FixedConnection(parent=parent, child=child))
    json_data = json.loads(
        json.dumps(to_json(connection, **WorldEntityReferenceWriter().create_kwargs()))
    )

    tracker = WorldEntityWithIDKwargsTracker.from_world(world)

    assert from_json(json_data, **tracker.create_kwargs()) is connection


def test_world_entity_missing_from_the_world_is_an_untracked_object():
    body = Body(name=PrefixedName("body"))
    point = Point3(1, 2, 3, reference_frame=body)
    json_data = point.to_json()
    tracker = WorldEntityWithIDKwargsTracker.from_world(World())
    with pytest.raises(UntrackedObjectError):
        Point3.from_json(json_data, **tracker.create_kwargs())


def test_vector3_json_serialization_with_expression():
    body = Body(name=PrefixedName("body"))
    vector = Vector3(f := FloatVariable(name="muh"), reference_frame=body)
    with pytest.raises(SpatialTypeNotJsonSerializable):
        vector.to_json()


def test_quaternion_json_serialization_with_expression():
    body = Body(name=PrefixedName("body"))
    quaternion = Quaternion(f := FloatVariable(name="muh"), reference_frame=body)
    with pytest.raises(SpatialTypeNotJsonSerializable):
        quaternion.to_json()


def test_rotation_matrix_json_serialization_with_expression():
    body = Body(name=PrefixedName("body"))
    f = FloatVariable(name="muh")
    rotation = RotationMatrix.from_rpy(roll=f, reference_frame=body)
    with pytest.raises(SpatialTypeNotJsonSerializable):
        rotation.to_json()


def test_transformation_matrix_json_serialization_with_expression():
    body = Body(name=PrefixedName("body"))
    transform = HomogeneousTransformationMatrix.from_xyz_rpy(
        f := FloatVariable(name="muh"), reference_frame=body
    )
    with pytest.raises(SpatialTypeNotJsonSerializable):
        transform.to_json()


def test_vector3_json_serialization():
    body = Body(name=PrefixedName("body"))
    vector = Vector3(1, 2, 3, reference_frame=body)
    json_data = vector.to_json()
    kwargs = {}
    tracker = WorldEntityWithIDKwargsTracker.from_kwargs(kwargs)
    tracker.add(body.id, body)
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
    tracker.add(body.id, body)
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
    tracker.add(body.id, body)
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
    tracker.add(body.id, body)
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


def test_screw_connection_json_serialization_with_world():
    world = World()
    body = Body(name=PrefixedName("body"))
    body2 = Body(name=PrefixedName("body2"))
    screw_pitch = 0.005
    with world.modify_world():
        world.add_kinematic_structure_entity(body)
        world.add_kinematic_structure_entity(body2)
        connection = ScrewConnection.create_with_dofs(
            world,
            body,
            body2,
            axis=Vector3.Z(),
            screw_pitch=screw_pitch,
            multiplier=2.0,
            offset=0.1,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=1, reference_frame=body, child_frame=body2
            ),
        )
        world.add_connection(connection)
    json_data = connection.to_json()
    tracker = WorldEntityWithIDKwargsTracker.from_world(world)
    restored_connection = ScrewConnection.from_json(
        json_data, **tracker.create_kwargs()
    )
    assert connection == restored_connection
    assert restored_connection.screw_pitch == screw_pitch
    assert restored_connection.multiplier == connection.multiplier
    assert restored_connection.offset == connection.offset
    assert np.allclose(restored_connection.axis.to_np(), connection.axis.to_np())
    assert restored_connection.raw_dof.id == connection.raw_dof.id


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

    json_data = to_json(body)
    body2 = from_json(json_data)

    for c1 in body.collision:
        for c2 in body2.collision:
            difference = trimesh.boolean.difference([c1.mesh, c2.mesh])
            # The mesh is re-exported as OBJ during serialization, and OBJ stores
            # coordinates as decimal text, so the round-trip loses a few nanometres
            # of precision. The boolean difference is therefore a vanishingly thin
            # shell rather than exactly empty; treat a negligible residual volume as
            # geometrically identical.
            assert difference.is_empty or difference.volume < c1.mesh.volume * 1e-3


# %% connection references survive same-name ambiguity


def _world_with_two_equally_named_connections() -> (
    tuple[World, RevoluteConnection, RevoluteConnection]
):
    """
    A world holding two revolute connections that share one name.

    Merging two instances of the same robot description produces exactly this, since
    every entity is named after the description it was parsed from.
    """
    world = World.create_with_root_body("root")
    connections = []
    for index in range(2):
        with world.modify_world():
            child = Body(name=PrefixedName("link", prefix=f"branch_{index}"))
            connection = RevoluteConnection.create_with_dofs(
                world=world,
                parent=world.root,
                child=child,
                axis=Vector3.Z(),
                name=PrefixedName("shared_joint"),
            )
            world.add_connection(connection)
        connections.append(connection)
    return world, connections[0], connections[1]


def test_joint_state_resolves_the_connection_it_was_built_from():
    world, first_connection, second_connection = (
        _world_with_two_equally_named_connections()
    )
    joint_state = JointState.from_mapping(
        mapping={second_connection: 0.5}, name=PrefixedName("state")
    )

    tracker = WorldEntityWithIDKwargsTracker.from_world(world)
    reconstructed = JointState.from_json(
        joint_state.to_json(), **tracker.create_kwargs()
    )

    assert reconstructed.connections == [second_connection]
    assert reconstructed.connections[0] is not first_connection


# %% json round trips of everything that writes its own json


@pytest.mark.parametrize(
    "spatial_type",
    [
        Point3(1.0, 2.0, 3.0),
        Point2(1.0, 2.0),
        Vector3(1.0, 2.0, 3.0),
        Quaternion(0.0, 0.0, 0.0, 1.0),
        RotationMatrix.from_rpy(0.1, 0.2, 0.3),
        HomogeneousTransformationMatrix.from_xyz_rpy(1.0, 2.0, 3.0, 0.1, 0.2, 0.3),
        Pose.from_xyz_quaternion(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0),
        Pose2D(1.0, 2.0, 0.3),
    ],
)
def test_a_spatial_type_survives_a_json_round_trip(spatial_type):
    """
    Each spatial type writes the numbers it carries itself and reads them back itself,
    so the two halves can drift apart with nothing else noticing.
    """
    payload = spatial_type.to_json()

    restored = from_json(payload)

    # The keys say that both halves talk about the same parts; the numbers say that the
    # parts arrived. They are compared with a tolerance because a rotation travels as a
    # quaternion and comes back through a conversion.
    assert sorted(restored.to_json()) == sorted(payload)
    np.testing.assert_allclose(restored.to_np(), spatial_type.to_np(), atol=1e-12)


def test_a_spatial_type_carries_the_frame_it_is_expressed_in():
    """
    A spatial type means nothing without the frame it is expressed in, so the frame
    travels with it and is resolved back to the entity of the reading world.
    """
    body = Body(name=PrefixedName("reference"))
    point = Point3(1.0, 2.0, 3.0, reference_frame=body)
    payload = point.to_json()

    tracker = WorldEntityWithIDKwargsTracker.from_world(World())
    tracker.add(body.id, body)
    restored = from_json(payload, **tracker.create_kwargs())

    assert restored.reference_frame is body


def test_a_degree_of_freedom_survives_a_json_round_trip():
    """
    A degree of freedom carries its limits into json, which is the part of it a world
    cannot recompute.
    """
    limits = DegreeOfFreedomLimits(
        lower=DerivativeMap(position=-1.5), upper=DerivativeMap(position=1.5)
    )
    degree_of_freedom = DegreeOfFreedom(name=PrefixedName("joint"), limits=limits)
    payload = degree_of_freedom.to_json()

    tracker = WorldEntityWithIDKwargsTracker.from_world(World())
    restored = from_json(payload, **tracker.create_kwargs())

    assert restored.id == degree_of_freedom.id
    assert restored.name == degree_of_freedom.name
    assert restored.limits.lower.position == limits.lower.position
    assert restored.limits.upper.position == limits.upper.position
    assert restored.to_json() == payload
