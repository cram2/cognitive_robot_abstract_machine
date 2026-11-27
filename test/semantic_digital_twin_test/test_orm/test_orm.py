"""
Tests of storing worlds, their robots and semantic annotations in a database.
"""

from __future__ import annotations

from typing_extensions import TYPE_CHECKING
import dataclasses
import os
import time
from copy import deepcopy, copy

import numpy as np
from krrood.ormatic.utils import create_engine
from sqlalchemy import select
from sqlalchemy.orm import Session

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.orm.utils import semantic_digital_twin_sessionmaker
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.pr2 import PR2, PR2RightArm
from semantic_digital_twin.world_description.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ScrewAxisConnection
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Box, Scale, Color
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Actuator, Body
from semantic_digital_twin.world_description.inertial_properties import (
    Inertial,
    InertiaTensor,
)
from semantic_digital_twin.adapters.multi_sim import MujocoCamera
from krrood_spatial_math.spatial_types import Vector3
from semantic_digital_twin.specifications.connections import (
    ScrewAxisConnectionSpecification,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Door,
    EntryWay,
    Handle,
)
from semantic_digital_twin.semantic_annotations.part_whole.relationship_marker import (
    IsPartWholeRelationship,
)
from krrood.class_diagrams.class_diagram import WrappedClass

if TYPE_CHECKING:
    from semantic_digital_twin.orm.ormatic_interface import DegreeOfFreedomLimitsDAO
from krrood.ormatic.data_access_objects.helper import to_dao


import pytest


@pytest.fixture
def engine():
    return create_engine("sqlite:///:memory:")


@pytest.fixture
def session(engine):
    """
    :return: A session on *engine* holding semantic_digital_twin's tables, which are dropped
        again after the test.
    """
    from semantic_digital_twin.orm.ormatic_interface import Base

    session = Session(engine)
    Base.metadata.create_all(bind=session.bind)
    yield session
    Base.metadata.drop_all(session.bind)
    session.close()


def test_table_world(session, table_world):
    """
    A stored world keeps all its bodies and comes back with the state and kinematics of
    its revolute connection.
    """
    from semantic_digital_twin.orm.ormatic_interface import (
        KinematicStructureEntityDAO,
        WorldMappingDAO,
    )

    revolute_connection = [
        connection
        for connection in table_world.get_connections_by_type(ScrewAxisConnection)
        if connection.behaves_like_a_revolute_connection
    ][0]
    revolute_connection.position = 1
    revolute_connection.velocity = 23
    revolute_connection.acceleration = 42
    revolute_connection.jerk = 69
    fk = table_world.compute_forward_kinematics_np(
        root=revolute_connection.parent, tip=revolute_connection.child
    )
    world_dao: WorldMappingDAO = to_dao(table_world)

    session.add(world_dao)
    session.commit()

    bodies_from_db = session.scalars(select(KinematicStructureEntityDAO)).all()
    assert len(bodies_from_db) == len(table_world.kinematic_structure_entities)

    queried_world = session.scalar(select(WorldMappingDAO))
    reconstructed: World = queried_world.from_dao()

    fk2 = reconstructed.compute_forward_kinematics_np(
        root=revolute_connection.parent, tip=revolute_connection.child
    )
    assert np.allclose(fk, fk2)
    reconstructed_connection = [
        connection
        for connection in reconstructed.get_connections_by_type(ScrewAxisConnection)
        if connection.behaves_like_a_revolute_connection
    ][0]
    assert reconstructed_connection.position == revolute_connection.position
    assert reconstructed_connection.velocity == revolute_connection.velocity
    assert reconstructed_connection.acceleration == revolute_connection.acceleration
    assert reconstructed_connection.jerk == revolute_connection.jerk


def test_insert(session):
    """
    A body keeps the origin of its collision shapes in the database, and comes back as
    the reference frame of that origin.
    """
    from semantic_digital_twin.orm.ormatic_interface import BodyDAO, BoxDAO, ShapeDAO

    origin = HomogeneousTransformationMatrix.from_xyz_rpy(1, 2, 3, 1, 2, 3)
    scale = Scale(1.0, 1.0, 1.0)
    color = Color.from_rgba(red=0.0, green=1.0, blue=1.0)
    shape1 = Box(origin=origin, scale=scale, color=color)
    b1 = Body(name=PrefixedName("b1"), collision=ShapeCollection([shape1]))

    dao: BodyDAO = to_dao(b1)
    assert dao.collision.shapes[0].target.origin is not None

    session.add(dao)
    session.commit()
    queried_body = session.scalar(select(BodyDAO))
    assert queried_body.collision.shapes[0].target.origin is not None
    reconstructed_body = queried_body.from_dao()
    assert reconstructed_body is reconstructed_body.collision[0].origin.reference_frame

    result = session.scalar(select(ShapeDAO))
    assert isinstance(result, BoxDAO)
    box = result.from_dao()


@pytest.mark.skipif(
    os.getenv("SEMANTIC_DIGITAL_TWIN_DATABASE_URI") is None,
    reason="Permanent Database not available",
)
def test_sessionmaker():
    s = semantic_digital_twin_sessionmaker()()
    assert s is not None


def test_degree_of_freedom_limits(session):
    lower = DerivativeMap()
    lower.position = -2.0
    lower.jerk = 1.0

    upper = DerivativeMap()
    upper.position = 2.0
    upper.velocity = 3.0
    obj = DegreeOfFreedomLimits(lower=lower, upper=upper)
    dao: DegreeOfFreedomLimitsDAO = to_dao(obj)
    reconstructed = dao.from_dao()

    assert obj == reconstructed


def test_pr2_world(pr2_world_state_reset, session):
    """
    The PR2's world can be stored, read back and copied, and its revolute connections
    are stored as rows.
    """
    from semantic_digital_twin.orm.ormatic_interface import (
        ScrewAxisConnectionDAO,
        WorldMappingDAO,
    )

    dao: WorldMappingDAO = to_dao(pr2_world_state_reset)
    session.add(dao)
    session.commit()

    to_dao(pr2_world_state_reset).from_dao()

    queried_world = session.scalar(select(WorldMappingDAO))
    reconstructed: World = queried_world.from_dao()

    # confirm the modification history
    deepcopy(reconstructed)

    q = select(ScrewAxisConnectionDAO)
    r = session.scalars(q).all()
    assert len(r) > 0


def test_pr2_semantic_annotation_and_safe_to_db(
    rclpy_node, pr2_world_state_reset, session
):
    world_fetcher = pytest.importorskip(
        "semantic_digital_twin.adapters.ros.world_fetcher"
    )
    fetcher = world_fetcher.FetchWorldServer(
        node=rclpy_node, world=pr2_world_state_reset
    )

    pr2_world_copy = world_fetcher.fetch_world_from_service(
        rclpy_node,
    )

    dao = to_dao(pr2_world_copy)

    session.add(dao)
    session.commit()


def _is_part_whole_relationship(annotation_type, field_name):
    """
    Return whether ``field_name`` on ``annotation_type`` is marked as a part-whole
    relationship.
    """
    metadata = IsPartWholeRelationship.of_field(
        clazz=annotation_type, field_name=field_name
    )
    return metadata is not None


def test_part_whole_relationship_field_survives_deepcopy():
    """
    A part-whole relationship field keeps its marker and its part when the world is
    copied.
    """
    copy_functions = [copy, deepcopy]
    for copy_function in copy_functions:
        world = World.create_with_root_body("root")
        with world.modify_world():
            door = Door.create_with_new_body_in_world(
                name="door", scale=Scale(0.03, 1, 2), world=world
            )
            handle = Handle.create_with_new_body_in_world(name="handle", world=world)
            door.add(part=handle)

        # The marker is present on the source class before persisting.
        assert _is_part_whole_relationship(Door, "handle")
        assert _is_part_whole_relationship(Door, "entry_way")

        copied_door = copy_function(door)

        # The reconstructed object is a real Door, so its fields still carry the marker.
        assert isinstance(copied_door, Door)
        assert _is_part_whole_relationship(type(copied_door), "handle")
        assert _is_part_whole_relationship(type(copied_door), "entry_way")

        # The marked-field discovery still resolves the same part-whole relationship fields.
        discovered = {
            spec.field.name
            for spec in WrappedClass(clazz=type(copied_door)).fields_with_metadata(
                metadata_type=IsPartWholeRelationship
            )
        }
        assert {"handle", "entry_way"} <= discovered

        # The field values themselves survived the round trip.
        assert isinstance(copied_door.handle, Handle)
        assert isinstance(copied_door.entry_way, EntryWay)


@pytest.fixture
def hsr_world_state_reset(_hsr_world_setup):
    """
    Single-HSRB world fixture that mirrors ``pr2_world_state_reset``.

    ``_hsr_world_setup`` already has HSRB annotations (added by
    ``world_with_urdf_factory``), so we only deepcopy — no second ``from_world`` call —
    and restore the joint-state vector afterwards.
    """
    world = deepcopy(_hsr_world_setup)
    state = world.state._data.copy()
    yield world
    world.state._data[:] = state


def test_hsrb_world(hsr_world_state_reset, session):
    """
    Verify that an HSRB world can be serialised, inserted into a database, queried back,
    and fully reconstructed — including the robot's mobile base.
    """
    from semantic_digital_twin.orm.ormatic_interface import WorldMappingDAO

    dao: WorldMappingDAO = to_dao(hsr_world_state_reset)
    session.add(dao)
    session.commit()

    queried_world = session.scalar(select(WorldMappingDAO))
    reconstructed: World = queried_world.from_dao()

    [hsrb] = reconstructed.get_semantic_annotations_by_type(HSRB)
    assert hsrb.mobile_base is not None
    assert hsrb.mobile_base.torso is not None
    assert hsrb.mobile_base.torso.arm is not None


def test_part_whole_relationship_field_metadata_survives_orm_round_trip(session):
    """
    The part-whole relationship marker is an ``IsPartWholeRelationship`` attached
    directly to the field's ``metadata`` mapping and lives on the dataclass definition,
    not in the persisted row (ORMatic never inspects the field metadata).

    Reconstructing an annotation from its DAO must therefore yield an instance whose
    type still carries the marker, the marked-field discovery must still find it, and
    the field *values* (handle, entry_way) must survive the round trip.
    """
    from semantic_digital_twin.orm.ormatic_interface import WorldMappingDAO

    world = World.create_with_root_body("root")
    with world.modify_world():
        door = Door.create_with_new_body_in_world(
            name="door", scale=Scale(0.03, 1, 2), world=world
        )
        handle = Handle.create_with_new_body_in_world(name="handle", world=world)
        door.add(part=handle)

    # The marker is present on the source class before persisting.
    assert _is_part_whole_relationship(Door, "handle")
    assert _is_part_whole_relationship(Door, "entry_way")

    world_dao: WorldMappingDAO = to_dao(world)
    session.add(world_dao)
    session.commit()

    reconstructed: World = session.scalar(select(WorldMappingDAO)).from_dao()
    [reconstructed_door] = reconstructed.get_semantic_annotations_by_type(Door)

    # The reconstructed object is a real Door, so its fields still carry the marker.
    assert isinstance(reconstructed_door, Door)
    assert _is_part_whole_relationship(type(reconstructed_door), "handle")
    assert _is_part_whole_relationship(type(reconstructed_door), "entry_way")

    # The marked-field discovery still resolves the same part-whole relationship fields.
    discovered = {
        spec.field.name
        for spec in WrappedClass(clazz=type(reconstructed_door)).fields_with_metadata(
            metadata_type=IsPartWholeRelationship
        )
    }
    assert {"handle", "entry_way"} <= discovered

    # The field values themselves survived the round trip.
    assert isinstance(reconstructed_door.handle, Handle)
    assert isinstance(reconstructed_door.entry_way, EntryWay)


def test_door_swings_about_its_hinge_after_orm_round_trip(session):
    """
    A door whose origin sits away from its hinge must still swing about the hinge once
    reconstructed, so the offset of the door from its joint has to be persisted.
    """
    from semantic_digital_twin.orm.ormatic_interface import WorldMappingDAO

    hinge_T_door = HomogeneousTransformationMatrix.from_xyz_rpy(y=0.5)
    world = World.create_with_root_body("root")
    with world.modify_world():
        Door.create_with_new_body_in_world(
            name="door",
            scale=Scale(0.03, 1, 2),
            world=world,
            parent_connection_specification=ScrewAxisConnectionSpecification(
                screw_axis=ScrewAxisConnection.create_revolute_screw_axis(
                    axis=Vector3.Z()
                ),
                connection_T_child=hinge_T_door,
            ),
        )

    session.add(to_dao(world))
    session.commit()
    reconstructed: World = session.scalar(select(WorldMappingDAO)).from_dao()
    [reconstructed_door] = reconstructed.get_semantic_annotations_by_type(Door)

    np.testing.assert_allclose(
        reconstructed_door.movable_joint.connection_T_child_expression.to_np(),
        hinge_T_door.to_np(),
    )


# %% screw axis connection


def test_screw_axis_connection_survives_an_orm_round_trip(
    session, world_with_two_bodies
):
    """
    The screw axis and the child's pose survive storing the world and reading it back.
    """
    from semantic_digital_twin.orm.ormatic_interface import WorldMappingDAO

    world, parent, child = world_with_two_bodies
    with world.modify_world():
        connection = ScrewAxisConnection.create_with_dofs(
            world=world,
            parent=parent,
            child=child,
            screw_axis=ScrewAxisConnection.create_helical_screw_axis(
                axis=Vector3.Y(), pitch=0.25
            ),
        )
        world.add_connection(connection)
    connection.position = 2.5
    parent_T_child = world.compute_forward_kinematics_np(root=parent, tip=child)

    session.add(to_dao(world))
    session.commit()
    reconstructed: World = session.scalar(select(WorldMappingDAO)).from_dao()

    reconstructed_connection = reconstructed.get_connections_by_type(
        ScrewAxisConnection
    )[0]
    assert np.allclose(reconstructed_connection.screw_axis, connection.screw_axis)
    assert np.allclose(
        reconstructed.compute_forward_kinematics_np(
            root=reconstructed_connection.parent, tip=reconstructed_connection.child
        ),
        parent_T_child,
    )


# %% spatial values given as expressions over the world state


def test_a_pose_given_by_the_world_state_is_stored_as_it_stands(
    pr2_world_state_reset: World,
):
    """
    A pose given as an expression of where a body stands is stored as the pose the body
    has when it is stored.
    """
    world = pr2_world_state_reset
    tool_frame = world.get_kinematic_structure_entity_by_name("r_gripper_tool_frame")
    world_T_tool_frame = world.compose_forward_kinematics_expression(
        world.root, tool_frame
    )

    stored = to_dao(world_T_tool_frame).from_dao()

    np.testing.assert_allclose(
        stored.to_np(), tool_frame.global_pose.to_np(), atol=1e-12
    )


# %% what a simulation needs of a world after a round trip


@pytest.fixture
def world_with_box() -> World:
    """
    :return: A world holding one box named ``box`` with a full inertia tensor.
    """
    world = World()
    box = Body(
        name=PrefixedName("box"),
        collision=ShapeCollection([Box(scale=Scale(0.1, 0.2, 0.3))]),
        inertial=Inertial(
            mass=2.0,
            inertia=InertiaTensor.from_values(
                ixx=0.3, iyy=0.4, izz=0.5, ixy=0.01, ixz=0.02, iyz=0.03
            ),
        ),
    )
    with world.modify_world():
        world.add_kinematic_structure_entity(box)
    return world


def load_after_storing(world: World, session: Session) -> World:
    """
    :return: The world as it is read back from the database it was stored in.
    """
    from semantic_digital_twin.orm.ormatic_interface import WorldMappingDAO

    session.add(to_dao(world))
    session.commit()
    return session.scalar(select(WorldMappingDAO)).from_dao()


def test_the_inertia_tensor_of_a_body_survives_a_round_trip(session, world_with_box):
    """
    A body read back from the database has the inertia tensor it was stored with.
    """
    box = world_with_box.get_body_by_name("box")

    loaded = load_after_storing(world_with_box, session)

    loaded_box = loaded.get_body_by_name("box")
    assert np.array_equal(loaded_box.inertial.inertia.data, box.inertial.inertia.data)


def test_a_camera_keeps_the_body_it_is_attached_to_through_a_round_trip(
    session, world_with_box
):
    """
    A camera read back from the database is attached to the body read back with it.
    """
    box = world_with_box.get_body_by_name("box")
    with world_with_box.modify_world():
        box.add_simulator_property(MujocoCamera(name="eye", body=box))

    loaded = load_after_storing(world_with_box, session)

    loaded_box = loaded.get_body_by_name("box")
    [camera] = [
        simulator_property
        for simulator_property in loaded_box.simulator_additional_properties
        if isinstance(simulator_property, MujocoCamera)
    ]
    assert camera.body is loaded_box


def test_an_actuator_keeps_the_degree_of_freedom_it_drives_through_a_round_trip(
    session, world_with_box
):
    """
    An actuator read back from the database drives the degree of freedom it drove when
    it was stored.
    """
    box = world_with_box.get_body_by_name("box")
    arm = Body(name=PrefixedName("arm"))
    actuator = Actuator(name=PrefixedName("servo"))
    with world_with_box.modify_world():
        world_with_box.add_kinematic_structure_entity(arm)
        joint = ScrewAxisConnection.create_with_dofs(
            world=world_with_box,
            parent=box,
            child=arm,
            screw_axis=ScrewAxisConnection.create_revolute_screw_axis(axis=Vector3.Z()),
        )
        world_with_box.add_connection(joint)
        actuator.add_dof(joint.dof)
        world_with_box.add_actuator(actuator)

    loaded = load_after_storing(world_with_box, session)

    [loaded_actuator] = loaded.actuators
    assert [dof.id for dof in loaded_actuator.dofs] == [joint.dof.id]
