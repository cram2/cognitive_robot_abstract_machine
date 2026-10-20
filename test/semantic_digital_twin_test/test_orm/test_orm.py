import dataclasses
import os
import time
from copy import deepcopy, copy

import numpy as np
from krrood.ormatic.utils import create_engine
from sqlalchemy import select
from sqlalchemy.orm import Session

from semantic_digital_twin.adapters.ros.world_fetcher import (
    FetchWorldServer,
    fetch_world_from_service,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.orm.utils import semantic_digital_twin_sessionmaker
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.pr2 import PR2, PR2RightArm
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import RevoluteConnection
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Box, Scale, Color
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.specifications.connections import (
    RevoluteConnectionSpecification,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Door,
    EntryWay,
    Handle,
)
from semantic_digital_twin.semantic_annotations.part_whole import (
    IsPartWholeRelationship,
)
from krrood.class_diagrams.class_diagram import WrappedClass
from semantic_digital_twin.orm.ormatic_interface import *
from krrood.ormatic.data_access_objects.helper import to_dao


import pytest


@pytest.fixture
def engine():
    return create_engine("sqlite:///:memory:")


@pytest.fixture
def session(engine):
    session = Session(engine)
    Base.metadata.create_all(bind=session.bind)
    yield session
    Base.metadata.drop_all(session.bind)
    session.close()


def test_table_world(session, table_world):
    revolute_connection = table_world.get_connections_by_type(RevoluteConnection)[0]
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
    reconstructed_connection = reconstructed.get_connections_by_type(
        RevoluteConnection
    )[0]
    assert reconstructed_connection.position == revolute_connection.position
    assert reconstructed_connection.velocity == revolute_connection.velocity
    assert reconstructed_connection.acceleration == revolute_connection.acceleration
    assert reconstructed_connection.jerk == revolute_connection.jerk


def test_insert(session):
    origin = HomogeneousTransformationMatrix.from_xyz_rpy(1, 2, 3, 1, 2, 3)
    scale = Scale(1.0, 1.0, 1.0)
    color = Color(0.0, 1.0, 1.0)
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
    dao: WorldMappingDAO = to_dao(pr2_world_state_reset)
    session.add(dao)
    session.commit()

    to_dao(pr2_world_state_reset).from_dao()

    queried_world = session.scalar(select(WorldMappingDAO))
    reconstructed: World = queried_world.from_dao()

    # confirm the modification history
    deepcopy(reconstructed)

    q = select(RevoluteConnectionDAO)
    r = session.scalars(q).all()
    assert len(r) > 0


def test_pr2_semantic_annotation_and_safe_to_db(
    rclpy_node, pr2_world_state_reset, session
):
    fetcher = FetchWorldServer(node=rclpy_node, world=pr2_world_state_reset)

    pr2_world_copy = fetch_world_from_service(
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
    metadata = IsPartWholeRelationship.of_field(annotation_type, field_name)
    return metadata is not None


def test_part_whole_relationship_field_survives_deepcopy():
    copy_functions = [copy, deepcopy]
    for copy_function in copy_functions:
        world = World.create_with_root_body("root")
        with world.modify_world():
            door = Door.create_with_new_body_in_world(
                name="door", scale=Scale(0.03, 1, 2), world=world
            )
            handle = Handle.create_with_new_body_in_world(name="handle", world=world)
            door.add(handle)

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
            for spec in WrappedClass(type(copied_door)).fields_with_metadata(
                IsPartWholeRelationship
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
    world = World.create_with_root_body("root")
    with world.modify_world():
        door = Door.create_with_new_body_in_world(
            name="door", scale=Scale(0.03, 1, 2), world=world
        )
        handle = Handle.create_with_new_body_in_world(name="handle", world=world)
        door.add(handle)

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
        for spec in WrappedClass(type(reconstructed_door)).fields_with_metadata(
            IsPartWholeRelationship
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
    hinge_T_door = HomogeneousTransformationMatrix.from_xyz_rpy(y=0.5)
    world = World.create_with_root_body("root")
    with world.modify_world():
        Door.create_with_new_body_in_world(
            name="door",
            scale=Scale(0.03, 1, 2),
            world=world,
            parent_connection_specification=RevoluteConnectionSpecification(
                axis=Vector3.Z(), connection_T_child=hinge_T_door
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
