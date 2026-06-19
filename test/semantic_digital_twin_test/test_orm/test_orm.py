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
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Handle,
    Slider,
)
from semantic_digital_twin.semantic_annotations.mixins import (
    _wrapped_part_whole_relationship_fields,
    PartWholeRelationshipField,
)
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


def _field(annotation_type, field_name):
    """Return the dataclass ``Field`` object for ``field_name`` on ``annotation_type``."""
    return {f.name: f for f in dataclasses.fields(annotation_type)}[field_name]


def test_part_whole_relationship_field_survives_deepcopy():
    copy_functions = [copy, deepcopy]
    for copy_function in copy_functions:
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)
        with world.modify_world():
            drawer = Drawer.create_with_new_body_in_world(
                name=PrefixedName("drawer"), scale=Scale(0.2, 0.3, 0.2), world=world
            )
            handle = Handle.create_with_new_body_in_world(
                name=PrefixedName("handle"), world=world
            )
            slider = Slider.create_with_new_body_in_world(
                name=PrefixedName("slider"), world=world, active_axis=Vector3.X()
            )
            drawer.add(handle)
            drawer.add(slider)

        # The marker is present on the source class before persisting.
        assert isinstance(_field(Drawer, "handle"), PartWholeRelationshipField)
        assert isinstance(
            _field(Drawer, "mechanical_joint"), PartWholeRelationshipField
        )

        copied_drawer = copy_function(drawer)

        # The reconstructed object is a real Drawer, so its fields still carry the marker.
        assert isinstance(copied_drawer, Drawer)
        assert isinstance(
            _field(type(copied_drawer), "handle"), PartWholeRelationshipField
        )
        assert isinstance(
            _field(type(copied_drawer), "mechanical_joint"),
            PartWholeRelationshipField,
        )

        # The marked-field discovery still resolves the same part-whole relationship fields.
        discovered = {
            spec.field.name
            for spec in _wrapped_part_whole_relationship_fields(type(copied_drawer))
        }
        assert {"handle", "mechanical_joint"} <= discovered

        # The field values themselves survived the round trip.
        assert isinstance(copied_drawer.handle, Handle)
        assert isinstance(copied_drawer.mechanical_joint, Slider)


def test_part_whole_relationship_field_metadata_survives_orm_round_trip(session):
    """
    The part-whole relationship marker is the field's ``PartWholeRelationshipField`` type and lives
    on the dataclass definition, not in the persisted row (ORMatic never inspects the field type).
    Reconstructing an annotation from its DAO must therefore yield an instance whose type still
    carries the marker, the marked-field discovery must still find it, and the field *values*
    (handle, mechanical_joint) must survive the round trip.
    """
    world = World()
    root = Body(name=PrefixedName("root"))
    with world.modify_world():
        world.add_body(root)
    with world.modify_world():
        drawer = Drawer.create_with_new_body_in_world(
            name=PrefixedName("drawer"), scale=Scale(0.2, 0.3, 0.2), world=world
        )
        handle = Handle.create_with_new_body_in_world(
            name=PrefixedName("handle"), world=world
        )
        slider = Slider.create_with_new_body_in_world(
            name=PrefixedName("slider"), world=world, active_axis=Vector3.X()
        )
        drawer.add(handle)
        drawer.add(slider)

    # The marker is present on the source class before persisting.
    assert isinstance(_field(Drawer, "handle"), PartWholeRelationshipField)
    assert isinstance(_field(Drawer, "mechanical_joint"), PartWholeRelationshipField)

    world_dao: WorldMappingDAO = to_dao(world)
    session.add(world_dao)
    session.commit()

    reconstructed: World = session.scalar(select(WorldMappingDAO)).from_dao()
    [reconstructed_drawer] = reconstructed.get_semantic_annotations_by_type(Drawer)

    # The reconstructed object is a real Drawer, so its fields still carry the marker.
    assert isinstance(reconstructed_drawer, Drawer)
    assert isinstance(
        _field(type(reconstructed_drawer), "handle"), PartWholeRelationshipField
    )
    assert isinstance(
        _field(type(reconstructed_drawer), "mechanical_joint"),
        PartWholeRelationshipField,
    )

    # The marked-field discovery still resolves the same part-whole relationship fields.
    discovered = {
        spec.field.name
        for spec in _wrapped_part_whole_relationship_fields(type(reconstructed_drawer))
    }
    assert {"handle", "mechanical_joint"} <= discovered

    # The field values themselves survived the round trip.
    assert isinstance(reconstructed_drawer.handle, Handle)
    assert isinstance(reconstructed_drawer.mechanical_joint, Slider)
