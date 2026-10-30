import dataclasses
import inspect

import numpy as np
import pytest

from krrood.entity_query_language.orm.model import (
    SymbolGraphMapping,
    WrappedInstanceMapping,
)
from krrood.symbol_graph.symbol_graph import SymbolGraph
from semantic_digital_twin.orm.model import (
    AxisAngleMapping,
    HomogeneousTransformationMatrixMapping,
    Point2Mapping,
    Point3Mapping,
    Pose2DMapping,
    PoseMapping,
    QuaternionMapping,
    RotationMatrixMapping,
    Vector3Mapping,
    WorldMapping,
    WorldStateMapping,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    AxisAngle,
    HomogeneousTransformationMatrix,
    Pose,
    Pose2D,
    RotationMatrix,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


@pytest.mark.parametrize(
    "mapping",
    [
        WorldMapping,
        WorldStateMapping,
        Vector3Mapping,
        Point3Mapping,
        QuaternionMapping,
        AxisAngleMapping,
        RotationMatrixMapping,
        HomogeneousTransformationMatrixMapping,
        PoseMapping,
        Point2Mapping,
        Pose2DMapping,
        SymbolGraphMapping,
        WrappedInstanceMapping,
    ],
)
def test_every_field_of_an_alternative_mapping_is_reachable_on_its_domain_class(
    mapping,
):
    """
    An access path written against the mapping has to lead somewhere on the domain
    object as well, so the two can be used interchangeably.
    """
    domain_class = mapping.original_class()
    domain_field_names = {field.name for field in dataclasses.fields(domain_class)}
    missing = [
        field.name
        for field in dataclasses.fields(mapping)
        if field.name not in domain_field_names
        and inspect.getattr_static(domain_class, field.name, None) is None
    ]
    assert missing == []


@pytest.mark.parametrize(
    "axis, angle",
    [
        ((0.0, 0.0, 1.0), 0.3),
        ((1.0, 0.0, 0.0), 2.0),
        ((0.0, 1.0, 0.0), np.pi / 2),
        ((0.0, 0.0, 1.0), 0.0),
        ((0.0, 1.0, 0.0), np.pi),
    ],
)
@pytest.mark.parametrize(
    "mapping, rotation_from_axis_angle",
    [
        (
            RotationMatrixMapping,
            lambda axis, angle: RotationMatrix.from_axis_angle(AxisAngle(axis, angle)),
        ),
        (
            HomogeneousTransformationMatrixMapping,
            HomogeneousTransformationMatrix.from_xyz_axis_angle,
        ),
        (PoseMapping, Pose.from_xyz_axis_angle),
    ],
    ids=["RotationMatrix", "HomogeneousTransformationMatrix", "Pose"],
)
def test_rotation_survives_being_stored_as_axis_angle(
    mapping, rotation_from_axis_angle, axis, angle
):
    rotation = rotation_from_axis_angle(axis=Vector3(*axis), angle=angle)

    stored = mapping.from_domain_object(rotation)
    restored = stored.to_domain_object()

    assert isinstance(stored.axis_angle, AxisAngle)
    assert np.allclose(restored.to_np(), rotation.to_np())


def test_world_state_data_and_ids_follow_the_stored_state():
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name="root"))
    assert world.state.ids == world.state._ids
    assert world.state.data == world.state._data.ravel().tolist()


def test_world_modification_history_lists_the_applied_blocks():
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name="root"))
    assert world.modification_history is world._model_manager.model_modification_blocks
    assert len(world.modification_history) == 1


def test_symbol_graph_exposes_instances_and_predicate_relations():
    symbol_graph = SymbolGraph()
    assert symbol_graph.instances == symbol_graph.wrapped_instances
    assert symbol_graph.predicate_relations == list(symbol_graph.relations())
