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
    HomogeneousTransformationMatrix,
    Pose2D,
    RotationMatrix,
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


def test_rotation_matrix_exposes_roll_pitch_and_yaw():
    rotation_matrix = RotationMatrix.from_rpy(roll=0.1, pitch=0.2, yaw=0.3)
    roll_pitch_yaw = rotation_matrix.roll_pitch_yaw
    assert dataclasses.astuple(roll_pitch_yaw) == pytest.approx((0.1, 0.2, 0.3))
    assert all(type(angle) is float for angle in dataclasses.astuple(roll_pitch_yaw))


def test_homogeneous_transformation_matrix_exposes_position_roll_pitch_and_yaw():
    transformation = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=1.0, y=2.0, z=3.0, roll=0.1, pitch=0.2, yaw=0.3
    )
    assert np.allclose(transformation.position.to_np()[:3], [1.0, 2.0, 3.0])
    assert dataclasses.astuple(transformation.roll_pitch_yaw) == pytest.approx(
        (0.1, 0.2, 0.3)
    )


@pytest.mark.parametrize(
    "roll, pitch, yaw",
    [(0.1, 0.2, 0.3), (-2.0, 1.0, 3.0), (0.4, np.pi / 2, 0.0), (0.0, 0.0, 0.0)],
)
def test_rotation_survives_being_stored_as_roll_pitch_and_yaw(roll, pitch, yaw):
    transformation = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=1.0, y=2.0, z=3.0, roll=roll, pitch=pitch, yaw=yaw
    )
    restored = HomogeneousTransformationMatrixMapping.from_domain_object(
        transformation
    ).to_domain_object()
    assert np.allclose(restored.to_np(), transformation.to_np())

    rotation_matrix = transformation.to_rotation_matrix()
    restored_rotation = RotationMatrixMapping.from_domain_object(
        rotation_matrix
    ).to_domain_object()
    assert np.allclose(restored_rotation.to_np(), rotation_matrix.to_np())


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
