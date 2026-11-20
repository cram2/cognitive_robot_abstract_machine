"""
Tests of the alternative mappings in :mod:`semantic_digital_twin.orm.model`.
"""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest

from krrood.ormatic.data_access_objects.symbol_graph_mappings import (
    SymbolGraphMapping,
    WrappedInstanceMapping,
)
from krrood.symbol_graph.symbol_graph import SymbolGraph
from semantic_digital_twin.orm.model import (
    AxisAngleMapping,
    HomogeneousTransformationMatrix3DimensionalMapping,
    Point2DimensionalMapping,
    Point3DimensionalMapping,
    Pose2DimensionalMapping,
    Pose3DimensionalMapping,
    QuaternionMapping,
    RotationMatrixMapping,
    DisplacementMapping,
    ScrewAxisMapping,
    TwistMapping,
    Vector3DimensionalMapping,
    WorldMapping,
    WorldStateMapping,
)
from krrood_spatial_math.spatial_types import (
    AxisAngle,
    HomogeneousTransformationMatrix3Dimensional,
    Pose3Dimensional,
    RotationMatrix,
    Displacement,
    ScrewAxis,
    Twist,
    Vector3Dimensional,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


@pytest.mark.parametrize(
    "mapping",
    [
        WorldMapping,
        WorldStateMapping,
        Vector3DimensionalMapping,
        Point3DimensionalMapping,
        QuaternionMapping,
        AxisAngleMapping,
        RotationMatrixMapping,
        HomogeneousTransformationMatrix3DimensionalMapping,
        Pose3DimensionalMapping,
        Point2DimensionalMapping,
        Pose2DimensionalMapping,
        DisplacementMapping,
        TwistMapping,
        ScrewAxisMapping,
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
            HomogeneousTransformationMatrix3DimensionalMapping,
            HomogeneousTransformationMatrix3Dimensional.from_xyz_axis_angle,
        ),
        (Pose3DimensionalMapping, Pose3Dimensional.from_xyz_axis_angle),
    ],
    ids=[
        "RotationMatrix",
        "HomogeneousTransformationMatrix3Dimensional",
        "Pose3Dimensional",
    ],
)
def test_rotation_survives_being_stored_as_axis_angle(
    mapping, rotation_from_axis_angle, axis, angle
):
    rotation = rotation_from_axis_angle(axis=Vector3Dimensional(*axis), angle=angle)

    stored = mapping.from_domain_object(obj=rotation)
    restored = stored.to_domain_object()

    assert isinstance(stored.axis_angle, AxisAngle)
    assert np.allclose(restored.to_np(), rotation.to_np())


@pytest.mark.parametrize(
    "mapping, vector_pair_type",
    [
        (DisplacementMapping, Displacement),
        (TwistMapping, Twist),
        (ScrewAxisMapping, ScrewAxis),
    ],
    ids=["Displacement", "Twist", "ScrewAxis"],
)
def test_a_vector_pair_survives_being_stored(mapping, vector_pair_type):
    """
    Storing a displacement, twist or screw axis through its mapping keeps its values and
    reference frame.
    """
    world = World()
    with world.modify_world():
        root = Body(name="root")
        world.add_kinematic_structure_entity(root)
    vector_pair = vector_pair_type(
        **{
            vector_pair_type._rotational_part_name: Vector3Dimensional(0.0, 0.4, 0.0),
            vector_pair_type._translational_part_name: Vector3Dimensional(
                0.1, 0.0, -0.2
            ),
        },
        reference_frame=root,
    )

    restored = mapping.from_domain_object(obj=vector_pair).to_domain_object()

    assert type(restored) is vector_pair_type
    assert restored.reference_frame is root
    assert np.allclose(restored.to_np(), vector_pair.to_np())


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
