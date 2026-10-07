"""
A query names an attribute by its access path on the domain object.

The same path has to reach that attribute in the database and name the variable a
probabilistic model learned for it, so one query can be answered by any of them.
"""

from __future__ import annotations

import numpy as np
import pytest
from random_events.interval import open
from random_events.product_algebra import SimpleEvent

import semantic_digital_twin.orm.ormatic_interface  # type: ignore  # noqa: F401
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import an, entity, probability_of, variable
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.eql_interface import eql_to_sql
from krrood.parametrization.feature_extraction.feature_extractor import FeatureExtractor
from krrood.parametrization.model_registries import DictRegistry
from krrood.parametrization.parameterizer import ConditionParameters
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from semantic_digital_twin.spatial_types.spatial_types import (
    AxisAngle,
    HomogeneousTransformationMatrix,
    Pose,
    RotationMatrix,
    Vector3,
)

# %% rotated instances of every spatial type that stores its rotation as an axis angle

axes = [
    Vector3(0.0, 0.0, 1.0),
    Vector3(0.0, 1.0, 0.0),
    Vector3(1.0, 0.0, 0.0),
    Vector3.unit_vector(1.0, 1.0, 1.0),
]
angles = np.linspace(0.2, 2.8, 12)


def rotations_about_varied_axes(rotation_from_axis_angle) -> list:
    return [
        rotation_from_axis_angle(axis=axes[index % len(axes)], angle=float(angle))
        for index, angle in enumerate(angles)
    ]


spatial_types = pytest.mark.parametrize(
    "spatial_type, rotation_from_axis_angle",
    [
        (
            RotationMatrix,
            lambda axis, angle: RotationMatrix.from_axis_angle(AxisAngle(axis, angle)),
        ),
        (
            HomogeneousTransformationMatrix,
            HomogeneousTransformationMatrix.from_xyz_axis_angle,
        ),
        (Pose, Pose.from_xyz_axis_angle),
    ],
    ids=["RotationMatrix", "HomogeneousTransformationMatrix", "Pose"],
)

access_paths = pytest.mark.parametrize(
    "access_path, threshold",
    [
        (lambda rotation: rotation.axis_angle.angle, 1.5),
        (lambda rotation: rotation.axis_angle.axis.z, 0.5),
    ],
    ids=["axis_angle.angle", "axis_angle.axis.z"],
)


# %% the model learned from domain objects


@spatial_types
@access_paths
def test_a_query_on_a_domain_attribute_reads_the_variable_learned_for_it(
    spatial_type, rotation_from_axis_angle, access_path, threshold
):
    instances = rotations_about_varied_axes(rotation_from_axis_angle)
    columns = (
        FeatureExtractor.from_instances(instances).create_dataframe(instances).columns
    )

    rotation = variable(spatial_type, domain=instances)
    parameters = ConditionParameters(access_path(rotation) < threshold)

    [queried_name] = parameters.variables
    assert queried_name in columns


@spatial_types
@access_paths
def test_probability_of_a_domain_attribute_is_answered_by_the_learned_model(
    spatial_type, rotation_from_axis_angle, access_path, threshold
):
    instances = rotations_about_varied_axes(rotation_from_axis_angle)
    model = JointProbabilityTree().fit(
        FeatureExtractor.from_instances(instances).create_dataframe(instances)
    )
    rotation = variable(spatial_type, domain=instances)
    condition = access_path(rotation) < threshold
    [queried_name] = ConditionParameters(condition).variables
    [learned_variable] = [
        learned for learned in model.variables if learned.name == queried_name
    ]

    result = probability_of(condition).first(
        backend=ProbabilisticBackend(model_registry=DictRegistry({spatial_type: model}))
    )

    expected = model.probability(
        SimpleEvent.from_data(
            {learned_variable: open(-np.inf, threshold)}
        ).as_composite_set()
    )
    assert result == pytest.approx(expected)


# %% the database built from domain objects


@spatial_types
@access_paths
def test_the_database_selects_the_same_objects_as_the_domain_model(
    spatial_type,
    rotation_from_axis_angle,
    access_path,
    threshold,
    in_memory_session_maker,
):
    instances = rotations_about_varied_axes(rotation_from_axis_angle)
    session = in_memory_session_maker()
    session.add_all([to_dao(instance) for instance in instances])
    session.commit()

    in_memory = variable(spatial_type, domain=instances)
    selected_in_memory = an(
        entity(in_memory).where(access_path(in_memory) < threshold)
    ).evaluate()
    stored = variable(spatial_type, domain=[])
    selected_in_database = eql_to_sql(
        an(entity(stored).where(access_path(stored) < threshold)), session
    ).evaluate()

    expected = sorted(float(access_path(selected)) for selected in selected_in_memory)
    actual = sorted(
        float(access_path(selected.from_dao())) for selected in selected_in_database
    )
    assert expected
    assert actual == pytest.approx(expected)
