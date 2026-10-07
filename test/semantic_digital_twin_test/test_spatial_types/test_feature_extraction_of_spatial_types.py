import pytest

import semantic_digital_twin.orm.ormatic_interface  # type: ignore  # noqa: F401
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.parametrization.feature_extraction.feature_extractor import FeatureExtractor
from krrood.symbolic_math.symbolic_math import FloatVariable
from semantic_digital_twin.spatial_types.spatial_types import (
    AxisAngle,
    HomogeneousTransformationMatrix,
    Point3,
    Pose,
    Quaternion,
    RotationMatrix,
    Vector3,
)
from semantic_digital_twin.world_description.inertial_properties import Inertial


def feature_names(extractor: FeatureExtractor) -> list[str]:
    return [
        feature.get_clean_name_from_mapped_variable() for feature in extractor.features
    ]


def test_features_of_a_pose_are_read_from_the_pose_itself():
    pose = Pose.from_xyz_rpy(x=1.0, y=2.0, z=3.0, yaw=0.5)

    extractor = FeatureExtractor.from_instances([pose])

    assert feature_names(extractor) == [
        "position.x",
        "position.y",
        "position.z",
        "axis_angle.angle",
        "axis_angle.axis.x",
        "axis_angle.axis.y",
        "axis_angle.axis.z",
    ]
    values = extractor.apply_mapping(pose)
    assert all(type(value) is float for value in values)
    assert values[:3] == [1.0, 2.0, 3.0]
    assert values[3:] == pytest.approx(
        [float(pose.axis_angle.angle), *pose.axis_angle.axis.to_np()[:3]]
    )


@pytest.mark.parametrize(
    "instance",
    [
        Point3(1.0, 2.0, 3.0),
        Vector3(1.0, 2.0, 3.0),
        Quaternion(0.0, 0.0, 0.0, 1.0),
        AxisAngle(axis=Vector3(0.0, 1.0, 0.0), angle=0.4),
        RotationMatrix.from_rpy(roll=0.1, pitch=0.2, yaw=0.3),
        HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0, y=2.0, z=3.0, yaw=0.3),
        Pose.from_xyz_rpy(x=1.0, y=2.0, z=3.0, yaw=0.5),
        Inertial(mass=2.0),
    ],
    ids=lambda instance: type(instance).__name__,
)
def test_features_of_a_domain_object_match_those_of_its_data_access_object(instance):
    """
    A data access object stores a number where the domain object may hold a constant
    symbolic expression, and both have to yield the same features.
    """
    extractor = FeatureExtractor.from_instances([instance])
    assert extractor.features

    data_access_object = to_dao(instance)
    expected = [
        feature.apply_mapping_on_external_root(data_access_object)
        for feature in extractor.features
    ]
    assert extractor.apply_mapping(instance) == pytest.approx(expected)


def test_a_symbolic_expression_with_free_variables_is_not_a_feature():
    point = Point3(FloatVariable(name="x"), 2.0, 3.0)

    extractor = FeatureExtractor.from_instances([point])

    assert feature_names(extractor) == ["y", "z"]
