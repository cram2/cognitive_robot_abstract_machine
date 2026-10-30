import numpy as np
import pytest

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import (
    AxisAngle,
    HomogeneousTransformationMatrix,
    Pose,
    Quaternion,
    RotationMatrix,
    Vector3,
)
from semantic_digital_twin.world_description.world_entity import Body
from .reference_implementations import (
    quaternion_from_axis_angle,
    rotation_matrix_from_axis_angle,
)

# %% rotations given as a unit axis and an angle in (0, pi)

rotations = [
    ((0.0, 0.0, 1.0), 0.5),
    ((1.0, 0.0, 0.0), 2.0),
    ((0.0, 1.0, 0.0), np.pi / 2),
    (tuple(np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)), 3.0),
]


def axis_and_angle_of(axis_angle: AxisAngle) -> tuple[np.ndarray, float]:
    return axis_angle.axis.to_np()[:3], float(axis_angle.angle)


# %% reading the axis angle off a rotation


@pytest.mark.parametrize("axis, angle", rotations)
@pytest.mark.parametrize(
    "rotation_from_axis_angle",
    [
        lambda axis, angle: RotationMatrix.from_axis_angle(AxisAngle(axis, angle)),
        lambda axis, angle: Quaternion.from_axis_angle(AxisAngle(axis, angle)),
        HomogeneousTransformationMatrix.from_xyz_axis_angle,
        Pose.from_xyz_axis_angle,
    ],
    ids=["RotationMatrix", "Quaternion", "HomogeneousTransformationMatrix", "Pose"],
)
def test_axis_angle_of_a_rotation_is_the_one_it_was_built_from(
    rotation_from_axis_angle, axis, angle
):
    rotation = rotation_from_axis_angle(axis=Vector3(*axis), angle=angle)

    actual_axis, actual_angle = axis_and_angle_of(rotation.axis_angle)

    assert actual_axis == pytest.approx(axis)
    assert actual_angle == pytest.approx(angle)


def test_axis_angle_of_the_identity_rotation_has_angle_zero():
    assert float(RotationMatrix().axis_angle.angle) == 0.0


def test_axis_angle_keeps_the_reference_frame_of_the_rotation():
    reference_frame = Body(name=PrefixedName("reference"))

    axis_angle = RotationMatrix(reference_frame=reference_frame).axis_angle

    assert axis_angle.reference_frame is reference_frame
    assert axis_angle.axis.reference_frame is reference_frame


# %% converting an axis angle into the other rotation representations


@pytest.mark.parametrize("axis, angle", rotations)
def test_axis_angle_converts_into_the_rotation_it_describes(axis, angle):
    axis_angle = AxisAngle(axis=Vector3(*axis), angle=angle)

    assert np.allclose(
        axis_angle.rotation_matrix.to_np(),
        rotation_matrix_from_axis_angle(np.array(axis), angle),
    )
    assert np.allclose(
        axis_angle.quaternion.to_np(),
        quaternion_from_axis_angle(axis, angle),
    )


def test_axis_angle_without_arguments_is_the_identity_rotation():
    assert np.allclose(AxisAngle().rotation_matrix.to_np(), np.eye(4))


@pytest.mark.parametrize("axis, angle", rotations)
def test_axis_angle_survives_json(axis, angle):
    axis_angle = AxisAngle(axis=Vector3(*axis), angle=angle)

    restored = AxisAngle.from_json(axis_angle.to_json())

    assert np.allclose(restored.to_np(), axis_angle.to_np())
