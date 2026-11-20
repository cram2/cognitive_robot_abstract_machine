"""
Which groups of rigid motions act on which spatial types, and how the symbolic values
and changes are split.
"""

from __future__ import annotations

import numpy as np
import pytest

from krrood.symbolic_math.exceptions import UnsupportedOperationError
from krrood_spatial_math.exceptions import ReferenceFrameMismatchError

from krrood_spatial_math.coordinate_system import (
    CoordinateRepresentation,
    RigidCoordinateChange,
)
from krrood_spatial_math.spatial_types import (
    AxisAngle,
    Point3Dimensional,
    HomogeneousCoordinates,
    HomogeneousTransformationMatrix3Dimensional,
    Orientation,
    Pose3Dimensional,
    Quaternion,
    RotationMatrix,
    SpecialEuclideanGroupAction,
    SymbolicSpatialChange,
    SymbolicSpatialValue,
    Vector3Dimensional,
)
from krrood_spatial_math.spatial_types_2_dimensional import (
    HomogeneousTransformationMatrix2Dimensional,
    Point2Dimensional,
)

# %% rigid motions of either dimension


def test_a_planar_value_is_moved_by_rigid_motions_of_its_plane():
    """
    A planar value is moved by the rigid motions of its plane, SE(2).
    """
    assert issubclass(Point2Dimensional, SpecialEuclideanGroupAction)


def test_a_transformation_refuses_a_value_of_another_dimension():
    """
    A 3D transformation does not act on a planar point, nor a planar one on a 3D point.
    """
    with pytest.raises(UnsupportedOperationError):
        HomogeneousTransformationMatrix3Dimensional() @ Point2Dimensional(x=1, y=0)
    with pytest.raises(UnsupportedOperationError):
        HomogeneousTransformationMatrix2Dimensional() @ Point3Dimensional(x=1)


@pytest.mark.parametrize(
    "spatial_type",
    [Point3Dimensional, Vector3Dimensional, Pose3Dimensional],
)
def test_values_stored_as_homogeneous_coordinates_are_moved_by_rigid_motions(
    spatial_type,
):
    """
    Values stored as homogeneous coordinates are moved by multiplying them.
    """
    assert issubclass(spatial_type, HomogeneousCoordinates)
    assert issubclass(spatial_type, SpecialEuclideanGroupAction)


@pytest.mark.parametrize("orientation_type", [Quaternion, AxisAngle])
def test_an_orientation_is_moved_by_rigid_motions(orientation_type):
    assert issubclass(orientation_type, Orientation)
    assert issubclass(orientation_type, SpecialEuclideanGroupAction)


def test_a_transformation_turns_an_orientation_by_its_rotation_only():
    rotation = RotationMatrix.from_rpy(roll=0.0, pitch=0.0, yaw=np.pi / 2)
    shifted_rotation = HomogeneousTransformationMatrix3Dimensional.from_xyz_rpy(
        x=1.0, y=2.0, z=3.0, yaw=np.pi / 2
    )
    orientation = Quaternion.from_rpy(roll=0.3, pitch=0.0, yaw=0.0)

    np.testing.assert_array_almost_equal(
        (shifted_rotation @ orientation).to_np(), (rotation @ orientation).to_np()
    )


# %% rotations as the transformations without translation, SO(3) within SE(3)


@pytest.fixture
def quarter_turn() -> RotationMatrix:
    """
    A quarter turn about the z-axis.
    """
    return RotationMatrix.from_rpy(roll=0.0, pitch=0.0, yaw=np.pi / 2)


@pytest.fixture
def shifted_quarter_turn() -> HomogeneousTransformationMatrix3Dimensional:
    """
    A quarter turn about the z-axis followed by a shift.
    """
    return HomogeneousTransformationMatrix3Dimensional.from_xyz_rpy(
        x=1.0, y=2.0, z=3.0, yaw=np.pi / 2
    )


def test_a_rotation_is_a_transformation():
    """
    SO(3) is the subgroup of SE(3) without translation, so a rotation is a transformation.
    """
    assert issubclass(RotationMatrix, HomogeneousTransformationMatrix3Dimensional)


def test_a_rotation_has_no_translation_and_is_its_own_rotation_part(quarter_turn):
    """
    A rotation shifts nothing, and its rotation part is itself.
    """
    np.testing.assert_array_equal(quarter_turn.position.to_np()[:3], np.zeros(3))
    assert quarter_turn.rotation_matrix is quarter_turn


def test_a_rotation_composed_with_a_rotation_is_a_rotation(quarter_turn):
    """
    Rotations are closed under composition.
    """
    assert type(quarter_turn @ quarter_turn) is RotationMatrix


def test_the_inverse_of_a_rotation_is_a_rotation_between_swapped_frames(
    table_frame, camera_frame
):
    """
    Rotations are closed under inversion, which swaps the frames they turn between.
    """
    table_R_camera = RotationMatrix.from_rpy(yaw=0.5, reference_frame=table_frame)
    table_R_camera.child_frame = camera_frame

    camera_R_table = table_R_camera.inverse()

    assert type(camera_R_table) is RotationMatrix
    assert (camera_R_table.reference_frame, camera_R_table.child_frame) == (
        camera_frame,
        table_frame,
    )


def test_a_transformation_composed_with_a_rotation_keeps_its_translation(
    quarter_turn, shifted_quarter_turn
):
    """
    Composing with a rotation leaves SO(3), so the translation is kept, never dropped.
    """
    composed = shifted_quarter_turn @ quarter_turn

    assert type(composed) is HomogeneousTransformationMatrix3Dimensional
    np.testing.assert_array_almost_equal(
        composed.position.to_np(), shifted_quarter_turn.position.to_np()
    )


def test_a_rotation_hashes_like_the_transformation_without_translation(quarter_turn):
    """
    A rotation and the transformation without translation are one group element.
    """
    without_translation = HomogeneousTransformationMatrix3Dimensional.from_xyz_rpy(
        yaw=np.pi / 2
    )

    assert hash(quarter_turn) == hash(without_translation)


def test_a_rotation_refuses_a_value_given_in_another_frame_than_its_child_frame(
    table_frame, camera_frame, gripper_frame
):
    """
    A rotation checks the frame of what it turns, as every transformation does.
    """
    table_R_camera = RotationMatrix.from_rpy(yaw=0.5, reference_frame=table_frame)
    table_R_camera.child_frame = camera_frame

    with pytest.raises(ReferenceFrameMismatchError):
        table_R_camera @ Point3Dimensional(1, 0, 0, reference_frame=gripper_frame)


def test_a_transformation_has_no_settable_translation(shifted_quarter_turn):
    """
    A transformation is a value: its translation cannot be changed in place.
    """
    with pytest.raises(AttributeError):
        shifted_quarter_turn.x = 0.0


def test_normalizing_a_rotation_leaves_it_unchanged():
    """
    Normalizing gives a new rotation and leaves the normalized one as it was.
    """
    scaled = RotationMatrix(data=2 * np.eye(4))

    normalized = scaled.normalize()

    np.testing.assert_array_almost_equal(normalized.to_np(), np.eye(4))
    np.testing.assert_array_almost_equal(scaled.to_np()[:3, :3], 2 * np.eye(3))


# %% symbolic values and symbolic changes


@pytest.mark.parametrize(
    "value_type",
    [
        Point3Dimensional,
        Vector3Dimensional,
        Quaternion,
        AxisAngle,
        Pose3Dimensional,
    ],
)
def test_a_symbolic_value_is_a_coordinate_representation(value_type):
    """
    Every symbolic 3D value is a value given in a coordinate system.
    """
    assert issubclass(value_type, SymbolicSpatialValue)
    assert issubclass(value_type, CoordinateRepresentation)


@pytest.mark.parametrize(
    "change_type",
    [
        HomogeneousTransformationMatrix3Dimensional,
        RotationMatrix,
        HomogeneousTransformationMatrix2Dimensional,
    ],
)
def test_a_symbolic_change_is_a_rigid_change_and_never_a_value(change_type):
    """
    A transformation of either dimension changes coordinates; it is no value itself.
    """
    assert issubclass(change_type, SymbolicSpatialChange)
    assert issubclass(change_type, RigidCoordinateChange)
    assert not issubclass(change_type, SymbolicSpatialValue)
