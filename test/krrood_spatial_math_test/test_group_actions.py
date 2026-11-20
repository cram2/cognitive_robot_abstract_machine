"""
How a change of frame moves each kind of value, and how the symbolic values and changes
are split.
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
from krrood_spatial_math.motion_vector_pair import MotionVectorPair
from krrood_spatial_math.point import Point
from krrood_spatial_math.pose import Orientation, Pose
from krrood_spatial_math.spatial_types import (
    AxisAngle,
    Displacement,
    Point3Dimensional,
    HomogeneousTransformationMatrix3Dimensional,
    Pose3Dimensional,
    Quaternion,
    RotationMatrix,
    ScrewAxis,
    SymbolicSpatialChange,
    SymbolicSpatialValue,
    Twist,
    Vector3Dimensional,
)
from krrood_spatial_math.spatial_types_2_dimensional import (
    HomogeneousTransformationMatrix2Dimensional,
    Point2Dimensional,
    Pose2Dimensional,
    Vector2Dimensional,
)
from krrood_spatial_math.vector import Vector

from .dataset.spaces import (
    AxisScalingChange,
    NamedScalablePlaneFrame,
    ScalablePlanePoint,
    ScalablePlaneVector,
)
from .reference_implementations import adjoint_matrix_from_homogeneous_matrix

# %% how a change of frame moves each kind of value


@pytest.fixture
def sheet_to_poster() -> AxisScalingChange:
    """
    An affine change of a scalable plane that doubles x, triples y and shifts the origin
    to (5, 7).
    """
    return AxisScalingChange(
        source_coordinate_system=NamedScalablePlaneFrame(name="sheet"),
        target_coordinate_system=NamedScalablePlaneFrame(name="poster"),
        x_scale=2.0,
        y_scale=3.0,
        x_shift=5.0,
        y_shift=7.0,
    )


def test_an_affine_change_moves_a_point_of_a_new_space_by_its_shift(sheet_to_poster):
    """
    A point of a space that declares nothing about changes of frame is scaled and
    shifted.
    """
    point = ScalablePlanePoint(
        x=1.0, y=1.0, coordinate_system=sheet_to_poster.source_coordinate_system
    )

    assert point.apply_transformation(transformation=sheet_to_poster) == (
        ScalablePlanePoint(
            x=7.0, y=10.0, coordinate_system=sheet_to_poster.target_coordinate_system
        )
    )


def test_an_affine_change_never_shifts_a_vector_of_a_new_space(sheet_to_poster):
    """
    A vector has no position, so the same change only scales it.
    """
    vector = ScalablePlaneVector(
        x=1.0, y=1.0, coordinate_system=sheet_to_poster.source_coordinate_system
    )

    assert vector.apply_transformation(transformation=sheet_to_poster) == (
        ScalablePlaneVector(
            x=2.0, y=3.0, coordinate_system=sheet_to_poster.target_coordinate_system
        )
    )


@pytest.mark.parametrize(
    "spatial_type, kind_of_value",
    [
        (Vector3Dimensional, Vector),
        (Vector2Dimensional, Vector),
        (Point3Dimensional, Point),
        (Point2Dimensional, Point),
        (Pose3Dimensional, Pose),
        (Pose2Dimensional, Pose),
        (Quaternion, Orientation),
        (AxisAngle, Orientation),
        (Twist, MotionVectorPair),
        (ScrewAxis, MotionVectorPair),
        (Displacement, MotionVectorPair),
    ],
)
def test_a_symbolic_value_is_moved_as_its_kind_of_value_says(
    spatial_type, kind_of_value
):
    """
    How a change of frame moves a symbolic value is decided by its kind of value, not by
    the symbolic class.
    """
    assert issubclass(spatial_type, kind_of_value)
    assert spatial_type.apply_transformation is kind_of_value.apply_transformation


@pytest.mark.parametrize(
    "point_type, vector_type",
    [(Point3Dimensional, Vector3Dimensional), (Point2Dimensional, Vector2Dimensional)],
)
def test_a_point_reads_its_change_type_from_its_vector_type(point_type, vector_type):
    """
    A point binds only its vector type, which already fixes the change type.
    """
    assert point_type.get_change_type() is vector_type.get_change_type()


def test_a_change_of_frame_moves_a_twist_by_its_adjoint_action():
    """
    Both velocities turn, and the linear one gains the angular one crossed with the
    shift between the origins.
    """
    transformation = HomogeneousTransformationMatrix3Dimensional.from_xyz_rpy(
        x=1.0, y=2.0, z=3.0, roll=0.1, pitch=0.2, yaw=0.3
    )
    twist = Twist(
        angular_velocity=Vector3Dimensional(x=0.4, y=-0.5, z=0.6),
        linear_velocity=Vector3Dimensional(x=1.0, y=0.5, z=-2.0),
    )

    moved = transformation @ twist

    np.testing.assert_allclose(
        moved.to_np(),
        adjoint_matrix_from_homogeneous_matrix(transformation.to_np()) @ twist.to_np(),
    )


def test_a_change_moves_a_planar_pose_as_composing_with_its_change():
    """
    A pose is moved as the change it is the value of.
    """
    transformation = HomogeneousTransformationMatrix2Dimensional.from_xy_yaw(
        x=1.0, y=2.0, yaw=0.5
    )
    pose = Pose2Dimensional.from_xy_yaw(x=3.0, y=-1.0, yaw=0.2)

    np.testing.assert_allclose(
        (transformation @ pose).to_coordinate_change().to_np(),
        (transformation @ pose.to_coordinate_change()).to_np(),
    )


def test_a_transformation_refuses_a_value_of_another_dimension():
    """
    A 3D transformation does not act on a planar point, nor a planar one on a 3D point.
    """
    with pytest.raises(UnsupportedOperationError):
        HomogeneousTransformationMatrix3Dimensional() @ Point2Dimensional(x=1, y=0)
    with pytest.raises(UnsupportedOperationError):
        HomogeneousTransformationMatrix2Dimensional() @ Point3Dimensional(x=1)


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
    SO(3) is the subgroup of SE(3) without translation, so a rotation is a
    transformation.
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
