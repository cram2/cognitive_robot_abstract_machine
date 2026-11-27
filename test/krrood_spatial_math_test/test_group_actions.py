"""
How a change of frame moves each kind of value, and how the symbolic values and changes
are split.
"""

from __future__ import annotations

import numpy as np
import pytest

from krrood.symbolic_math.exceptions import UnsupportedOperationError

from krrood_spatial_math.coordinate_system import (
    CoordinateRepresentation,
    RigidCoordinateChange,
)
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
    VectorPair,
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
        (RotationMatrix, Orientation),
        (Twist, VectorPair),
        (ScrewAxis, VectorPair),
        (Displacement, VectorPair),
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
        (shifted_rotation @ orientation).to_np(),
        (rotation.to_coordinate_change() @ orientation).to_np(),
    )


# %% rotation matrices as orientations, and the rotation of a transformation


@pytest.fixture
def quarter_turn() -> RotationMatrix:
    """
    The orientation a quarter turn about the z-axis gives a frame.
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


def test_a_rotation_matrix_is_an_orientation_and_never_a_change():
    """
    A rotation matrix gives which way a frame faces, as a quaternion does.
    """
    assert issubclass(RotationMatrix, Orientation)
    assert not issubclass(RotationMatrix, SymbolicSpatialChange)


def test_the_rotation_of_a_transformation_is_the_change_without_its_translation(
    table_frame, shifted_quarter_turn
):
    """
    The rotation turns what it is applied to as the transformation does, and shifts
    nothing.
    """
    shifted_quarter_turn.reference_frame = table_frame

    rotation = shifted_quarter_turn.rotation

    assert type(rotation) is HomogeneousTransformationMatrix3Dimensional
    assert rotation.reference_frame is table_frame
    np.testing.assert_array_equal(rotation.position.to_np()[:3], np.zeros(3))
    np.testing.assert_array_almost_equal(
        rotation.to_np()[:3, :3], shifted_quarter_turn.to_np()[:3, :3]
    )


def test_the_rotation_matrix_of_a_transformation_is_its_orientation(
    table_frame, shifted_quarter_turn
):
    shifted_quarter_turn.reference_frame = table_frame

    orientation = shifted_quarter_turn.rotation_matrix

    assert type(orientation) is RotationMatrix
    assert orientation.reference_frame is table_frame
    np.testing.assert_array_almost_equal(
        orientation.to_np()[:3, :3], shifted_quarter_turn.to_np()[:3, :3]
    )


def test_a_rotation_matrix_is_the_value_of_its_change(quarter_turn):
    """
    A rotation matrix and the rotation it is the value of convert into each other.
    """
    change = quarter_turn.to_coordinate_change()

    assert type(change) is HomogeneousTransformationMatrix3Dimensional
    np.testing.assert_array_almost_equal(
        RotationMatrix.from_coordinate_change(change=change).to_np(),
        quarter_turn.to_np(),
    )


def test_a_transformation_turns_a_rotation_matrix_by_its_rotation_only(
    quarter_turn, shifted_quarter_turn
):
    """
    A change of frame re-expresses an orientation, so the translation is dropped.
    """
    turned = shifted_quarter_turn @ quarter_turn

    assert type(turned) is RotationMatrix
    np.testing.assert_array_almost_equal(
        turned.to_np(),
        (shifted_quarter_turn.rotation @ quarter_turn.to_coordinate_change()).to_np(),
    )


def test_the_inverse_of_a_rotation_matrix_turns_back_and_has_no_reference_frame(
    table_frame,
):
    """
    The inverse is the orientation of the reference frame in the oriented frame, which a
    rotation matrix does not name.
    """
    table_R_camera = RotationMatrix.from_rpy(yaw=0.5, reference_frame=table_frame)

    camera_R_table = table_R_camera.inverse()

    assert camera_R_table.reference_frame is None
    np.testing.assert_array_almost_equal(
        camera_R_table.to_np()[:3, :3], table_R_camera.to_np()[:3, :3].T
    )


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
        RotationMatrix,
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
