"""
Which groups of rigid motions act on which spatial types, and how the symbolic values
and changes are split.
"""

from __future__ import annotations

import numpy as np
import pytest

from krrood.symbolic_math.exceptions import UnsupportedOperationError

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
    SpecialOrthogonalGroupAction,
    SymbolicSpatialChange,
    SymbolicSpatialValue,
    Vector3Dimensional,
)
from krrood_spatial_math.spatial_types_2_dimensional import (
    HomogeneousTransformationMatrix2Dimensional,
    Point2Dimensional,
)

# %% rigid motions of either dimension, and rotations of 3D values


def test_rigid_motions_and_rotations_are_separate_group_actions():
    """
    A planar value is moved by rigid motions of its plane but turned by no rotation.
    """
    assert not issubclass(SpecialEuclideanGroupAction, SpecialOrthogonalGroupAction)
    assert issubclass(Point2Dimensional, SpecialEuclideanGroupAction)
    assert not issubclass(Point2Dimensional, SpecialOrthogonalGroupAction)


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
    [Point3Dimensional, Vector3Dimensional, RotationMatrix, Pose3Dimensional],
)
def test_values_stored_as_homogeneous_coordinates_are_moved_by_rigid_motions(
    spatial_type,
):
    """
    Values stored as homogeneous coordinates are moved and turned by multiplying them.
    """
    assert issubclass(spatial_type, HomogeneousCoordinates)
    assert issubclass(spatial_type, SpecialEuclideanGroupAction)
    assert issubclass(spatial_type, SpecialOrthogonalGroupAction)


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
