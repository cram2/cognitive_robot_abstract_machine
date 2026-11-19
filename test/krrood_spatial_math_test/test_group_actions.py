"""
Which groups of rigid motions act on which spatial types, and how the symbolic values
and changes are split.
"""

from __future__ import annotations

import numpy as np
import pytest

from krrood_spatial_math.coordinate_system import (
    CoordinateRepresentation,
    RigidCoordinateChange,
)
from krrood_spatial_math.spatial_types import (
    AxisAngle,
    CartesianPoint3Dimensional,
    HomogeneousCoordinates,
    HomogeneousTransformationMatrix3Dimensional,
    Orientation,
    Pose3Dimensional,
    Quaternion,
    RotationMatrix,
    SpecialEuclideanGroupAction,
    SpecialOrthogonalGroupAction,
    SymbolicSpatialChange,
    SymbolicSpatialValue3Dimensional,
    Vector3Dimensional,
)
from krrood_spatial_math.spatial_types_2_dimensional import (
    HomogeneousTransformationMatrix2Dimensional,
)

# %% the rotation group is a subgroup of the rigid motion group


def test_whatever_rigid_motions_act_on_rotations_act_on_too():
    assert issubclass(SpecialEuclideanGroupAction, SpecialOrthogonalGroupAction)


@pytest.mark.parametrize(
    "spatial_type",
    [CartesianPoint3Dimensional, Vector3Dimensional, RotationMatrix, Pose3Dimensional],
)
def test_values_stored_as_homogeneous_coordinates_are_moved_by_rigid_motions(
    spatial_type,
):
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


# %% symbolic values and symbolic changes


@pytest.mark.parametrize(
    "value_type",
    [
        CartesianPoint3Dimensional,
        Vector3Dimensional,
        Quaternion,
        AxisAngle,
        Pose3Dimensional,
    ],
)
def test_a_symbolic_value_is_a_coordinate_representation(value_type):
    assert issubclass(value_type, SymbolicSpatialValue3Dimensional)
    assert issubclass(value_type, CoordinateRepresentation)


@pytest.mark.parametrize(
    "change_type",
    [
        HomogeneousTransformationMatrix3Dimensional,
        HomogeneousTransformationMatrix2Dimensional,
    ],
)
def test_a_symbolic_change_is_a_rigid_change_and_never_a_value(change_type):
    assert issubclass(change_type, SymbolicSpatialChange)
    assert issubclass(change_type, RigidCoordinateChange)
    assert not issubclass(change_type, SymbolicSpatialValue3Dimensional)
