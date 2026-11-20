"""
The frames spatial values are given in, and how values move between frames of one space.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from krrood.adapters.json_serializer import from_json, to_json
import krrood.symbolic_math.symbolic_math as sm
from krrood_spatial_math.line import Line
from krrood_spatial_math.point import GenericPoint, Point
from krrood_spatial_math.vector import GenericVector, Vector
from krrood_spatial_math.cartesian_frame_3_dimensional import CartesianFrame3Dimensional
from krrood_spatial_math.coordinate_system import (
    CoordinateRepresentation,
    GenericCoordinateSystem,
    GenericCoordinate,
    RigidCoordinateChange,
)
from krrood_spatial_math.exceptions import (
    FrameOutsideSpaceError,
    FramesOfDifferentSpacesError,
    MissingCartesianFrame3DimensionalResolver,
    MissingReferenceFrameError,
    ReferenceFrameMismatchError,
)
from krrood_spatial_math.spatial_types import (
    SymbolicSpatialValue,
    AxisAngle,
    HomogeneousTransformationMatrix3Dimensional,
    Point3Dimensional,
    Line3Dimensional,
    Pose3Dimensional,
    Quaternion,
    RotationMatrix,
    Vector3Dimensional,
)
from krrood_spatial_math.spatial_types_2_dimensional import (
    EmbeddedPlaneProjection,
    Pose2Dimensional,
)

from .dataset.frames import KnownFrames, NamedFrame, Room
from .dataset.mounted_frames import MountedFrame

# %% frames fixed on another frame


@pytest.fixture
def table_frame(room) -> NamedFrame:
    """
    A table in a room, so that values given in frames fixed on it can be expressed in
    it.
    """
    return NamedFrame("table", space=room)


@pytest.fixture
def shelf_frame(table_frame) -> MountedFrame:
    return MountedFrame(
        "shelf",
        space=table_frame.space,
        mount_T_self=HomogeneousTransformationMatrix3Dimensional.from_xyz_rpy(
            x=1, z=2, reference_frame=table_frame
        ),
    )


# %% values given in a frame


spatial_values = [
    Point3Dimensional,
    Vector3Dimensional,
    Quaternion,
    AxisAngle,
    Pose3Dimensional,
]


@pytest.mark.parametrize("spatial_value", spatial_values)
def test_a_spatial_value_is_a_value_given_in_a_cartesian_frame(spatial_value):
    """
    A shelf fixed on the table, one metre along its x-axis and two up.
    """
    assert issubclass(spatial_value, SymbolicSpatialValue)


@pytest.mark.parametrize("spatial_value", spatial_values)
def test_a_spatial_value_is_given_in_a_three_dimensional_cartesian_frame(
    spatial_value,
):
    """
    Each 3D value binds the kind of frame it is given in, through its value type.
    """
    assert issubclass(spatial_value, CoordinateRepresentation)
    assert (
        spatial_value.get_type_of_generic_parameter(parameter=GenericCoordinateSystem)
        is CartesianFrame3Dimensional
    )


@pytest.mark.parametrize("spatial_value", spatial_values)
def test_a_spatial_values_coordinates_are_given_in_its_reference_frame(
    spatial_value, table_frame
):
    value = spatial_value(reference_frame=table_frame)

    assert value.coordinate_system is table_frame


# %% a point in a frame's space as an affine point


def test_a_three_dimensional_point_is_given_in_a_three_dimensional_cartesian_frame():
    assert (
        Point3Dimensional.get_type_of_generic_parameter(
            parameter=GenericCoordinateSystem
        )
        is CartesianFrame3Dimensional
    )


def test_a_three_dimensional_point_is_a_point():
    """
    A 3D point measures distances, since its frame is Cartesian.
    """
    assert issubclass(Point3Dimensional, Point)


def test_a_three_dimensional_point_stores_its_coordinates_as_symbolic_scalars():
    assert (
        Point3Dimensional.get_type_of_generic_parameter(parameter=GenericCoordinate)
        is sm.Scalar
    )


def test_a_three_dimensional_point_is_moved_by_a_three_dimensional_vector():
    assert (
        Point3Dimensional.get_type_of_generic_parameter(parameter=GenericVector)
        is Vector3Dimensional
    )


# %% a vector in a frame's space


def test_a_three_dimensional_vector_is_a_vector_of_a_three_dimensional_frame():
    """
    A 3D vector measures lengths and is given in a 3D Cartesian frame.
    """
    assert issubclass(Vector3Dimensional, Vector)
    assert (
        Vector3Dimensional.get_type_of_generic_parameter(
            parameter=GenericCoordinateSystem
        )
        is CartesianFrame3Dimensional
    )


def test_the_difference_of_two_three_dimensional_points_moves_one_onto_the_other():
    start = Point3Dimensional(1, 2, 3)
    end = Point3Dimensional(4, 6, 3)

    np.testing.assert_allclose((start + (end - start)).to_np(), end.to_np())


# %% a line in a frame's space as an affine line


def test_a_three_dimensional_line_runs_through_a_point_along_a_vector():
    assert issubclass(Line3Dimensional, Line)
    assert (
        Line3Dimensional.get_type_of_generic_parameter(parameter=GenericPoint)
        is Point3Dimensional
    )
    assert (
        Line3Dimensional.get_type_of_generic_parameter(parameter=GenericVector)
        is Vector3Dimensional
    )


# %% a transformation as a change between two frames


def test_a_transformation_is_a_rigid_coordinate_change():
    assert issubclass(
        HomogeneousTransformationMatrix3Dimensional, RigidCoordinateChange
    )


def test_a_transformation_is_no_value_given_in_a_frame():
    assert not issubclass(
        HomogeneousTransformationMatrix3Dimensional, CoordinateRepresentation
    )


def test_a_transformation_binds_cartesian_frames_as_its_coordinate_systems():
    assert (
        HomogeneousTransformationMatrix3Dimensional.get_type_of_generic_parameter(
            parameter=GenericCoordinateSystem
        )
        is CartesianFrame3Dimensional
    )


def test_a_transformation_changes_from_its_child_frame_to_its_reference_frame(
    table_frame, camera_frame
):
    table_T_camera = HomogeneousTransformationMatrix3Dimensional(
        reference_frame=table_frame, child_frame=camera_frame
    )

    assert (
        table_T_camera.source_coordinate_system,
        table_T_camera.target_coordinate_system,
    ) == (camera_frame, table_frame)


def test_an_inverted_transformation_changes_back_to_its_child_frame(
    table_frame, camera_frame
):
    camera_T_table = HomogeneousTransformationMatrix3Dimensional(
        reference_frame=table_frame, child_frame=camera_frame
    ).inverse()

    assert (
        camera_T_table.source_coordinate_system,
        camera_T_table.target_coordinate_system,
    ) == (table_frame, camera_frame)


def test_composed_transformations_change_from_the_first_source_to_the_last_target(
    table_frame, camera_frame, gripper_frame
):
    table_T_camera = HomogeneousTransformationMatrix3Dimensional(
        reference_frame=table_frame, child_frame=camera_frame
    )
    camera_T_gripper = HomogeneousTransformationMatrix3Dimensional(
        reference_frame=camera_frame, child_frame=gripper_frame
    )

    table_T_gripper = table_T_camera @ camera_T_gripper

    assert (
        table_T_gripper.source_coordinate_system,
        table_T_gripper.target_coordinate_system,
    ) == (gripper_frame, table_frame)


# %% combining values expressed in frames


def test_a_transformation_from_parts_in_different_frames_raises(
    table_frame, camera_frame
):
    position = Point3Dimensional(1, 2, 3, reference_frame=table_frame)
    rotation = RotationMatrix.from_rpy(yaw=0.5, reference_frame=camera_frame)

    with pytest.raises(ReferenceFrameMismatchError):
        HomogeneousTransformationMatrix3Dimensional.from_point_rotation_matrix(
            position, rotation
        )


def test_a_transformed_point_is_expressed_in_the_transformations_frame(
    table_frame, camera_frame
):
    table_T_camera = HomogeneousTransformationMatrix3Dimensional.from_xyz_rpy(
        x=1, reference_frame=table_frame, child_frame=camera_frame
    )

    point_in_table = table_T_camera @ Point3Dimensional(
        0, 2, 3, reference_frame=camera_frame
    )

    assert point_in_table.reference_frame is table_frame
    assert np.allclose(point_in_table.to_np()[:3], [1, 2, 3])


def test_a_point_changed_there_and_back_keeps_its_coordinates(
    table_frame, camera_frame
):
    table_T_camera = HomogeneousTransformationMatrix3Dimensional.from_xyz_rpy(
        x=1, y=2, yaw=0.5, reference_frame=table_frame, child_frame=camera_frame
    )
    point_in_camera = Point3Dimensional(0, 2, 3, reference_frame=camera_frame)

    point_copy = table_T_camera.inverse() @ (table_T_camera @ point_in_camera)

    assert point_copy.reference_frame is camera_frame
    assert np.allclose(point_copy.to_np(), point_in_camera.to_np())


def test_a_transformation_applies_to_a_point_without_a_frame(table_frame, camera_frame):
    table_T_camera = HomogeneousTransformationMatrix3Dimensional(
        reference_frame=table_frame, child_frame=camera_frame
    )

    point_in_table = table_T_camera @ Point3Dimensional(0, 2, 3)

    assert point_in_table.reference_frame is table_frame


# %% bringing a value onto the plane of another frame


def test_a_point_projected_onto_another_frames_plane_is_expressed_in_that_frame(
    table_frame, shelf_frame
):
    projection = EmbeddedPlaneProjection(plane_frame=shelf_frame)

    point_on_shelf = projection.project(
        Point3Dimensional(1, 2, 3, reference_frame=table_frame)
    )

    assert point_on_shelf.reference_frame is shelf_frame
    assert np.allclose(
        np.concatenate([point_on_shelf.x.to_np(), point_on_shelf.y.to_np()]), [0, 2]
    )


def test_a_pose_flattened_onto_another_frames_plane_is_expressed_in_that_frame(
    table_frame, shelf_frame
):
    pose_in_table = Pose3Dimensional.from_xyz_rpy(
        x=3, y=2, z=5, yaw=0.5, reference_frame=table_frame
    )

    pose_on_shelf = Pose2Dimensional.from_pose(pose_in_table, plane_frame=shelf_frame)

    assert pose_on_shelf.reference_frame is shelf_frame
    assert np.allclose(
        np.concatenate(
            [
                pose_on_shelf.x.to_np(),
                pose_on_shelf.y.to_np(),
                pose_on_shelf.yaw.to_np(),
            ]
        ),
        [2, 2, 0.5],
    )


# %% expressing a value in a frame of the same space


def test_a_frame_expresses_a_value_of_its_own_space(table_frame, shelf_frame):
    """
    A value given in a frame of the same space is changed into the frame's coordinates.
    """
    point_in_table = Point3Dimensional(1, 0, 2, reference_frame=table_frame)

    point_in_shelf = shelf_frame.express_in_own_frame(value=point_in_table)

    assert point_in_shelf.reference_frame is shelf_frame
    assert np.allclose(point_in_shelf.to_np()[:3], [0, 0, 0])


def test_a_frame_cannot_express_a_value_given_in_no_frame(table_frame):
    """
    Without a frame, nothing says which coordinates the value's numbers are.
    """
    with pytest.raises(MissingReferenceFrameError):
        table_frame.express_in_own_frame(value=Point3Dimensional(1, 2, 3))


def test_a_frame_cannot_express_a_value_of_a_frame_in_no_space(table_frame):
    """
    A frame in no known space has no change to any other frame.
    """
    loose_frame = NamedFrame("loose")

    with pytest.raises(FrameOutsideSpaceError):
        table_frame.express_in_own_frame(
            value=Point3Dimensional(reference_frame=loose_frame)
        )


def test_a_frame_in_no_space_cannot_express_a_value(table_frame):
    """
    The frame expressing a value must chart a known space too.
    """
    loose_frame = NamedFrame("loose")

    with pytest.raises(FrameOutsideSpaceError):
        loose_frame.express_in_own_frame(
            value=Point3Dimensional(reference_frame=table_frame)
        )


def test_a_frame_cannot_express_a_value_of_another_space(table_frame):
    """
    Points of two spaces relate only through a map between the spaces.
    """
    frame_in_another_room = NamedFrame("door", space=Room())

    with pytest.raises(FramesOfDifferentSpacesError):
        table_frame.express_in_own_frame(
            value=Point3Dimensional(reference_frame=frame_in_another_room)
        )


# %% serializing the frames of a value


def test_a_deserialized_point_refers_to_the_readers_frame(table_frame):
    point = Point3Dimensional(1, 2, 3, reference_frame=table_frame)
    data = json.loads(json.dumps(to_json(point)))
    known_frames = KnownFrames()
    known_frames.add(key=table_frame.id, tracked_object=table_frame)

    point_copy = from_json(data, **known_frames.create_kwargs())

    assert point_copy.reference_frame is table_frame
    assert np.allclose(point_copy.to_np(), point.to_np())


def test_a_transformation_keeps_both_of_its_frames(table_frame, camera_frame):
    table_T_camera = HomogeneousTransformationMatrix3Dimensional.from_xyz_rpy(
        x=1, reference_frame=table_frame, child_frame=camera_frame
    )
    known_frames = KnownFrames()
    known_frames.add(key=table_frame.id, tracked_object=table_frame)
    known_frames.add(key=camera_frame.id, tracked_object=camera_frame)

    copy = from_json(to_json(table_T_camera), **known_frames.create_kwargs())

    assert copy.reference_frame is table_frame
    assert copy.child_frame is camera_frame


def test_a_point_referring_to_a_frame_cannot_be_read_without_a_resolver(
    table_frame,
):
    data = to_json(Point3Dimensional(1, 2, 3, reference_frame=table_frame))

    with pytest.raises(MissingCartesianFrame3DimensionalResolver):
        from_json(data)


def test_a_point_without_a_frame_is_read_without_a_resolver():
    point = Point3Dimensional(1, 2, 3)

    point_copy = from_json(to_json(point))

    assert point_copy.reference_frame is None
    assert np.allclose(point_copy.to_np(), point.to_np())
