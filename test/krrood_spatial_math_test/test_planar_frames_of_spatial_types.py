from __future__ import annotations

import numpy as np
import pytest

import krrood.symbolic_math.symbolic_math as sm
from krrood.adapters.json_serializer import from_json, to_json
from krrood_spatial_math.point import MetricPoint
from krrood_spatial_math.vector import VectorT, MetricVector
from krrood_spatial_math.cartesian_frame_2_dimensional import CartesianFrame2Dimensional
from krrood_spatial_math.coordinate_system import (
    CoordinateSystemT,
    CoordinateT,
    RigidCoordinateChange,
)
from krrood_spatial_math.exceptions import (
    FramesOnDifferentPlanesError,
    MirroredPlaneFramesError,
    ReferenceFrameMismatchError,
    XAxisNormalToPlaneError,
)
from krrood_spatial_math.space import EuclideanSpace2Dimensional
from krrood_spatial_math.spatial_types import (
    SymbolicSpatialValue3Dimensional,
    HomogeneousTransformationMatrix3Dimensional,
    HomogeneousTransformationMatrix2Dimensional,
    CartesianPoint2Dimensional,
    CartesianPoint3Dimensional,
    Pose3Dimensional,
    Pose2Dimensional,
    FramePlane,
    PlanarFrame,
    Vector2Dimensional,
)

from .dataset.frames import KnownFrames, NamedFrame, Room
from .dataset.mounted_frames import MountedFrame

# %% frames in one room


@pytest.fixture
def floor_frame(room) -> NamedFrame:
    return NamedFrame("floor", space=room)


def mounted_on(
    floor_frame: NamedFrame, name: str, **pose_on_floor: float
) -> MountedFrame:
    """
    :param floor_frame: The frame the new frame is fixed on.
    :param name: What the new frame is called.
    :param pose_on_floor: The x, y, z, roll, pitch and yaw of the new frame on the
        floor.
    :return: A frame fixed on the floor, in the floor's room.
    """
    return MountedFrame(
        name,
        space=floor_frame.space,
        mount_T_self=HomogeneousTransformationMatrix3Dimensional.from_xyz_rpy(
            **pose_on_floor, reference_frame=floor_frame
        ),
    )


@pytest.fixture
def robot_base_frame(floor_frame) -> MountedFrame:
    return mounted_on(floor_frame, "robot base", x=1, y=2, yaw=0.5)


@pytest.fixture
def upside_down_frame(floor_frame) -> MountedFrame:
    return mounted_on(floor_frame, "upside down", x=3, roll=np.pi)


@pytest.fixture
def shelf_frame(floor_frame) -> MountedFrame:
    return mounted_on(floor_frame, "shelf", x=1, z=2)


@pytest.fixture
def wall_frame(floor_frame) -> MountedFrame:
    return mounted_on(floor_frame, "wall", roll=np.pi / 2)


def coordinates(*values: sm.Scalar) -> np.ndarray:
    """
    :param values: Constant symbolic scalars.
    :return: Their numbers, in the order given.
    """
    return np.concatenate([value.to_np() for value in values])


# %% the plane a frame's x- and y-axes span


def test_a_frames_xy_plane_is_a_two_dimensional_euclidean_space():
    assert issubclass(FramePlane, EuclideanSpace2Dimensional)


def test_a_plane_lies_in_the_space_of_its_frame(floor_frame, room):
    assert FramePlane(floor_frame).ambient_space is room


def test_a_frame_moved_within_a_plane_spans_that_plane(floor_frame, robot_base_frame):
    assert FramePlane(robot_base_frame) == FramePlane(floor_frame)


def test_a_frame_turned_upside_down_spans_the_same_plane(
    floor_frame, upside_down_frame
):
    assert FramePlane(upside_down_frame) == FramePlane(floor_frame)


def test_a_frame_lifted_off_a_plane_spans_another_plane(floor_frame, shelf_frame):
    assert FramePlane(shelf_frame) != FramePlane(floor_frame)


def test_a_tilted_frame_spans_another_plane(floor_frame, wall_frame):
    assert FramePlane(wall_frame) != FramePlane(floor_frame)


def test_frames_in_no_known_space_share_a_plane_only_with_themselves(
    table_frame, camera_frame
):
    assert FramePlane(table_frame) == FramePlane(table_frame)
    assert FramePlane(table_frame) != FramePlane(camera_frame)


def test_frames_in_different_spaces_span_different_planes(floor_frame):
    other_floor_frame = NamedFrame("other floor", space=Room())

    assert FramePlane(other_floor_frame) != FramePlane(floor_frame)


def test_equal_planes_hash_alike(floor_frame, robot_base_frame):
    assert hash(FramePlane(robot_base_frame)) == hash(FramePlane(floor_frame))


# %% the frame a plane's values are given in


def test_a_plane_frame_is_a_two_dimensional_cartesian_frame():
    assert issubclass(PlanarFrame, CartesianFrame2Dimensional)


def test_a_plane_frame_lies_on_the_xy_plane_of_its_frame(floor_frame):
    assert PlanarFrame(floor_frame).space == FramePlane(floor_frame)


def test_a_plane_frame_is_called_like_its_frame(floor_frame):
    assert PlanarFrame(floor_frame).name == floor_frame.name


def test_the_plane_frames_of_one_frame_are_one_frame(floor_frame):
    plane_frame = PlanarFrame(floor_frame)
    plane_frame_copy = PlanarFrame(floor_frame)

    assert plane_frame == plane_frame_copy
    assert plane_frame.id == plane_frame_copy.id
    assert hash(plane_frame) == hash(plane_frame_copy)


def test_the_plane_frames_of_two_frames_on_one_plane_are_two_frames(
    floor_frame, robot_base_frame
):
    plane_frame = PlanarFrame(floor_frame)
    other_plane_frame = PlanarFrame(robot_base_frame)

    assert plane_frame != other_plane_frame
    assert plane_frame.id != other_plane_frame.id


def test_a_plane_frame_is_told_apart_from_its_frame(floor_frame):
    assert PlanarFrame(floor_frame).id != floor_frame.id


# %% values on a frame's plane

planar_values = [CartesianPoint2Dimensional, Vector2Dimensional, Pose2Dimensional]


@pytest.mark.parametrize("planar_value", planar_values)
def test_a_planar_value_is_given_in_the_frame_on_its_reference_frames_plane(
    planar_value, floor_frame
):
    value = planar_value(reference_frame=floor_frame)

    assert value.coordinate_system == PlanarFrame(floor_frame)


@pytest.mark.parametrize("planar_value", planar_values)
def test_a_planar_value_keeps_its_reference_frame(planar_value, floor_frame):
    assert planar_value(reference_frame=floor_frame).reference_frame is floor_frame


@pytest.mark.parametrize("planar_value", planar_values)
def test_a_planar_value_without_a_reference_frame_has_no_coordinate_system(
    planar_value,
):
    assert planar_value().coordinate_system is None


@pytest.mark.parametrize("planar_value", planar_values)
def test_a_planar_value_is_no_value_given_in_a_three_dimensional_frame(planar_value):
    assert not issubclass(planar_value, SymbolicSpatialValue3Dimensional)


def test_a_two_dimensional_point_is_given_in_a_two_dimensional_cartesian_frame():
    assert (
        CartesianPoint2Dimensional.get_type_of_generic_parameter(
            parameter=CoordinateSystemT
        )
        is CartesianFrame2Dimensional
    )


def test_a_two_dimensional_point_is_a_cartesian_point():
    assert issubclass(CartesianPoint2Dimensional, MetricPoint)


def test_the_euclidean_distance_between_two_dimensional_points_is_measured_on_their_plane():
    origin = CartesianPoint2Dimensional(1, 2)
    other = CartesianPoint2Dimensional(4, 6)

    assert float(origin.euclidean_distance(other)) == 5.0


def test_a_two_dimensional_point_stores_its_coordinates_as_symbolic_scalars():
    assert (
        CartesianPoint2Dimensional.get_type_of_generic_parameter(parameter=CoordinateT)
        is sm.Scalar
    )


def test_a_two_dimensional_point_is_moved_by_a_two_dimensional_vector():
    assert (
        CartesianPoint2Dimensional.get_type_of_generic_parameter(parameter=VectorT)
        is Vector2Dimensional
    )


# %% vectors on a frame's plane


def test_a_two_dimensional_vector_is_a_cartesian_vector_of_a_two_dimensional_frame():
    assert issubclass(Vector2Dimensional, MetricVector)
    assert (
        Vector2Dimensional.get_type_of_generic_parameter(parameter=CoordinateSystemT)
        is CartesianFrame2Dimensional
    )


def test_the_difference_of_two_points_on_a_plane_is_the_vector_between_them(
    floor_frame,
):
    start = CartesianPoint2Dimensional(1, 2, reference_frame=floor_frame)
    end = CartesianPoint2Dimensional(4, 6, reference_frame=floor_frame)

    difference = end - start

    assert isinstance(difference, Vector2Dimensional)
    assert difference.reference_frame is floor_frame
    assert np.allclose(coordinates(difference.x, difference.y), [3, 4])


def test_a_point_moved_by_the_difference_to_another_point_reaches_it(floor_frame):
    start = CartesianPoint2Dimensional(1, 2, reference_frame=floor_frame)
    end = CartesianPoint2Dimensional(4, 6, reference_frame=floor_frame)

    moved = start + (end - start)

    assert moved.reference_frame is floor_frame
    assert np.allclose(moved.to_np(), end.to_np())


def test_points_on_the_planes_of_two_frames_have_no_difference(
    floor_frame, robot_base_frame
):
    with pytest.raises(ReferenceFrameMismatchError):
        CartesianPoint2Dimensional(
            1, 2, reference_frame=floor_frame
        ) - CartesianPoint2Dimensional(reference_frame=robot_base_frame)


def test_a_point_is_not_moved_by_a_vector_on_another_frames_plane(
    floor_frame, robot_base_frame
):
    with pytest.raises(ReferenceFrameMismatchError):
        CartesianPoint2Dimensional(reference_frame=floor_frame) + Vector2Dimensional(
            1, 0, reference_frame=robot_base_frame
        )


def test_two_dimensional_vectors_add_scale_and_subtract():
    first, second = Vector2Dimensional(1, 2), Vector2Dimensional(3, -1)

    assert np.allclose(((first + second) * 2 - second).to_np().ravel(), [5, 3])


def test_the_length_of_a_two_dimensional_vector_is_measured_on_its_plane():
    assert float(Vector2Dimensional(3, 4).norm()) == 5.0


def test_the_dot_product_of_perpendicular_vectors_on_a_plane_is_zero():
    assert float(Vector2Dimensional(1, 2).dot(Vector2Dimensional(-2, 1))) == 0.0


# %% rigid changes between frames on one plane


def test_a_planar_transformation_is_a_rigid_change_of_two_dimensional_frames():
    assert issubclass(
        HomogeneousTransformationMatrix2Dimensional, RigidCoordinateChange
    )
    assert (
        HomogeneousTransformationMatrix2Dimensional.get_type_of_generic_parameter(
            parameter=CoordinateSystemT
        )
        is CartesianFrame2Dimensional
    )


def test_a_planar_transformation_changes_between_the_plane_frames_of_its_frames(
    floor_frame, robot_base_frame
):
    floor_T_robot_base = HomogeneousTransformationMatrix2Dimensional(
        reference_frame=floor_frame, child_frame=robot_base_frame
    )

    assert (
        floor_T_robot_base.source_coordinate_system,
        floor_T_robot_base.target_coordinate_system,
    ) == (PlanarFrame(robot_base_frame), PlanarFrame(floor_frame))


def test_a_planar_transformation_without_frames_has_no_coordinate_systems():
    transformation = HomogeneousTransformationMatrix2Dimensional()

    assert (
        transformation.source_coordinate_system,
        transformation.target_coordinate_system,
    ) == (None, None)


def test_a_change_within_one_plane_becomes_the_planar_transformation_of_its_motion(
    floor_frame, robot_base_frame
):
    floor_T_robot_base = (
        HomogeneousTransformationMatrix2Dimensional.from_transformation(
            robot_base_frame.mount_T_self
        )
    )

    assert floor_T_robot_base.reference_frame is floor_frame
    assert floor_T_robot_base.child_frame is robot_base_frame
    assert np.allclose(
        coordinates(floor_T_robot_base.x, floor_T_robot_base.y, floor_T_robot_base.yaw),
        [1, 2, 0.5],
    )


@pytest.mark.parametrize("frame_off_the_floor", ["shelf_frame", "wall_frame"])
def test_a_change_between_two_planes_has_no_planar_transformation(
    frame_off_the_floor, request
):
    frame = request.getfixturevalue(frame_off_the_floor)

    with pytest.raises(FramesOnDifferentPlanesError):
        HomogeneousTransformationMatrix2Dimensional.from_transformation(
            frame.mount_T_self
        )


def test_a_change_to_a_frame_facing_the_other_side_of_a_plane_has_no_planar_transformation(
    upside_down_frame,
):
    with pytest.raises(MirroredPlaneFramesError):
        HomogeneousTransformationMatrix2Dimensional.from_transformation(
            upside_down_frame.mount_T_self
        )


def test_a_symbolic_change_becomes_a_planar_transformation_unchecked(floor_frame):
    height = sm.FloatVariable(name="height")
    lifted = HomogeneousTransformationMatrix3Dimensional.from_xyz_rpy(
        x=1, z=height, reference_frame=floor_frame
    )

    floor_T_lifted = HomogeneousTransformationMatrix2Dimensional.from_transformation(
        lifted
    )

    assert np.allclose(floor_T_lifted.x.to_np(), [1])


def test_a_planar_transformation_moves_a_point_into_its_reference_frame(
    floor_frame, robot_base_frame
):
    floor_T_robot_base = HomogeneousTransformationMatrix2Dimensional.from_xy_yaw(
        x=1,
        y=2,
        yaw=np.pi / 2,
        reference_frame=floor_frame,
        child_frame=robot_base_frame,
    )

    point_on_floor = floor_T_robot_base @ CartesianPoint2Dimensional(
        1, 0, reference_frame=robot_base_frame
    )

    assert point_on_floor.reference_frame is floor_frame
    assert np.allclose(coordinates(point_on_floor.x, point_on_floor.y), [1, 3])


def test_a_planar_transformation_turns_a_vector_without_shifting_it(
    floor_frame, robot_base_frame
):
    floor_T_robot_base = HomogeneousTransformationMatrix2Dimensional.from_xy_yaw(
        x=1,
        y=2,
        yaw=np.pi / 2,
        reference_frame=floor_frame,
        child_frame=robot_base_frame,
    )

    vector_on_floor = floor_T_robot_base @ Vector2Dimensional(
        1, 0, reference_frame=robot_base_frame
    )

    assert vector_on_floor.reference_frame is floor_frame
    assert np.allclose(coordinates(vector_on_floor.x, vector_on_floor.y), [0, 1])


def test_a_transformation_turns_a_vector_on_a_plane_without_shifting_it(
    floor_frame, robot_base_frame
):
    floor_T_robot_base = HomogeneousTransformationMatrix3Dimensional.from_xyz_rpy(
        x=1,
        y=2,
        yaw=np.pi / 2,
        reference_frame=floor_frame,
        child_frame=robot_base_frame,
    )

    vector_on_floor = Vector2Dimensional(
        1, 0, reference_frame=robot_base_frame
    ).transform(floor_T_robot_base)

    assert vector_on_floor.reference_frame is floor_frame
    assert np.allclose(coordinates(vector_on_floor.x, vector_on_floor.y), [0, 1])


def test_a_planar_transformation_moves_a_pose_into_its_reference_frame(
    floor_frame, robot_base_frame
):
    floor_T_robot_base = HomogeneousTransformationMatrix2Dimensional.from_xy_yaw(
        x=1,
        y=2,
        yaw=np.pi / 2,
        reference_frame=floor_frame,
        child_frame=robot_base_frame,
    )

    pose_on_floor = floor_T_robot_base @ Pose2Dimensional.from_xy_yaw(
        x=1, yaw=0.25, reference_frame=robot_base_frame
    )

    assert pose_on_floor.reference_frame is floor_frame
    assert np.allclose(
        coordinates(pose_on_floor.x, pose_on_floor.y, pose_on_floor.yaw),
        [1, 3, np.pi / 2 + 0.25],
    )


def test_composed_planar_transformations_change_from_the_first_source_to_the_last_target(
    floor_frame, robot_base_frame, upside_down_frame
):
    floor_T_robot_base = HomogeneousTransformationMatrix2Dimensional.from_xy_yaw(
        x=1, yaw=np.pi / 2, reference_frame=floor_frame, child_frame=robot_base_frame
    )
    robot_base_T_other = HomogeneousTransformationMatrix2Dimensional.from_xy_yaw(
        x=1, reference_frame=robot_base_frame, child_frame=upside_down_frame
    )

    floor_T_other = floor_T_robot_base @ robot_base_T_other

    assert floor_T_other.reference_frame is floor_frame
    assert floor_T_other.child_frame is upside_down_frame
    assert np.allclose(
        coordinates(floor_T_other.x, floor_T_other.y, floor_T_other.yaw),
        [1, 1, np.pi / 2],
    )


def test_a_point_changed_there_and_back_on_a_plane_keeps_its_coordinates(
    floor_frame, robot_base_frame
):
    floor_T_robot_base = HomogeneousTransformationMatrix2Dimensional.from_xy_yaw(
        x=1, y=2, yaw=0.5, reference_frame=floor_frame, child_frame=robot_base_frame
    )
    point = CartesianPoint2Dimensional(3, 4, reference_frame=robot_base_frame)

    point_copy = floor_T_robot_base.inverse() @ (floor_T_robot_base @ point)

    assert point_copy.reference_frame is robot_base_frame
    assert np.allclose(point_copy.to_np(), point.to_np())


def test_a_planar_transformation_moves_a_point_as_its_change_does(
    floor_frame, robot_base_frame
):
    floor_T_robot_base = robot_base_frame.mount_T_self
    point = CartesianPoint2Dimensional(3, 4, reference_frame=robot_base_frame)

    point_on_floor = (
        HomogeneousTransformationMatrix2Dimensional.from_transformation(
            floor_T_robot_base
        )
        @ point
    )

    expected = CartesianPoint2Dimensional.from_cartesian_point_3_dimensional(
        floor_T_robot_base @ point.to_cartesian_point_3_dimensional()
    )
    assert np.allclose(point_on_floor.to_np(), expected.to_np())


def test_a_two_dimensional_vector_keeps_its_coordinates_and_frame_in_json(floor_frame):
    vector = Vector2Dimensional(1, 2, reference_frame=floor_frame)
    known_frames = KnownFrames()
    known_frames.add(key=floor_frame.id, tracked_object=floor_frame)

    copy = from_json(to_json(vector), **known_frames.create_kwargs())

    assert copy.reference_frame is floor_frame
    assert np.allclose(copy.to_np(), vector.to_np())


def test_a_planar_transformation_keeps_its_motion_and_frames_in_json(
    floor_frame, robot_base_frame
):
    floor_T_robot_base = HomogeneousTransformationMatrix2Dimensional.from_xy_yaw(
        x=1, y=2, yaw=0.5, reference_frame=floor_frame, child_frame=robot_base_frame
    )
    known_frames = KnownFrames()
    known_frames.add(key=floor_frame.id, tracked_object=floor_frame)
    known_frames.add(key=robot_base_frame.id, tracked_object=robot_base_frame)

    copy = from_json(to_json(floor_T_robot_base), **known_frames.create_kwargs())

    assert copy.reference_frame is floor_frame
    assert copy.child_frame is robot_base_frame
    assert np.allclose(copy.to_np(), floor_T_robot_base.to_np())


# %% a pose flattened onto its frame's plane


def test_a_flattened_pose_faces_the_way_its_x_axis_points_on_the_plane(floor_frame):
    pose = Pose3Dimensional.from_xyz_rpy(
        x=1, y=2, z=3, roll=0.3, pitch=0.2, yaw=0.5, reference_frame=floor_frame
    )

    pose_on_floor = Pose2Dimensional.from_pose(pose)

    x_axis = pose.rotation_matrix.to_np()[:2, 0]
    assert np.allclose(
        coordinates(pose_on_floor.x, pose_on_floor.y, pose_on_floor.yaw),
        [1, 2, np.arctan2(x_axis[1], x_axis[0])],
    )


def test_a_pose_whose_x_axis_is_normal_to_the_plane_cannot_be_flattened(
    floor_frame,
):
    pose = Pose3Dimensional.from_xyz_rpy(pitch=np.pi / 2, reference_frame=floor_frame)

    with pytest.raises(XAxisNormalToPlaneError):
        Pose2Dimensional.from_pose(pose)
