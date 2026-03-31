from copy import deepcopy

import numpy as np
import pytest

from pycram.locations.costmaps import (
    OccupancyCostmap,
    GaussianCostmap,
    OrientationGenerator,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.spatial_types.spatial_types import Pose, Point3


# ---- Occupancy locations tests ----


def test_attachment_exclusion(immutable_model_world, rclpy_node):

    world, robot_view, context = immutable_model_world

    robot_view.root.parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(
            -1.5, 1, 0, reference_frame=world.root
        )
    )
    world.get_body_by_name("milk.stl").parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(
            -1.8, 1, 1, reference_frame=world.root
        )
    )

    test_world = deepcopy(world)
    with test_world.modify_world():
        test_world.move_branch(
            test_world.get_body_by_name("milk.stl"),
            test_world.get_body_by_name("r_gripper_tool_frame"),
        )
    o = OccupancyCostmap(
        distance_to_obstacle=0.2,
        height=200,
        width=200,
        resolution=0.02,
        robot_view=robot_view,
        origin=Pose.from_xyz_quaternion(-1.5, 1, 0, 0, 0, 0, 1, test_world.root),
        world=test_world,
    )

    assert 400 == np.sum(o.map[90:110, 90:110])
    assert np.sum(o.map[80:90, 90:110]) != 0


def test_merge_costmap(immutable_model_world):
    world, robot_view, context = immutable_model_world
    o = OccupancyCostmap(
        distance_to_obstacle=0.2,
        height=200,
        width=200,
        resolution=0.02,
        robot_view=robot_view,
        origin=Pose.from_xyz_quaternion(0, 0, 0, 0, 0, 0, 1, world.root),
        world=world,
    )
    o2 = OccupancyCostmap(
        distance_to_obstacle=0.2,
        height=200,
        width=200,
        resolution=0.02,
        robot_view=robot_view,
        origin=Pose.from_xyz_quaternion(0, 0, 0, 0, 0, 0, 1, world.root),
        world=world,
    )
    o3 = o + o2
    assert np.all(o.map == o3.map)
    o2.map[100:120, 100:120] = 0
    o3 = o + o2
    assert np.all(o3.map[100:120, 100:120] == 0)
    assert np.all(o3.map[0:100, 0:100] == o.map[0:100, 0:100])
    o2.map = np.zeros_like(o2.map)
    o3 = o + o2
    assert np.all(o3.map == o2.map)


def test_occupancy_robot_exclusion(immutable_model_world):
    world, robot_view, context = immutable_model_world
    robot_view.root.parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(10, 10)
    )
    occupancy_map = OccupancyCostmap(
        resolution=0.02,
        height=400,
        width=400,
        world=world,
        robot_view=robot_view,
        origin=Pose.from_xyz_quaternion(10, 10, 0, 0, 0, 0, 1, world.root),
        distance_to_obstacle=0.3,
    )
    assert np.sum(occupancy_map.map) == 137641


def test_gaussian_costmap(immutable_model_world):

    world, robot_view, context = immutable_model_world
    gaussian_map = GaussianCostmap(
        resolution=0.02,
        origin=Pose.from_xyz_quaternion(3.1, 2.2, 0, 0, 0, 1, 0, world.root),
        mean=400,
        sigma=150,  # Change back
        world=world,
    )

    # Checks that 5% of the size around the middle is cut out
    assert np.sum(gaussian_map.map == 0) == (400 * 0.05 * 2) ** 2


def test_sample_reachability(immutable_model_world):
    world, robot_view, context = immutable_model_world
    occupancy_map = OccupancyCostmap(
        resolution=0.02,
        height=400,
        width=400,
        world=world,
        robot_view=robot_view,
        origin=Pose.from_xyz_quaternion(3.0, 2.2, 0, 0, 0, 1, 0, world.root),
        distance_to_obstacle=0.3,
    )

    gaussian_map = GaussianCostmap(
        resolution=0.02,
        origin=Pose.from_xyz_quaternion(3.0, 2.2, 0, 0, 0, 1, 0, world.root),
        mean=400,
        sigma=15,  # Change back
        world=world,
    )

    reach_map = occupancy_map + gaussian_map

    assert np.sum(reach_map.map[:200, :]) < 5

    for pose in reach_map:
        assert pose.to_position().x > 3


# ----- Sampling test ---------------


def test_position_generation(immutable_model_world):
    world, robot_view, context = immutable_model_world
    np_map = np.zeros((200, 200))
    np_map[90:110, 90:110] = 1
    gaussian_map = GaussianCostmap(
        resolution=0.02,
        origin=Pose.from_xyz_quaternion(1, 1, 0, 0, 0, 0, 1, world.root),
        mean=200,
        sigma=15,  # Change back
        world=world,
    )
    gaussian_map.map = np_map

    for pose in gaussian_map:
        assert 0.8 <= pose.to_position().x <= 1.2
        assert 0.8 <= pose.to_position().y <= 1.2


def test_segment_map(immutable_model_world):
    world, robot_view, context = immutable_model_world
    np_map = np.zeros((200, 200))
    np_map[90:110, 90:110] = 1
    np_map[20:40, 20:40] = 1
    gaussian_map = GaussianCostmap(
        resolution=0.02,
        origin=Pose.from_xyz_quaternion(1, 1, 0, 0, 0, 0, 1, world.root),
        mean=200,
        sigma=15,
        world=world,
    )
    gaussian_map.map = np_map

    seg_maps = gaussian_map.segment_map()

    assert len(seg_maps) == 2
    map_2 = seg_maps[0] if np.sum(seg_maps[0][20:40, 20:40]) > 1 else seg_maps[1]
    map_1 = seg_maps[1] if np.sum(seg_maps[1][90:110, 90:110]) > 1 else seg_maps[0]

    assert np.sum(map_2[20:40, 20:40]) == 20**2 and np.sum(map_2[90:110, 90:110]) == 0
    assert np.sum(map_1[90:110, 90:110]) == 20**2 and np.sum(map_1[20:40, 20:40]) == 0


def test_orientation_generation(immutable_model_world):
    world, robot_view, context = immutable_model_world

    orientation = OrientationGenerator.generate_origin_orientation(
        Point3(0, 1, 0),
        origin=Pose.from_xyz_quaternion(0, 0, 0, 0, 0, 0, 1, world.root),
    )

    assert orientation.to_list() == pytest.approx([0, 0, -0.707, 0.707], abs=0.001)

    orientation = OrientationGenerator.generate_origin_orientation(
        Point3(0, -1, 0),
        origin=Pose.from_xyz_quaternion(0, 0, 0, 0, 0, 0, 1, world.root),
    )

    assert orientation.to_list() == pytest.approx([0, 0, 0.707, 0.707], abs=0.001)


def test_sample_x_axis(immutable_model_world):
    world, robot_view, context = immutable_model_world
    np_map = np.zeros((200, 200))
    np_map[:, 99:101] = 1

    gaussian_map = GaussianCostmap(
        resolution=0.02,
        origin=Pose.from_xyz_quaternion(0, 0, 0, 0, 0, 0, 1, world.root),
        mean=200,
        sigma=15,
        world=world,
    )

    gaussian_map.map = np_map

    for pose in gaussian_map:
        assert -0.05 < pose.to_position().y < 0.05


def test_sample_x_axis_offset(immutable_model_world):
    world, robot_view, context = immutable_model_world
    np_map = np.zeros((200, 200))
    np_map[120:140, 90:110] = 1

    gaussian_map = GaussianCostmap(
        resolution=0.02,
        origin=Pose.from_xyz_quaternion(0, 0, 0, 0, 0, 0, 1, world.root),
        mean=200,
        sigma=15,
        world=world,
    )

    gaussian_map.map = np_map

    for pose in gaussian_map:
        assert -0.2 <= pose.to_position().y <= 0.2
        assert 0.4 <= pose.to_position().x <= 0.8


def test_sample_x_axis_offset_non_id(immutable_model_world):
    world, robot_view, context = immutable_model_world
    np_map = np.zeros((200, 200))
    np_map[120:140, 90:110] = 1

    gaussian_map = GaussianCostmap(
        resolution=0.02,
        origin=Pose.from_xyz_quaternion(3, 2, 0, 0, 0, 0, 1, world.root),
        mean=200,
        sigma=15,
        world=world,
    )
    gaussian_map.map = np_map

    tolerance = 0.01
    for pose in gaussian_map:
        assert 1.8 <= pose.to_position().y <= 2.2 + tolerance
        assert 3.4 <= pose.to_position().x <= 3.8 + tolerance


def test_sample_to_pose_gau(immutable_model_world):
    world, robot_view, context = immutable_model_world
    np_map = np.zeros((200, 200))
    np_map[120:140, 90:110] = 1
    gaussian_map = GaussianCostmap(
        resolution=0.02,
        origin=Pose.from_xyz_quaternion(3, 2, 0, 0, 0, 0, 1, world.root),
        mean=200,
        sigma=15,
        world=world,
    )
    gaussian_map.map = np_map

    gaussian_map2 = GaussianCostmap(
        resolution=0.02,
        origin=Pose.from_xyz_quaternion(3, 2, 0, 0, 0, 0, 1, world.root),
        mean=200,
        sigma=15,
        world=world,
    )

    final_map = gaussian_map + gaussian_map2

    for pose in final_map:
        assert -1.8 < pose.to_position().y < 2.2
        assert 2.6 <= pose.to_position().x <= 3.6


def test_sample_y_axis(immutable_model_world):
    world, robot_view, context = immutable_model_world
    np_map = np.zeros((200, 200))
    np_map[99:101, :] = 1
    gaussian_map = GaussianCostmap(
        resolution=0.02,
        origin=Pose.from_xyz_quaternion(0, 0, 0, 0, 0, 0, 1, world.root),
        mean=200,
        sigma=15,
        world=world,
    )

    gaussian_map.map = np_map
    for pose in gaussian_map:
        assert -0.05 < pose.to_position().x < 0.05


def test_sample_rotated(immutable_model_world):
    world, robot_view, context = immutable_model_world
    np_map = np.zeros((200, 200))
    np_map[120:121, 99:101] = 1
    gaussian_map = GaussianCostmap(
        resolution=0.02,
        origin=Pose.from_xyz_quaternion(0, 0, 0, 0, 0, 0, 1, world.root),
        mean=200,
        sigma=15,
        world=world,
    )
    gaussian_map.map = np_map
    assert len(list(gaussian_map)) == 2

    for pose in gaussian_map:
        assert -0.05 < pose.to_position().y < 0.05
        assert 0.4 <= pose.to_position().x <= 0.45

    gaussian_map.origin = Pose.from_xyz_quaternion(0, 0, 0, 0, 0, 1, 1, world.root)

    assert len(list(gaussian_map)) == 2

    for pose in gaussian_map:
        assert -0.05 < pose.to_position().y < 0.05
        assert 0.4 <= pose.to_position().x <= 0.45


def test_sample_to_pose(immutable_model_world):
    world, robot_view, context = immutable_model_world

    np_map = np.zeros((200, 200))
    np_map[130, 160] = 1
    gaussian_map = GaussianCostmap(
        resolution=0.02,
        origin=Pose.from_xyz_quaternion(1, 1, 0, 0, 0, 0, 1, world.root),
        mean=200,
        sigma=15,
        world=world,
    )

    gaussian_map.map = np_map

    pose = list(gaussian_map)[0]

    assert pose.to_position().x == 1.6
    assert pose.to_position().y == 2.2
    assert pose.to_position().z == 0


def test_sample_highest_first(immutable_model_world):
    world, robot_view, context = immutable_model_world
    np_map = np.zeros((200, 200))
    np_map[40, 40] = 1
    np_map[80, 80] = 2
    np_map[120, 120] = 3
    gaussian_map = GaussianCostmap(
        resolution=0.02,
        origin=Pose.from_xyz_quaternion(0, 0, 0, 0, 0, 0, 1, world.root),
        mean=200,
        sigma=15,
        world=world,
    )

    gaussian_map.map = np_map

    poses = list(gaussian_map)

    assert len(poses) == 3

    assert (
        poses[2].to_position().x < poses[1].to_position().x < poses[0].to_position().x
    )
    assert (
        poses[2].to_position().y < poses[1].to_position().y < poses[0].to_position().y
    )


def test_segment_highest_first(immutable_model_world):
    world, robot_view, context = immutable_model_world
    np_map = np.zeros((200, 200))
    np_map[40:45, 40:45] = 1
    np_map[80:85, 80:85] = 3
    np_map[120:125, 120:125] = 2
    gaussian_map = GaussianCostmap(
        resolution=0.02,
        origin=Pose.from_xyz_quaternion(0, 0, 0, 0, 0, 0, 1, world.root),
        mean=200,
        sigma=15,
        world=world,
    )
    gaussian_map.map = np_map

    segmented_maps = gaussian_map.segment_map()

    assert len(segmented_maps) == 3
    assert np.max(segmented_maps[0]) == 3
    assert np.max(segmented_maps[1]) == 2
    assert np.max(segmented_maps[2]) == 1


def test_segment_empty_map(immutable_model_world):
    world, robot_view, context = immutable_model_world
    np_map = np.zeros((200, 200))
    gaussian_map = GaussianCostmap(
        resolution=0.02,
        origin=Pose(reference_frame=world.root),
        mean=200,
        sigma=15,
        world=world,
    )
    gaussian_map.map = np_map

    segmented_maps = gaussian_map.segment_map()

    assert len(segmented_maps) == 1
    assert np.sum(segmented_maps[0]) == 0


def test_orientation_generator_by_axis_y(immutable_model_world):
    world, robot_view, context = immutable_model_world

    ori_gen = OrientationGenerator.orientation_generator_for_axis(
        Vector3.from_iterable([0, 1, 0])
    )

    origin_pose = Pose(reference_frame=world.root)
    target_position = Point3.from_iterable([1, 0, 0])

    generated_orientation = ori_gen(target_position, origin_pose)

    assert generated_orientation.to_list() == pytest.approx(
        [0, 0, 0.7071, 0.7071], abs=0.001
    )


def test_orientation_generator_by_axis_minus_y(immutable_model_world):
    world, robot_view, context = immutable_model_world

    ori_gen = OrientationGenerator.orientation_generator_for_axis(
        Vector3.from_iterable([0, -1, 0])
    )

    origin_pose = Pose(reference_frame=world.root)
    target_position = Point3.from_iterable([1, 0, 0])

    generated_orientation = ori_gen(target_position, origin_pose)

    assert generated_orientation.to_list() == pytest.approx(
        [0, 0, -0.7071, 0.7071], abs=0.001
    )


def test_orientation_generator_by_axis_x(immutable_model_world):
    world, robot_view, context = immutable_model_world

    ori_gen = OrientationGenerator.orientation_generator_for_axis(
        Vector3.from_iterable([1, 0, 0])
    )

    origin_pose = Pose(reference_frame=world.root)
    target_position = Point3.from_iterable([1, 0, 0])

    generated_orientation = ori_gen(target_position, origin_pose)

    assert generated_orientation.to_list() == pytest.approx([0, 0, 1, 0], abs=0.001)
