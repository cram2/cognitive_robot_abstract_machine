"""
Tests for the grasp geometry :mod:`experiments.causal_reasoning.tracy_clutter_picking.tr
acy_mujoco_addons.pick_and_place_action` derives from a mounted Tracy.
"""

from __future__ import annotations

import math

import numpy
import pytest

from coraplex.datastructures.enums import Arms
from experiments.causal_reasoning.tracy_clutter_picking.tracy_mujoco_addons.equipment import (
    mount_stationary_robot,
    parse_tracy,
    tracy_table_mount_position,
)
from experiments.causal_reasoning.tracy_clutter_picking.tracy_mujoco_addons.pick_and_place_action import (
    TopDownGraspGeometry,
    bounding_box_center_world,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.utils import tracy_installed
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

pytestmark = pytest.mark.skipif(
    not tracy_installed(), reason="iai_tracy_description is not installed"
)


@pytest.fixture
def mounted_tracy() -> tuple[World, Tracy]:
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(
            Body(name=PrefixedName(name="root", prefix="world"))
        )
    tracy_world = parse_tracy()
    mount_position, _ = tracy_table_mount_position(tracy_world, x=0.0, y=0.0)
    robot = mount_stationary_robot(world, Tracy, tracy_world, mount_position)
    return world, robot


def test_bounding_box_center_world_is_the_average_of_the_bodys_own_min_and_max(
    mounted_tracy,
):
    world, _ = mounted_tracy
    body = world.get_body_by_name("left_robotiq_85_left_finger_tip_link")

    center = bounding_box_center_world(world, body)

    bounding_box = body.collision[0].local_frame_bounding_box
    root_transform_body = world.compute_forward_kinematics_np(world.root, body)
    expected_local_center = [
        (bounding_box.min_x + bounding_box.max_x) / 2,
        (bounding_box.min_y + bounding_box.max_y) / 2,
        (bounding_box.min_z + bounding_box.max_z) / 2,
    ]
    expected = (
        root_transform_body[:3, :3] @ expected_local_center + root_transform_body[:3, 3]
    )
    assert list(center) == list(expected)


def test_finger_midpoint_offset_differs_between_left_and_right_arm(mounted_tracy):
    world, robot = mounted_tracy

    left_offset = TopDownGraspGeometry(world, robot, Arms.LEFT).finger_midpoint_offset()
    right_offset = TopDownGraspGeometry(
        world, robot, Arms.RIGHT
    ).finger_midpoint_offset()

    assert list(left_offset) != list(right_offset)


def test_grasp_yaw_turns_the_closing_axis_about_the_vertical_without_tilting_it(
    mounted_tracy,
):
    world, robot = mounted_tracy
    yaw = math.pi / 6

    straight = TopDownGraspGeometry(world, robot, Arms.LEFT).tool_frame_pose(
        0.0, 0.0, 1.0
    )
    turned = TopDownGraspGeometry(
        world, robot, Arms.LEFT, grasp_yaw=yaw
    ).tool_frame_pose(0.0, 0.0, 1.0)

    straight_rotation = straight.to_rotation_matrix().evaluate()[:3, :3]
    turned_rotation = turned.to_rotation_matrix().evaluate()[:3, :3]
    about_vertical = numpy.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    numpy.testing.assert_allclose(
        turned_rotation, about_vertical @ straight_rotation, atol=1e-9
    )
    numpy.testing.assert_allclose(turned_rotation[:, 2], [0.0, 0.0, -1.0], atol=1e-9)
