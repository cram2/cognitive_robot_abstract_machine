"""
The connections the parallel gripper command servers of real grippers command.
"""

from __future__ import annotations

from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.robots.tracy import Tracy, TracyJoint
from semantic_digital_twin.world import World

# %% the connection each server commands


def test_the_servers_of_tracys_grippers_command_the_knuckle_joint_of_each_gripper(
    tracy_world: World,
) -> None:
    tracy = tracy_world.get_semantic_annotations_by_type(Tracy)[0]

    assert [
        arm.end_effector.commanded_connection.name.name
        for arm in (tracy.left_arm, tracy.right_arm)
    ] == [TracyJoint.LEFT_GRIPPER_LEFT_KNUCKLE, TracyJoint.RIGHT_GRIPPER_LEFT_KNUCKLE]


def test_the_servers_of_tiagos_grippers_command_the_finger_joint_of_each_gripper(
    _tiago_world_setup: World,
) -> None:
    tiago = _tiago_world_setup.get_semantic_annotations_by_type(Tiago)[0]

    assert [
        arm.end_effector.commanded_connection.name.name for arm in tiago.all_arms
    ] == [arm.end_effector.finger_joint for arm in tiago.all_arms]
