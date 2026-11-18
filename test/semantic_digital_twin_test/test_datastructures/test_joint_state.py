import pytest

from semantic_digital_twin.datastructures.exceptions import (
    ConnectionNotInJointStateError,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.testing import (
    StateChangeCounter,
    two_arm_robot_world,
    world_setup,
)

# %% applying a joint state to a world


def test_applying_a_joint_state_announces_one_state_change(two_arm_robot_world):
    world = two_arm_robot_world
    right_joint = world.get_connection_by_name("r_joint_1")
    left_joint = world.get_connection_by_name("l_joint_1")
    joint_state = JointState.from_mapping({right_joint: 0.5, left_joint: -0.25})
    counter = StateChangeCounter(_world=world)

    joint_state.apply_to(world)

    assert counter.count == 1
    assert right_joint.position == 0.5
    assert left_joint.position == -0.25


def test_applying_a_joint_state_respects_multiplier_and_offset(world_setup):
    world, l1, l2, _, _, _ = world_setup
    prismatic_connection = world.get_connection(l1, l2)
    prismatic_connection.multiplier = 2.0
    prismatic_connection.offset = 0.5

    JointState.from_mapping({prismatic_connection: 1.5}).apply_to(world)

    assert prismatic_connection.position == 1.5
    assert world.state[prismatic_connection.raw_dof.id].position == 0.5


# %% reading the target value of one connection


def test_a_joint_state_gives_the_target_value_of_each_of_its_connections(
    two_arm_robot_world,
):
    right_joint = two_arm_robot_world.get_connection_by_name("r_joint_1")
    left_joint = two_arm_robot_world.get_connection_by_name("l_joint_1")
    joint_state = JointState.from_mapping({right_joint: 0.5, left_joint: -0.25})

    assert [
        joint_state.target_value(right_joint),
        joint_state.target_value(left_joint),
    ] == [0.5, -0.25]


def test_a_joint_state_has_no_target_value_for_a_connection_it_leaves_out(
    two_arm_robot_world,
):
    right_joint = two_arm_robot_world.get_connection_by_name("r_joint_1")
    left_joint = two_arm_robot_world.get_connection_by_name("l_joint_1")
    joint_state = JointState.from_mapping({right_joint: 0.5})

    with pytest.raises(ConnectionNotInJointStateError):
        joint_state.target_value(left_joint)
