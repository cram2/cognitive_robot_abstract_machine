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
