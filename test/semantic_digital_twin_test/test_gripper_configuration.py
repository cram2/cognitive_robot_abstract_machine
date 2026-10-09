from __future__ import annotations

import numpy as np
import pytest
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.robots.gripper_configuration import (
    MAXIMUM_OPENING_WIDTH_MM,
    GriplinkFlexConfiguration,
    GripperStateConfiguration,
)
from semantic_digital_twin.exceptions import ConnectionsOutsideEndEffector
from semantic_digital_twin.robots.daisy import DAiSy

# %% GripperStateConfiguration


def test_closed_configuration_carries_the_end_effectors_own_joint_state(daisy_world):
    """
    ``GripperStateConfiguration.closed(gripper).joint_state`` is the very object
    ``gripper.get_joint_state_by_type(GripperState.CLOSE)`` returns -- not a copy.
    """
    daisy = daisy_world.get_semantic_annotations_by_type(DAiSy)[0]

    for end_effector in daisy.all_end_effectors:
        expected = end_effector.get_joint_state_by_type(GripperState.CLOSE)
        configuration = GripperStateConfiguration.closed(end_effector)
        assert configuration.joint_state is expected


def test_configuration_rejects_foreign_connections(daisy_world):
    """
    A configuration built with a joint state from the other arm's gripper raises
    ``ConnectionsOutsideEndEffector``.
    """
    daisy = daisy_world.get_semantic_annotations_by_type(DAiSy)[0]
    left_gripper = daisy.left_arm.end_effector
    right_gripper = daisy.right_arm.end_effector

    right_close_state = right_gripper.get_joint_state_by_type(GripperState.CLOSE)
    with pytest.raises(ConnectionsOutsideEndEffector):
        GripperStateConfiguration(
            end_effector=left_gripper,
            joint_state=right_close_state,
        )


def test_configuration_end_effector_and_joint_state_are_carried(daisy_world):
    """
    The configuration carries the end effector and the joint state it was built from.
    """
    daisy = daisy_world.get_semantic_annotations_by_type(DAiSy)[0]
    end_effector = daisy.left_arm.end_effector
    close_state = end_effector.get_joint_state_by_type(GripperState.CLOSE)

    configuration = GripperStateConfiguration.closed(end_effector)
    assert configuration.end_effector is end_effector
    assert configuration.joint_state is close_state


# %% GriplinkFlexConfiguration


def test_flex_configuration_interpolates_between_the_declared_states(daisy_world):
    """
    A flex configuration commands, for every connection of the open state, the point
    between the declared open and close states that the configured opening width
    corresponds to.
    """
    daisy = daisy_world.get_semantic_annotations_by_type(DAiSy)[0]
    end_effector = daisy.left_arm.end_effector
    open_state = end_effector.get_joint_state_by_type(GripperState.OPEN)
    close_state = end_effector.get_joint_state_by_type(GripperState.CLOSE)

    configuration = GriplinkFlexConfiguration.from_state_type(
        end_effector, GripperState.FLEXCLOSE, grip_position=60
    )

    fraction = (MAXIMUM_OPENING_WIDTH_MM - 60) / MAXIMUM_OPENING_WIDTH_MM
    close_targets = dict(close_state.items())
    expected = [
        open_target + fraction * (close_targets[connection] - open_target)
        for connection, open_target in open_state.items()
    ]
    assert configuration.joint_state.target_values == pytest.approx(expected)


def test_flex_configuration_without_grip_position_commands_the_open_state(daisy_world):
    """
    Without a configured opening width, a flex configuration commands the declared
    open state, i.e. the fully opened state.
    """
    daisy = daisy_world.get_semantic_annotations_by_type(DAiSy)[0]
    end_effector = daisy.left_arm.end_effector
    open_state = end_effector.get_joint_state_by_type(GripperState.OPEN)

    configuration = GriplinkFlexConfiguration.from_state_type(
        end_effector, GripperState.FLEXOPEN
    )

    expected = [target for _, target in open_state.items()]
    assert configuration.joint_state.target_values == pytest.approx(expected)


def test_flexclose_configuration_without_grip_position_commands_the_fully_closed_state(
    daisy_world,
):
    """
    Without a configured opening width, a ``FLEXCLOSE`` configuration commands the
    declared close state, i.e. the state where the fingers touch.

    The joint's upper position limit is not the closed state: beyond the close state
    the fingers cross and separate again, so anchoring the interpolation on the limit
    makes a flex close visually open the gripper.
    """
    daisy = daisy_world.get_semantic_annotations_by_type(DAiSy)[0]
    end_effector = daisy.left_arm.end_effector
    close_state = end_effector.get_joint_state_by_type(GripperState.CLOSE)

    configuration = GriplinkFlexConfiguration.from_state_type(
        end_effector, GripperState.FLEXCLOSE
    )

    expected = [target for _, target in close_state.items()]
    assert configuration.joint_state.target_values == pytest.approx(expected)


def test_flexclose_target_closes_the_fingers(daisy_world):
    """
    Commanding the flex close configuration's target brings the fingers to the same
    separation as the declared close state, i.e. the gripper closes instead of
    stopping short or moving past the closed state, where the fingers cross and
    separate again.
    """
    daisy = daisy_world.get_semantic_annotations_by_type(DAiSy)[0]
    end_effector = daisy.left_arm.end_effector
    open_configuration = GriplinkFlexConfiguration.from_state_type(
        end_effector, GripperState.FLEXOPEN
    )
    close_configuration = GriplinkFlexConfiguration.from_state_type(
        end_effector, GripperState.FLEXCLOSE
    )
    close_state = end_effector.get_joint_state_by_type(GripperState.CLOSE)

    def tip_separation(joint_state):
        for connection, target in joint_state.items():
            connection.position = target
        left_tip = end_effector.fingers[0].tip.global_pose.position.to_np()[:3]
        right_tip = end_effector.fingers[1].tip.global_pose.position.to_np()[:3]
        return float(np.linalg.norm(left_tip - right_tip))

    open_separation = tip_separation(open_configuration.joint_state)
    closed_separation = tip_separation(close_state)
    flexclose_separation = tip_separation(close_configuration.joint_state)
    assert flexclose_separation == pytest.approx(closed_separation)
    assert flexclose_separation < open_separation
