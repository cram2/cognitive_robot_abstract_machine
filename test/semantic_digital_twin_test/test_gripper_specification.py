from __future__ import annotations

import pytest

from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.robots.gripper_specification import (
    GripperStateSpecification,
)
from semantic_digital_twin.exceptions import ConnectionsOutsideEndEffector
from semantic_digital_twin.robots.daisy import DAiSy

# %% GripperStateSpecification


def test_closed_specification_carries_the_end_effectors_own_joint_state(daisy_world):
    """
    ``GripperStateSpecification.closed(gripper).joint_state`` is the very object
    ``gripper.get_joint_state_by_type(GripperState.CLOSE)`` returns -- not a copy.
    """
    daisy = daisy_world.get_semantic_annotations_by_type(DAiSy)[0]

    for end_effector in daisy.get_end_effectors():
        expected = end_effector.get_joint_state_by_type(GripperState.CLOSE)
        specification = GripperStateSpecification.closed(end_effector)
        assert specification.joint_state is expected


def test_specification_rejects_foreign_connections(daisy_world):
    """
    A specification built with a joint state from the other arm's gripper raises
    ``ConnectionsOutsideEndEffector``.
    """
    daisy = daisy_world.get_semantic_annotations_by_type(DAiSy)[0]
    left_gripper = daisy.left_arm.end_effector
    right_gripper = daisy.right_arm.end_effector

    right_close_state = right_gripper.get_joint_state_by_type(GripperState.CLOSE)
    with pytest.raises(ConnectionsOutsideEndEffector):
        GripperStateSpecification(
            end_effector=left_gripper,
            joint_state=right_close_state,
        )


def test_specification_end_effector_and_joint_state_are_carried(daisy_world):
    """
    The specification carries the end effector and the joint state it was built from.
    """
    daisy = daisy_world.get_semantic_annotations_by_type(DAiSy)[0]
    end_effector = daisy.left_arm.end_effector
    close_state = end_effector.get_joint_state_by_type(GripperState.CLOSE)

    specification = GripperStateSpecification.closed(end_effector)
    assert specification.end_effector is end_effector
    assert specification.joint_state is close_state
