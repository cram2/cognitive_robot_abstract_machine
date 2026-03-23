from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pycram.datastructures.enums import Arms

from llmr.pipeline.action_pipeline import _is_robot_link, _serialise_robot_state
from llmr.pipeline.entity_grounder import _camel_to_tokens
from llmr.planning.motion_precondition_planner import ExecutionState
from llmr.workflows._utils import _pose_to_xyz


class TestCamelToTokens:
    def test_two_hump(self):
        assert _camel_to_tokens("DrinkingContainer") == "drinking container"

    def test_single_word(self):
        assert _camel_to_tokens("Milk") == "milk"

    def test_already_lower(self):
        assert _camel_to_tokens("already") == "already"

    def test_three_humps(self):
        assert _camel_to_tokens("PickUpAction") == "pick up action"

    def test_empty_string(self):
        assert _camel_to_tokens("") == ""


class TestPoseToXyz:
    def test_success(self):
        pose = MagicMock()
        pose.to_position.return_value.x = 1.0
        pose.to_position.return_value.y = 2.5
        pose.to_position.return_value.z = 0.8
        result = _pose_to_xyz(pose)
        assert result == pytest.approx((1.0, 2.5, 0.8))

    def test_exception_returns_none(self):
        pose = MagicMock()
        pose.to_position.side_effect = RuntimeError("no pose")
        assert _pose_to_xyz(pose) is None

    def test_none_input_returns_none(self):
        assert _pose_to_xyz(None) is None

    def test_zero_coords(self):
        pose = MagicMock()
        pose.to_position.return_value.x = 0.0
        pose.to_position.return_value.y = 0.0
        pose.to_position.return_value.z = 0.0
        assert _pose_to_xyz(pose) == pytest.approx((0.0, 0.0, 0.0))


class TestIsRobotLink:
    def test_link_suffix(self):
        assert _is_robot_link("r_forearm_link") is True

    def test_frame_suffix(self):
        assert _is_robot_link("base_frame") is True

    def test_joint_suffix(self):
        assert _is_robot_link("shoulder_joint") is True

    def test_finger_suffix(self):
        assert _is_robot_link("r_gripper_finger") is True

    def test_pad_suffix(self):
        assert _is_robot_link("contact_pad") is True

    def test_motor_suffix(self):
        assert _is_robot_link("elbow_motor") is True

    def test_scene_object(self):
        assert _is_robot_link("milk") is False

    def test_counter_table(self):
        assert _is_robot_link("kitchen_counter") is False

    def test_cup(self):
        assert _is_robot_link("cup") is False


class TestSerialiseRobotState:
    def test_empty_state_shows_both_arms_empty(self):
        state = ExecutionState()
        result = _serialise_robot_state(state)
        assert "Both arms: empty" in result

    def test_left_arm_holding_shows_name(self):
        state = ExecutionState()
        body = MagicMock()
        body.name.name = "milk"
        state.held_objects[Arms.LEFT] = body
        result = _serialise_robot_state(state)
        assert "LEFT" in result
        assert "milk" in result

    def test_right_arm_shows_empty_when_left_held(self):
        state = ExecutionState()
        body = MagicMock()
        body.name.name = "milk"
        state.held_objects[Arms.LEFT] = body
        result = _serialise_robot_state(state)
        # RIGHT arm should appear as empty
        assert "RIGHT" in result
        assert "empty" in result

    def test_both_arms_holding(self):
        state = ExecutionState()
        milk = MagicMock()
        milk.name.name = "milk"
        cup = MagicMock()
        cup.name.name = "cup"
        state.held_objects[Arms.LEFT] = milk
        state.held_objects[Arms.RIGHT] = cup
        result = _serialise_robot_state(state)
        assert "milk" in result
        assert "cup" in result

    def test_fallback_to_last_pickup_arm(self):
        """When held_objects is empty, fall back to last_pickup_arm."""
        state = ExecutionState()
        body = MagicMock()
        body.name.name = "bottle"
        state.last_pickup_arm = Arms.RIGHT
        state.last_pickup_body = body
        result = _serialise_robot_state(state)
        assert "RIGHT" in result
        assert "bottle" in result
        assert "LEFT" in result

    def test_fallback_no_body_says_unknown(self):
        state = ExecutionState()
        state.last_pickup_arm = Arms.LEFT
        state.last_pickup_body = None
        result = _serialise_robot_state(state)
        assert "unknown object" in result
