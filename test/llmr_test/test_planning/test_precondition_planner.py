from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pycram.datastructures.enums import Arms

from llmr.planning.motion_precondition_planner import (
    ExecutionState,
    MotionPreconditionPlanner,
    PickUpPreconditionProvider,
    PlacePreconditionProvider,
    PreconditionResult,
)


# ── ExecutionState ────────────────────────────────────────────────────────────


class TestExecutionStateCopy:
    def test_copy_is_independent(self):
        state = ExecutionState()
        body = MagicMock()
        state.held_objects[Arms.LEFT] = body
        state.last_pickup_arm = Arms.LEFT
        state.last_pickup_body = body

        copy = state.copy()
        # Modify copy — should not affect original
        copy.held_objects[Arms.RIGHT] = MagicMock()
        copy.last_pickup_arm = Arms.RIGHT

        assert state.last_pickup_arm is Arms.LEFT
        assert Arms.RIGHT not in state.held_objects

    def test_copy_preserves_values(self):
        state = ExecutionState()
        body = MagicMock()
        state.last_pickup_arm = Arms.RIGHT
        state.last_pickup_body = body
        state.held_objects[Arms.RIGHT] = body

        copy = state.copy()
        assert copy.last_pickup_arm is Arms.RIGHT
        assert copy.last_pickup_body is body
        assert copy.held_objects[Arms.RIGHT] is body

    def test_empty_state_copy(self):
        state = ExecutionState()
        copy = state.copy()
        assert copy.last_pickup_arm is None
        assert copy.last_pickup_body is None
        assert copy.held_objects == {}


# ── PickUpPreconditionProvider._free_arms ────────────────────────────────────


class TestFreeArms:
    def test_empty_held_objects_returns_both(self):
        state = ExecutionState()
        assert PickUpPreconditionProvider._free_arms(state) is Arms.BOTH

    def test_both_none_values_returns_both(self):
        state = ExecutionState()
        state.held_objects[Arms.LEFT] = None
        state.held_objects[Arms.RIGHT] = None
        assert PickUpPreconditionProvider._free_arms(state) is Arms.BOTH

    def test_left_held_returns_right(self):
        state = ExecutionState()
        state.held_objects[Arms.LEFT] = MagicMock()
        assert PickUpPreconditionProvider._free_arms(state) is Arms.RIGHT

    def test_right_held_returns_left(self):
        state = ExecutionState()
        state.held_objects[Arms.RIGHT] = MagicMock()
        assert PickUpPreconditionProvider._free_arms(state) is Arms.LEFT

    def test_both_held_returns_none(self):
        state = ExecutionState()
        state.held_objects[Arms.LEFT] = MagicMock()
        state.held_objects[Arms.RIGHT] = MagicMock()
        assert PickUpPreconditionProvider._free_arms(state) is None


# ── PickUpPreconditionProvider._ensure_free_arm ──────────────────────────────


class TestEnsureFreeArm:
    def test_free_arm_unchanged(self):
        state = ExecutionState()  # no held objects
        action = MagicMock()
        action.arm = Arms.RIGHT
        result = PickUpPreconditionProvider._ensure_free_arm(action, state)
        assert result is action

    def test_none_arm_unchanged(self):
        state = ExecutionState()
        action = MagicMock()
        action.arm = None
        result = PickUpPreconditionProvider._ensure_free_arm(action, state)
        assert result is action

    def test_occupied_arm_switches_to_free(self):
        state = ExecutionState()
        body = MagicMock()
        body.name = MagicMock()
        body.name.name = "milk"
        state.held_objects[Arms.RIGHT] = body

        action = MagicMock()
        action.arm = Arms.RIGHT
        action.object_designator = MagicMock()
        action.grasp_description = MagicMock()

        with patch(
            "llmr.planning.motion_precondition_planner.PickUpAction"
        ) as mock_pickup:
            mock_new_action = MagicMock()
            mock_new_action.arm = Arms.LEFT
            mock_pickup.return_value = mock_new_action

            result = PickUpPreconditionProvider._ensure_free_arm(action, state)

        mock_pickup.assert_called_once()
        assert result.arm is Arms.LEFT

    def test_both_arms_occupied_returns_original(self):
        state = ExecutionState()
        state.held_objects[Arms.LEFT] = MagicMock()
        state.held_objects[Arms.RIGHT] = MagicMock()
        action = MagicMock()
        action.arm = Arms.RIGHT
        result = PickUpPreconditionProvider._ensure_free_arm(action, state)
        assert result is action


# ── PickUpPreconditionProvider.update_state ──────────────────────────────────


class TestPickUpUpdateState:
    def test_updates_held_objects_and_tracking(self, mock_world):
        provider = PickUpPreconditionProvider(world=mock_world)
        state = ExecutionState()
        body = MagicMock()
        action = MagicMock()
        action.arm = Arms.LEFT
        action.object_designator = body

        provider.update_state(action, state)

        assert state.held_objects[Arms.LEFT] is body
        assert state.last_pickup_arm is Arms.LEFT
        assert state.last_pickup_body is body


# ── PlacePreconditionProvider._find_holding_arm ──────────────────────────────


class TestFindHoldingArm:
    def test_left_arm_match(self):
        state = ExecutionState()
        body = MagicMock()
        state.held_objects[Arms.LEFT] = body
        result = PlacePreconditionProvider._find_holding_arm(body, state)
        assert result is Arms.LEFT

    def test_right_arm_match(self):
        state = ExecutionState()
        body = MagicMock()
        state.held_objects[Arms.RIGHT] = body
        result = PlacePreconditionProvider._find_holding_arm(body, state)
        assert result is Arms.RIGHT

    def test_no_match_falls_back_to_last_pickup_arm(self):
        state = ExecutionState()
        state.last_pickup_arm = Arms.RIGHT
        body = MagicMock()
        result = PlacePreconditionProvider._find_holding_arm(body, state)
        assert result is Arms.RIGHT

    def test_none_body_falls_back_to_last_pickup_arm(self):
        state = ExecutionState()
        state.last_pickup_arm = Arms.LEFT
        result = PlacePreconditionProvider._find_holding_arm(None, state)
        assert result is Arms.LEFT

    def test_no_match_no_fallback_returns_none(self):
        state = ExecutionState()
        body = MagicMock()
        result = PlacePreconditionProvider._find_holding_arm(body, state)
        assert result is None


# ── PlacePreconditionProvider.update_state ───────────────────────────────────


class TestPlaceUpdateState:
    def test_clears_held_object_for_arm(self, mock_world):
        provider = PlacePreconditionProvider(world=mock_world)
        state = ExecutionState()
        body = MagicMock()
        state.held_objects[Arms.LEFT] = body
        state.last_pickup_arm = Arms.LEFT
        state.last_pickup_body = body

        action = MagicMock()
        action.arm = Arms.LEFT
        action.object_designator = body

        provider.update_state(action, state)

        assert state.held_objects[Arms.LEFT] is None
        assert state.last_pickup_arm is None
        assert state.last_pickup_body is None

    def test_does_not_clear_last_pickup_for_different_object(self, mock_world):
        provider = PlacePreconditionProvider(world=mock_world)
        state = ExecutionState()
        original_body = MagicMock()
        placed_body = MagicMock()
        state.held_objects[Arms.RIGHT] = placed_body
        state.last_pickup_arm = Arms.LEFT
        state.last_pickup_body = original_body

        action = MagicMock()
        action.arm = Arms.RIGHT
        action.object_designator = placed_body

        provider.update_state(action, state)

        # last_pickup tracking should not be cleared — different object
        assert state.last_pickup_arm is Arms.LEFT
        assert state.last_pickup_body is original_body


# ── MotionPreconditionPlanner ─────────────────────────────────────────────────


class TestMotionPreconditionPlanner:
    def test_compute_unregistered_type_returns_empty_preconditions(self, mock_world):
        planner = MotionPreconditionPlanner(mock_world)
        state = ExecutionState()

        class _FakeAction:
            pass

        action = _FakeAction()
        result = planner.compute(action, state)
        assert result.preconditions == []
        assert result.action is action

    def test_update_state_unregistered_type_is_noop(self, mock_world):
        planner = MotionPreconditionPlanner(mock_world)
        state = ExecutionState()

        class _FakeAction:
            pass

        # Should not raise
        planner.update_state(_FakeAction(), state)
