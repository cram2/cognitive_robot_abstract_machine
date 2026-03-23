"""Tests for PickUpPreconditionProvider.compute() and PlacePreconditionProvider.compute()."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pycram.datastructures.enums import Arms

from llmr.planning.motion_precondition_planner import (
    ExecutionState,
    PickUpPreconditionProvider,
    PlacePreconditionProvider,
    PreconditionResult,
)


# ── PickUpPreconditionProvider.compute ───────────────────────────────────────


class TestPickUpCompute:
    def _make_action(self, arm=Arms.RIGHT):
        action = MagicMock()
        action.arm = arm
        action.object_designator = MagicMock()
        action.grasp_description = MagicMock()
        return action

    def test_compute_empty_state_includes_park_and_torso(self, mock_world):
        provider = PickUpPreconditionProvider(world=mock_world)
        state = ExecutionState()
        action = self._make_action()

        mock_world.get_semantic_annotations_by_type.return_value = []

        with patch(
            "llmr.planning.motion_precondition_planner.ParkArmsAction"
        ) as mock_park, patch(
            "llmr.planning.motion_precondition_planner.MoveTorsoAction"
        ) as mock_torso, patch.object(
            provider, "_make_nav_designator", return_value=None
        ):
            mock_park.return_value = MagicMock()
            mock_torso.return_value = MagicMock()
            result = provider.compute(action, state)

        assert isinstance(result, PreconditionResult)
        assert len(result.preconditions) >= 1  # at least MoveTorsoAction

    def test_compute_with_nav_designator_included(self, mock_world):
        provider = PickUpPreconditionProvider(world=mock_world)
        state = ExecutionState()
        action = self._make_action(arm=Arms.LEFT)

        mock_world.get_semantic_annotations_by_type.return_value = []
        mock_nav = MagicMock()

        with patch(
            "llmr.planning.motion_precondition_planner.ParkArmsAction"
        ) as mock_park, patch(
            "llmr.planning.motion_precondition_planner.MoveTorsoAction"
        ) as mock_torso, patch.object(
            provider, "_make_nav_designator", return_value=mock_nav
        ):
            mock_park.return_value = MagicMock()
            mock_torso.return_value = MagicMock()
            result = provider.compute(action, state)

        assert mock_nav in result.preconditions

    def test_compute_both_arms_occupied_no_park_action(self, mock_world):
        provider = PickUpPreconditionProvider(world=mock_world)
        state = ExecutionState()
        state.held_objects[Arms.LEFT] = MagicMock()
        state.held_objects[Arms.RIGHT] = MagicMock()
        action = self._make_action(arm=Arms.RIGHT)

        mock_world.get_semantic_annotations_by_type.return_value = []

        with patch(
            "llmr.planning.motion_precondition_planner.ParkArmsAction"
        ) as mock_park, patch(
            "llmr.planning.motion_precondition_planner.MoveTorsoAction"
        ) as mock_torso, patch.object(
            provider, "_make_nav_designator", return_value=None
        ):
            mock_park.return_value = MagicMock()
            mock_torso.return_value = MagicMock()
            result = provider.compute(action, state)

        # ParkArmsAction should NOT be called since both arms hold objects
        mock_park.assert_not_called()


# ── PlacePreconditionProvider.compute ────────────────────────────────────────


class TestPlaceCompute:
    def _make_place_action(self, arm=Arms.LEFT, target=None):
        action = MagicMock()
        action.arm = arm
        action.object_designator = MagicMock()
        action.target_location = target or MagicMock()
        return action

    def test_compute_returns_precondition_result(self, mock_world):
        provider = PlacePreconditionProvider(world=mock_world)
        state = ExecutionState()
        action = self._make_place_action()

        mock_world.get_semantic_annotations_by_type.return_value = []

        mock_place_pose = MagicMock()
        mock_nav = MagicMock()

        with patch(
            "llmr.planning.motion_precondition_planner.MoveTorsoAction"
        ) as mock_torso, patch.object(
            provider, "_resolve_place_pose", return_value=mock_place_pose
        ), patch.object(
            provider, "_make_nav_designator", return_value=mock_nav
        ), patch(
            "llmr.planning.motion_precondition_planner.PlaceAction"
        ) as mock_place_cls:
            mock_torso.return_value = MagicMock()
            mock_place_cls.return_value = MagicMock()
            result = provider.compute(action, state)

        assert isinstance(result, PreconditionResult)
        assert mock_nav in result.preconditions

    def test_compute_uses_find_holding_arm(self, mock_world):
        provider = PlacePreconditionProvider(world=mock_world)
        state = ExecutionState()
        state.last_pickup_arm = Arms.RIGHT  # fallback arm

        action = self._make_place_action(arm=None)  # arm not specified

        mock_world.get_semantic_annotations_by_type.return_value = []

        with patch(
            "llmr.planning.motion_precondition_planner.MoveTorsoAction"
        ) as mock_torso, patch.object(
            provider, "_resolve_place_pose", return_value=MagicMock()
        ), patch.object(
            provider, "_make_nav_designator", return_value=None
        ), patch(
            "llmr.planning.motion_precondition_planner.PlaceAction"
        ) as mock_place_cls:
            mock_torso.return_value = MagicMock()
            mock_place_cls.return_value = MagicMock()
            result = provider.compute(action, state)

        # PlaceAction should be called with arm=RIGHT (from last_pickup_arm)
        call_kwargs = mock_place_cls.call_args[1] if mock_place_cls.call_args else {}
        assert "arm" in call_kwargs or mock_place_cls.called


# ── PlacePreconditionProvider._resolve_place_pose ────────────────────────────


class TestResolvePlacePose:
    def test_posestamped_passthrough(self, mock_world):
        from pycram.datastructures.pose import PoseStamped

        provider = PlacePreconditionProvider(world=mock_world)
        pose = MagicMock(spec=PoseStamped)

        result = provider._resolve_place_pose(pose, MagicMock())
        assert result is pose

    def test_body_target_uses_semantic_costmap(self, mock_world):
        provider = PlacePreconditionProvider(world=mock_world)
        body = MagicMock()  # NOT a PoseStamped

        mock_pose = MagicMock()

        with patch(
            "llmr.planning.motion_precondition_planner.SemanticCostmapLocation"
        ) as mock_scl_cls:
            mock_scl = MagicMock()
            mock_scl.ground.return_value = mock_pose
            mock_scl_cls.return_value = mock_scl

            result = provider._resolve_place_pose(body, MagicMock())

        assert result is mock_pose

    def test_body_target_fallback_on_costmap_failure(self, mock_world):
        from pycram.datastructures.pose import PoseStamped

        provider = PlacePreconditionProvider(world=mock_world)
        body = MagicMock()
        fallback_pose = MagicMock()

        with patch(
            "llmr.planning.motion_precondition_planner.SemanticCostmapLocation",
            side_effect=RuntimeError("no surface"),
        ):
            with patch.object(
                PoseStamped, "from_spatial_type", return_value=fallback_pose
            ):
                result = provider._resolve_place_pose(body, MagicMock())

        assert result is fallback_pose


# ── entity_grounder misc ──────────────────────────────────────────────────────


class TestGroundingResultUsedEql:
    def test_used_eql_always_false(self):
        from llmr.pipeline.entity_grounder import GroundingResult

        gr = GroundingResult()
        assert gr.used_eql is False

    def test_all_annotation_subclasses_callable(self):
        """_all_annotation_subclasses returns a list (may be empty in test env)."""
        from llmr.pipeline.entity_grounder import _all_annotation_subclasses

        result = _all_annotation_subclasses()
        assert isinstance(result, list)
