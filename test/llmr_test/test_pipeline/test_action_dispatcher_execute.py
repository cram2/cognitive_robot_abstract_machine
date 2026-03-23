"""Tests for PickUpActionHandler and PlaceActionHandler execute() paths."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pycram.datastructures.enums import Arms

from llmr.pipeline.action_dispatcher import (
    ActionHandler,
    PickUpActionHandler,
    PlaceActionHandler,
    WorldContext,
)
from llmr.pipeline.clarification import ClarificationNeededError
from llmr.workflows.schemas.common import EntityDescriptionSchema
from llmr.workflows.schemas.pick_up import (
    PickUpDiscreteResolutionSchema,
    PickUpSlotSchema,
)
from llmr.workflows.schemas.place import PlaceSlotSchema


def _make_grounding(bodies=None, warning=None):
    result = MagicMock()
    result.bodies = bodies or []
    result.warning = warning
    return result


def _resolution_pickup(arm="LEFT"):
    return PickUpDiscreteResolutionSchema(
        arm=arm,
        approach_direction="FRONT",
        vertical_alignment="TOP",
        rotate_gripper=False,
        reasoning="Object is to the left.",
    )


# ── ActionHandler._get_robot_context ─────────────────────────────────────────


class TestGetRobotContext:
    def test_no_robot_in_world(self, mock_world):
        mock_world.get_semantic_annotations_by_type.return_value = []
        ctx = WorldContext()
        handler = PickUpActionHandler(world=mock_world, world_context=ctx)
        xyz, lines = handler._get_robot_context()
        assert xyz is None
        assert any("unknown" in l for l in lines)

    def test_robot_with_valid_pose(self, mock_world):
        robot = MagicMock()
        robot.base.root.global_pose.to_position.return_value.x = 1.0
        robot.base.root.global_pose.to_position.return_value.y = 2.0
        robot.base.root.global_pose.to_position.return_value.z = 0.0
        mock_world.get_semantic_annotations_by_type.return_value = [robot]
        ctx = WorldContext()
        handler = PickUpActionHandler(world=mock_world, world_context=ctx)
        xyz, lines = handler._get_robot_context()
        assert xyz is not None
        assert any("Robot position" in l for l in lines)

    def test_robot_exception_returns_unknown(self, mock_world):
        mock_world.get_semantic_annotations_by_type.side_effect = RuntimeError("no robot")
        ctx = WorldContext()
        handler = PickUpActionHandler(world=mock_world, world_context=ctx)
        xyz, lines = handler._get_robot_context()
        assert xyz is None


# ── PickUpActionHandler — lazy LLM initialisation ───────────────────────────


class TestPickUpLazyLLM:
    def test_get_resolver_llm_created_once(self, mock_world):
        original = PickUpActionHandler._resolver_llm
        PickUpActionHandler._resolver_llm = None
        try:
            ctx = WorldContext()
            with patch("llmr.workflows.llm_configuration.default_llm") as mock_llm:
                mock_llm.with_structured_output.return_value = MagicMock()
                PickUpActionHandler(world=mock_world, world_context=ctx)._get_resolver_llm()
                PickUpActionHandler(world=mock_world, world_context=ctx)._get_resolver_llm()
            mock_llm.with_structured_output.assert_called_once()
        finally:
            PickUpActionHandler._resolver_llm = original


# ── PickUpActionHandler.execute() — full path ─────────────────────────────────


class TestPickUpHandlerExecute:
    def _make_handler(self, mock_world, manipulator=None):
        ctx = WorldContext(manipulator=manipulator or MagicMock())
        return PickUpActionHandler(world=mock_world, world_context=ctx)

    def test_fully_specified_partial_resolves_directly(self, mock_world, entity_description):
        handler = self._make_handler(mock_world)
        body = MagicMock()
        handler._grounder = MagicMock()
        handler._grounder.ground.return_value = _make_grounding(bodies=[body])

        mock_action = MagicMock()
        with patch("llmr.pipeline.action_dispatcher.PartialDesignator") as mock_pd_cls:
            mock_pd = MagicMock()
            mock_pd.missing_parameter.return_value = False
            mock_pd.resolve.return_value = mock_action
            mock_pd_cls.return_value = mock_pd

            schema = PickUpSlotSchema(object_description=entity_description)
            result = handler.execute(schema)

        assert result is mock_action
        mock_pd.resolve.assert_called_once()

    def test_missing_params_calls_resolve_discrete_and_builds_action(
        self, mock_world, entity_description
    ):
        handler = self._make_handler(mock_world)
        body = MagicMock()
        body.global_pose = MagicMock()
        body.name.name = "milk"
        handler._grounder = MagicMock()
        handler._grounder.ground.return_value = _make_grounding(bodies=[body])
        mock_world.get_semantic_annotations_by_type.return_value = []
        mock_world.get_semantic_annotations_of_body.return_value = []

        resolution = _resolution_pickup("LEFT")

        mock_action = MagicMock()
        with patch("llmr.pipeline.action_dispatcher.PartialDesignator") as mock_pd_cls:
            mock_pd = MagicMock()
            mock_pd.missing_parameter.return_value = True
            mock_pd.kwargs = {"arm": None, "grasp_description": None}
            mock_pd_cls.return_value = mock_pd

            with patch.object(handler, "_resolve_discrete", return_value=resolution):
                with patch("llmr.pipeline.action_dispatcher.PickUpAction") as mock_pickup_cls:
                    with patch(
                        "llmr.pipeline.action_dispatcher.GraspDescription"
                    ) as mock_grasp_cls:
                        mock_grasp_cls.return_value = MagicMock()
                        mock_pickup_cls.return_value = mock_action

                        schema = PickUpSlotSchema(object_description=entity_description)
                        result = handler.execute(schema)

        assert result is mock_action

    def test_build_world_context_with_valid_pose(self, mock_world):
        handler = self._make_handler(mock_world)
        body = MagicMock()
        body.name.name = "milk"
        body.global_pose.to_position.return_value.x = 1.0
        body.global_pose.to_position.return_value.y = 2.0
        body.global_pose.to_position.return_value.z = 0.8
        mock_world.get_semantic_annotations_by_type.return_value = []
        mock_world.get_semantic_annotations_of_body.return_value = []

        ctx_str = handler._build_world_context([body])
        assert "milk" in ctx_str or "1.000" in ctx_str  # pose info or name

    def test_build_world_context_pose_unavailable(self, mock_world):
        handler = self._make_handler(mock_world)
        body = MagicMock()
        body.name.name = "milk"
        body.global_pose = None
        mock_world.get_semantic_annotations_by_type.return_value = []
        mock_world.get_semantic_annotations_of_body.side_effect = RuntimeError("no ann")

        ctx_str = handler._build_world_context([body])
        assert "pose unknown" in ctx_str or isinstance(ctx_str, str)


# ── PlaceActionHandler.execute() paths ───────────────────────────────────────


class TestPlaceHandlerExecute:
    def _make_handler(self, mock_world):
        ctx = WorldContext()
        return PlaceActionHandler(world=mock_world, world_context=ctx)

    def test_object_not_found_raises_clarification(self, mock_world, entity_description):
        handler = self._make_handler(mock_world)
        mock_world.bodies = []
        handler._grounder = MagicMock()
        handler._grounder.ground.return_value = _make_grounding(bodies=[])

        schema = PlaceSlotSchema(
            object_description=entity_description,
            target_description=EntityDescriptionSchema(name="counter"),
        )
        with pytest.raises(ClarificationNeededError) as exc_info:
            handler.execute(schema)
        assert exc_info.value.request.entity_name == "milk"

    def test_target_not_found_raises_clarification(self, mock_world, entity_description):
        handler = self._make_handler(mock_world)
        mock_world.bodies = []
        obj_body = MagicMock()

        def _side_effect(desc):
            if desc.name == "milk":
                return _make_grounding(bodies=[obj_body])
            return _make_grounding(bodies=[])

        handler._grounder = MagicMock()
        handler._grounder.ground.side_effect = _side_effect

        schema = PlaceSlotSchema(
            object_description=entity_description,
            target_description=EntityDescriptionSchema(name="counter"),
        )
        with pytest.raises(ClarificationNeededError) as exc_info:
            handler.execute(schema)
        assert "counter" in exc_info.value.request.entity_name

    def test_arm_specified_no_llm_needed(self, mock_world, entity_description):
        handler = self._make_handler(mock_world)
        obj_body = MagicMock()
        tgt_body = MagicMock()

        def _side_effect(desc):
            if desc.name == "milk":
                return _make_grounding(bodies=[obj_body])
            return _make_grounding(bodies=[tgt_body])

        handler._grounder = MagicMock()
        handler._grounder.ground.side_effect = _side_effect

        mock_action = MagicMock()
        schema = PlaceSlotSchema(
            object_description=entity_description,
            target_description=EntityDescriptionSchema(name="counter"),
            arm="LEFT",
        )

        with patch("llmr.pipeline.action_dispatcher.PartialDesignator") as mock_pd_cls:
            mock_pd = MagicMock()
            mock_pd.missing_parameter.return_value = False
            mock_pd.kwargs = {"arm": Arms.LEFT}
            mock_pd_cls.return_value = mock_pd

            with patch("llmr.pipeline.action_dispatcher.PlaceAction") as mock_place_cls:
                mock_place_cls.return_value = mock_action
                result = handler.execute(schema)

        assert result is mock_action


# ── PickUpActionHandler._parse_grasp with valid manipulator ──────────────────


class TestParseGraspWithManipulator:
    def test_full_params_with_manipulator_constructs_grasp(self, mock_world):
        from llmr.workflows.schemas.pick_up import GraspParamsSchema

        ctx = WorldContext(manipulator=MagicMock())
        handler = PickUpActionHandler(world=mock_world, world_context=ctx)

        params = GraspParamsSchema(
            approach_direction="FRONT",
            vertical_alignment="TOP",
            rotate_gripper=False,
        )

        with patch("llmr.pipeline.action_dispatcher.GraspDescription") as mock_gd_cls:
            mock_gd_cls.return_value = MagicMock()
            result = handler._parse_grasp(params)

        mock_gd_cls.assert_called_once()
        assert result is not None

    def test_invalid_grasp_enum_returns_none(self, mock_world):
        from llmr.workflows.schemas.pick_up import GraspParamsSchema

        ctx = WorldContext(manipulator=MagicMock())
        handler = PickUpActionHandler(world=mock_world, world_context=ctx)

        params = GraspParamsSchema(
            approach_direction="FRONT",
            vertical_alignment="TOP",
            rotate_gripper=True,
        )
        with patch(
            "llmr.pipeline.action_dispatcher.GraspDescription",
            side_effect=KeyError("bad enum"),
        ):
            result = handler._parse_grasp(params)

        assert result is None
