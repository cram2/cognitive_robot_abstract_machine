from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pycram.datastructures.enums import Arms

from llmr.pipeline.action_pipeline import (
    ActionPipeline,
    _serialise_world_for_llm,
)
from llmr.pipeline.action_dispatcher import WorldContext
from llmr.planning.motion_precondition_planner import ExecutionState


# ── _serialise_world_for_llm ──────────────────────────────────────────────────


def _make_body_mock(name: str, is_link: bool = False) -> MagicMock:
    body = MagicMock()
    # If link, use a link-style name; else a plain scene object name
    body.name.name = name
    return body


class TestSerialiseWorldForLlm:
    def test_scene_objects_listed(self):
        world = MagicMock()
        b1 = _make_body_mock("milk")
        b2 = _make_body_mock("cup")
        world.bodies = [b1, b2]
        world.semantic_annotations = []
        result = _serialise_world_for_llm(world)
        assert "milk" in result
        assert "cup" in result
        assert "Scene objects" in result

    def test_robot_links_filtered_out(self):
        world = MagicMock()
        b_link = _make_body_mock("r_forearm_link")
        b_obj = _make_body_mock("cup")
        world.bodies = [b_link, b_obj]
        world.semantic_annotations = []
        result = _serialise_world_for_llm(world)
        assert "cup" in result
        assert "r_forearm_link" not in result

    def test_all_robot_links_falls_back_to_raw_list(self):
        world = MagicMock()
        world.bodies = [_make_body_mock("r_forearm_link")]
        world.semantic_annotations = []
        result = _serialise_world_for_llm(world)
        assert "Bodies present" in result

    def test_more_than_30_bodies_truncated(self):
        world = MagicMock()
        world.bodies = [_make_body_mock(f"r_{i}_link") for i in range(35)]
        world.semantic_annotations = []
        result = _serialise_world_for_llm(world)
        assert "5 more" in result

    def test_bodies_exception_shows_unavailable(self):
        world = MagicMock()
        type(world).bodies = property(fget=MagicMock(side_effect=RuntimeError("no bodies")))
        world.semantic_annotations = []
        result = _serialise_world_for_llm(world)
        assert "unavailable" in result

    def test_no_semantic_annotations_shows_none_found(self):
        world = MagicMock()
        world.bodies = [_make_body_mock("milk")]
        world.semantic_annotations = []
        result = _serialise_world_for_llm(world)
        assert "None found" in result

    def test_semantic_annotations_listed(self):
        world = MagicMock()
        body = _make_body_mock("milk_0")

        ann = MagicMock()
        type(ann).__name__ = "Milk"
        ann.bodies = [body]

        world.bodies = [body]
        world.semantic_annotations = [ann]

        result = _serialise_world_for_llm(world)
        assert "Milk" in result
        assert "milk_0" in result

    def test_robot_link_bodies_in_annotations_filtered(self):
        world = MagicMock()
        link_body = _make_body_mock("r_forearm_link")

        ann = MagicMock()
        type(ann).__name__ = "SomeAnnotation"
        ann.bodies = [link_body]

        world.bodies = []
        world.semantic_annotations = [ann]

        result = _serialise_world_for_llm(world)
        assert "r_forearm_link" not in result
        assert "None found" in result

    def test_exec_state_appended(self):
        world = MagicMock()
        world.bodies = [_make_body_mock("milk")]
        world.semantic_annotations = []

        state = ExecutionState()
        body = MagicMock()
        body.name.name = "milk"
        state.held_objects[Arms.LEFT] = body

        result = _serialise_world_for_llm(world, exec_state=state)
        assert "Robot Arm State" in result
        assert "milk" in result

    def test_no_exec_state_no_arm_section(self):
        world = MagicMock()
        world.bodies = []
        world.semantic_annotations = []
        result = _serialise_world_for_llm(world, exec_state=None)
        assert "Robot Arm State" not in result

    def test_annotation_body_exception_is_swallowed(self):
        """Exception inside ann.bodies iteration does not propagate."""
        world = MagicMock()
        world.bodies = []

        ann = MagicMock()
        type(ann).__name__ = "Milk"
        type(ann).bodies = property(fget=MagicMock(side_effect=RuntimeError("no bodies")))

        world.semantic_annotations = [ann]

        # Should not raise
        result = _serialise_world_for_llm(world)
        assert "Semantic annotations" in result


# ── ActionPipeline.run / classify_and_extract / dispatch ─────────────────────


class TestActionPipelineRun:
    def _make_pipeline(self, world=None, world_context=None):
        from llmr.pipeline.action_pipeline import ActionPipeline
        from llmr.pipeline.action_dispatcher import WorldContext

        return ActionPipeline(
            world=world or MagicMock(),
            world_context=world_context or WorldContext(),
        )

    def test_classify_and_extract_calls_run_slot_filler(self):
        pipeline = self._make_pipeline()
        mock_schema = MagicMock()

        with patch("llmr.pipeline.action_pipeline.run_slot_filler", return_value=mock_schema):
            result = pipeline.classify_and_extract("pick up milk")

        assert result is mock_schema

    def test_classify_and_extract_none_on_slot_filler_failure(self):
        pipeline = self._make_pipeline()

        with patch("llmr.pipeline.action_pipeline.run_slot_filler", return_value=None):
            result = pipeline.classify_and_extract("pick up milk")

        assert result is None

    def test_dispatch_delegates_to_action_dispatcher(self):
        pipeline = self._make_pipeline()
        mock_action = MagicMock()

        from llmr.workflows.schemas.pick_up import PickUpSlotSchema

        schema = PickUpSlotSchema(
            object_description=EntityDescriptionSchema(name="milk")
        )

        with patch(
            "llmr.pipeline.action_pipeline.ActionDispatcher"
        ) as mock_dispatcher_cls:
            mock_dispatcher = MagicMock()
            mock_dispatcher.dispatch.return_value = mock_action
            mock_dispatcher_cls.return_value = mock_dispatcher
            result = pipeline.dispatch(schema)

        assert result is mock_action

    def test_run_raises_on_slot_filler_failure(self):
        pipeline = self._make_pipeline()

        with patch("llmr.pipeline.action_pipeline.run_slot_filler", return_value=None):
            with pytest.raises(RuntimeError, match="Slot-filler failed"):
                pipeline.run("pick up milk")

    def test_run_returns_action_on_success(self):
        pipeline = self._make_pipeline()
        mock_action = MagicMock()
        mock_schema = MagicMock()

        with patch("llmr.pipeline.action_pipeline.run_slot_filler", return_value=mock_schema):
            with patch(
                "llmr.pipeline.action_pipeline.ActionDispatcher"
            ) as mock_dispatcher_cls:
                mock_dispatcher = MagicMock()
                mock_dispatcher.dispatch.return_value = mock_action
                mock_dispatcher_cls.return_value = mock_dispatcher
                result = pipeline.run("pick up milk")

        assert result is mock_action


from llmr.workflows.schemas.common import EntityDescriptionSchema
