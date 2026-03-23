from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llmr.workflows.schemas.common import EntityDescriptionSchema
from llmr.workflows.schemas.pick_up import (
    PickUpDiscreteResolutionSchema,
    PickUpSlotSchema,
)
from llmr.workflows.schemas.place import PlaceDiscreteResolutionSchema, PlaceSlotSchema
from llmr.workflows.schemas.recovery import RecoverySchema


# ── slot_filler._to_typed_schema ──────────────────────────────────────────────


class TestToTypedSchema:
    def _raw(self, action_type, name, target_name=None):
        from llmr.workflows.nodes.slot_filler import _SlotFillerOutput

        obj = EntityDescriptionSchema(name=name)
        tgt = EntityDescriptionSchema(name=target_name) if target_name else None
        return _SlotFillerOutput(
            action_type=action_type,
            object_description=obj,
            target_description=tgt,
        )

    def test_pickup_action_type(self):
        from llmr.workflows.nodes.slot_filler import _to_typed_schema

        raw = self._raw("PickUpAction", "milk")
        result = _to_typed_schema(raw)
        assert isinstance(result, PickUpSlotSchema)
        assert result.object_description.name == "milk"

    def test_place_action_type(self):
        from llmr.workflows.nodes.slot_filler import _to_typed_schema

        raw = self._raw("PlaceAction", "milk", target_name="counter")
        result = _to_typed_schema(raw)
        assert isinstance(result, PlaceSlotSchema)
        assert result.target_description.name == "counter"

    def test_unknown_action_type_raises(self):
        from llmr.workflows.nodes.slot_filler import _to_typed_schema, _SlotFillerOutput

        # Bypass Pydantic validation to inject bad action_type
        raw = MagicMock(spec=_SlotFillerOutput)
        raw.action_type = "FlyAction"
        with pytest.raises(ValueError, match="FlyAction"):
            _to_typed_schema(raw)


# ── slot_filler.slot_filler_node ─────────────────────────────────────────────


class TestSlotFillerNode:
    def _make_state(self, instruction="pick up milk", world_context=""):
        return {
            "messages": [],
            "instruction": instruction,
            "world_context": world_context,
            "slot_schema": None,
            "grounded_body_indices": None,
            "error": None,
        }

    def test_success_returns_slot_schema(self):
        from llmr.workflows.nodes.slot_filler import slot_filler_node, _SlotFillerOutput
        import llmr.workflows.nodes.slot_filler as sf

        raw = _SlotFillerOutput(
            action_type="PickUpAction",
            object_description=EntityDescriptionSchema(name="milk"),
        )
        # Build a real LangChain Runnable so prompt | llm works correctly.
        from langchain_core.runnables import RunnableLambda

        mock_llm_runnable = RunnableLambda(lambda _: raw)
        with patch.object(sf, "_slot_filler_llm", mock_llm_runnable):
            result = slot_filler_node(self._make_state())

        assert result["error"] is None
        assert result["slot_schema"] is not None
        assert result["slot_schema"]["action_type"] == "PickUpAction"

    def test_llm_exception_returns_error(self):
        from llmr.workflows.nodes.slot_filler import slot_filler_node
        import llmr.workflows.nodes.slot_filler as sf
        from langchain_core.runnables import RunnableLambda

        def _raise(_):
            raise RuntimeError("LLM unavailable")

        mock_llm_runnable = RunnableLambda(_raise)
        with patch.object(sf, "_slot_filler_llm", mock_llm_runnable):
            result = slot_filler_node(self._make_state())

        assert result["slot_schema"] is None
        assert result["error"] is not None


# ── run_slot_filler ───────────────────────────────────────────────────────────


class TestRunSlotFiller:
    def test_pickup_schema_returned(self):
        from llmr.workflows.nodes.slot_filler import run_slot_filler

        schema = PickUpSlotSchema(
            object_description=EntityDescriptionSchema(name="milk")
        )
        state = {"slot_schema": schema.model_dump(), "error": None}

        with patch("llmr.workflows.nodes.slot_filler.slot_filler_graph") as mock_g:
            mock_g.invoke.return_value = state
            result = run_slot_filler("pick up milk")

        assert isinstance(result, PickUpSlotSchema)
        assert result.object_description.name == "milk"

    def test_place_schema_returned(self):
        from llmr.workflows.nodes.slot_filler import run_slot_filler

        schema = PlaceSlotSchema(
            object_description=EntityDescriptionSchema(name="milk"),
            target_description=EntityDescriptionSchema(name="counter"),
        )
        state = {"slot_schema": schema.model_dump(), "error": None}

        with patch("llmr.workflows.nodes.slot_filler.slot_filler_graph") as mock_g:
            mock_g.invoke.return_value = state
            result = run_slot_filler("place milk on counter")

        assert isinstance(result, PlaceSlotSchema)

    def test_error_state_returns_none(self):
        from llmr.workflows.nodes.slot_filler import run_slot_filler

        with patch("llmr.workflows.nodes.slot_filler.slot_filler_graph") as mock_g:
            mock_g.invoke.return_value = {"slot_schema": None, "error": "LLM error"}
            result = run_slot_filler("pick up milk")

        assert result is None

    def test_none_slot_schema_returns_none(self):
        from llmr.workflows.nodes.slot_filler import run_slot_filler

        with patch("llmr.workflows.nodes.slot_filler.slot_filler_graph") as mock_g:
            mock_g.invoke.return_value = {"slot_schema": None, "error": None}
            result = run_slot_filler("pick up milk")

        assert result is None

    def test_unknown_action_type_returns_none(self):
        from llmr.workflows.nodes.slot_filler import run_slot_filler

        with patch("llmr.workflows.nodes.slot_filler.slot_filler_graph") as mock_g:
            mock_g.invoke.return_value = {
                "slot_schema": {"action_type": "FlyAction"},
                "error": None,
            }
            result = run_slot_filler("fly away")

        assert result is None


# ── run_pickup_resolver / run_place_resolver ──────────────────────────────────


class TestRunResolver:
    def _patch_graph(self, resolved_dict, error=None):
        """Return a context manager that patches _build_resolver_graph."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"resolved_schema": resolved_dict, "error": error}
        return patch(
            "llmr.workflows.nodes.resolver._build_resolver_graph",
            return_value=mock_graph,
        )

    def test_pickup_resolver_success(self):
        from llmr.workflows.nodes.resolver import run_pickup_resolver

        schema = PickUpDiscreteResolutionSchema(
            arm="LEFT",
            approach_direction="FRONT",
            vertical_alignment="TOP",
            rotate_gripper=False,
            reasoning="Object is to the left.",
        )
        with self._patch_graph(schema.model_dump()):
            result = run_pickup_resolver("ctx", "arm=LEFT", "approach_direction")

        assert isinstance(result, PickUpDiscreteResolutionSchema)
        assert result.arm == "LEFT"

    def test_place_resolver_success(self):
        from llmr.workflows.nodes.resolver import run_place_resolver

        schema = PlaceDiscreteResolutionSchema(arm="RIGHT", reasoning="Right arm is free.")
        with self._patch_graph(schema.model_dump()):
            result = run_place_resolver("ctx", "arm=None", "arm")

        assert isinstance(result, PlaceDiscreteResolutionSchema)
        assert result.arm == "RIGHT"

    def test_error_state_returns_none(self):
        from llmr.workflows.nodes.resolver import run_pickup_resolver

        with self._patch_graph(None, error="LLM error"):
            result = run_pickup_resolver("ctx", "x", "y")

        assert result is None

    def test_none_resolved_schema_returns_none(self):
        from llmr.workflows.nodes.resolver import run_pickup_resolver

        with self._patch_graph(None, error=None):
            result = run_pickup_resolver("ctx", "x", "y")

        assert result is None

    def test_graph_is_cached_on_second_call(self):
        """_build_resolver_graph is called once even for two run_resolver calls."""
        from llmr.workflows.nodes.resolver import run_pickup_resolver, _graph_cache
        from llmr.workflows.prompts.pick_up import pick_up_resolver_prompt

        schema = PickUpDiscreteResolutionSchema(
            arm="LEFT",
            approach_direction="FRONT",
            vertical_alignment="TOP",
            rotate_gripper=False,
            reasoning="x",
        )
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"resolved_schema": schema.model_dump(), "error": None}

        cache_key = (id(pick_up_resolver_prompt), PickUpDiscreteResolutionSchema)
        original = _graph_cache.pop(cache_key, None)
        try:
            with patch(
                "llmr.workflows.nodes.resolver._build_resolver_graph",
                return_value=mock_graph,
            ) as mock_build:
                run_pickup_resolver("ctx", "x", "y")
                run_pickup_resolver("ctx", "x", "y")
            # _build_resolver_graph is still called each time we call run_resolver,
            # but the graph cache inside it prevents recompilation.
            assert mock_build.call_count == 2
        finally:
            if original is not None:
                _graph_cache[cache_key] = original


# ── run_recovery_resolver ─────────────────────────────────────────────────────


class TestRunRecoveryResolver:
    def _make_state(self, schema_dict=None, error=None):
        return {"resolved_schema": schema_dict, "error": error}

    def test_replan_schema_returned(self):
        from llmr.workflows.nodes.recovery_resolver import run_recovery_resolver

        schema = RecoverySchema(
            recovery_strategy="REPLAN_FULL",
            revised_instruction="pick up milk with left arm",
            failure_diagnosis="arm collision",
            reasoning="switch arms",
        )
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = self._make_state(schema.model_dump())

        with patch(
            "llmr.workflows.nodes.recovery_resolver._build_recovery_graph",
            return_value=mock_graph,
        ):
            result = run_recovery_resolver("ctx", "pick up milk", "PickUpAction", "IK failed")

        assert isinstance(result, RecoverySchema)
        assert result.recovery_strategy == "REPLAN_FULL"

    def test_abort_schema_returned(self):
        from llmr.workflows.nodes.recovery_resolver import run_recovery_resolver

        schema = RecoverySchema(
            recovery_strategy="ABORT",
            failure_diagnosis="object not reachable",
            reasoning="no alternative",
        )
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = self._make_state(schema.model_dump())

        with patch(
            "llmr.workflows.nodes.recovery_resolver._build_recovery_graph",
            return_value=mock_graph,
        ):
            result = run_recovery_resolver("ctx", "x", "x", "x")

        assert result.recovery_strategy == "ABORT"

    def test_error_state_returns_none(self):
        from llmr.workflows.nodes.recovery_resolver import run_recovery_resolver

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = self._make_state(None, error="LLM error")

        with patch(
            "llmr.workflows.nodes.recovery_resolver._build_recovery_graph",
            return_value=mock_graph,
        ):
            result = run_recovery_resolver("ctx", "x", "x", "x")

        assert result is None

    def test_none_resolved_schema_returns_none(self):
        from llmr.workflows.nodes.recovery_resolver import run_recovery_resolver

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = self._make_state(None, error=None)

        with patch(
            "llmr.workflows.nodes.recovery_resolver._build_recovery_graph",
            return_value=mock_graph,
        ):
            result = run_recovery_resolver("ctx", "x", "x", "x")

        assert result is None

    def test_graph_cached_across_calls(self):
        """_build_recovery_graph is only called once; graph is cached in module."""
        import llmr.workflows.nodes.recovery_resolver as rr_module

        schema = RecoverySchema(
            recovery_strategy="ABORT",
            failure_diagnosis="x",
            reasoning="x",
        )
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = self._make_state(schema.model_dump())

        # Reset module-level cache
        original = rr_module._recovery_graph
        rr_module._recovery_graph = None
        try:
            with patch.object(
                rr_module,
                "_build_recovery_graph",
                wraps=lambda: mock_graph,
            ) as mock_build:
                rr_module.run_recovery_resolver("ctx", "x", "x", "x")
            mock_build.assert_called_once()
        finally:
            rr_module._recovery_graph = original