"""Tests that invoke LangGraph node closures to cover the inner node functions."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.runnables import RunnableLambda

from llmr.workflows.schemas.common import EntityDescriptionSchema
from llmr.workflows.schemas.pick_up import PickUpDiscreteResolutionSchema
from llmr.workflows.schemas.place import PlaceDiscreteResolutionSchema
from llmr.workflows.schemas.recovery import RecoverySchema


# ── resolver._build_resolver_graph / _resolver_node ──────────────────────────


class TestResolverNode:
    """Invoke the compiled resolver graph to exercise the inner _resolver_node."""

    def _clear_cache_for(self, prompt, schema_cls):
        from llmr.workflows.nodes.resolver import _graph_cache

        key = (id(prompt), schema_cls)
        return _graph_cache.pop(key, None)

    def _restore_cache(self, prompt, schema_cls, original):
        from llmr.workflows.nodes.resolver import _graph_cache

        key = (id(prompt), schema_cls)
        if original is not None:
            _graph_cache[key] = original

    def test_pickup_resolver_node_success(self):
        from llmr.workflows.nodes.resolver import _build_resolver_graph
        from llmr.workflows.prompts.pick_up import pick_up_resolver_prompt

        schema_obj = PickUpDiscreteResolutionSchema(
            arm="LEFT",
            approach_direction="FRONT",
            vertical_alignment="TOP",
            rotate_gripper=False,
            reasoning="Object is to the left.",
        )
        original = self._clear_cache_for(pick_up_resolver_prompt, PickUpDiscreteResolutionSchema)
        try:
            with patch("llmr.workflows.nodes.resolver.default_llm") as mock_llm:
                mock_llm.with_structured_output.return_value = RunnableLambda(
                    lambda _: schema_obj
                )
                graph = _build_resolver_graph(
                    pick_up_resolver_prompt, PickUpDiscreteResolutionSchema
                )

            state = {
                "messages": [],
                "world_context": "Object is to the left.",
                "known_parameters": "None",
                "parameters_to_resolve": "arm",
                "resolved_schema": None,
                "error": None,
            }
            result = graph.invoke(state)
        finally:
            self._restore_cache(pick_up_resolver_prompt, PickUpDiscreteResolutionSchema, original)

        assert result["error"] is None
        assert result["resolved_schema"]["arm"] == "LEFT"

    def test_pickup_resolver_node_exception_sets_error(self):
        from llmr.workflows.nodes.resolver import _build_resolver_graph
        from llmr.workflows.prompts.pick_up import pick_up_resolver_prompt

        def _raise(_):
            raise RuntimeError("LLM timeout")

        original = self._clear_cache_for(pick_up_resolver_prompt, PickUpDiscreteResolutionSchema)
        try:
            with patch("llmr.workflows.nodes.resolver.default_llm") as mock_llm:
                mock_llm.with_structured_output.return_value = RunnableLambda(_raise)
                graph = _build_resolver_graph(
                    pick_up_resolver_prompt, PickUpDiscreteResolutionSchema
                )

            state = {
                "messages": [],
                "world_context": "ctx",
                "known_parameters": "x",
                "parameters_to_resolve": "arm",
                "resolved_schema": None,
                "error": None,
            }
            result = graph.invoke(state)
        finally:
            self._restore_cache(pick_up_resolver_prompt, PickUpDiscreteResolutionSchema, original)

        assert result["resolved_schema"] is None
        assert "LLM timeout" in result["error"]

    def test_graph_cached_after_first_build(self):
        from llmr.workflows.nodes.resolver import _build_resolver_graph, _graph_cache
        from llmr.workflows.prompts.pick_up import pick_up_resolver_prompt

        schema_obj = PickUpDiscreteResolutionSchema(
            arm="RIGHT",
            approach_direction="BACK",
            vertical_alignment="BOTTOM",
            rotate_gripper=True,
            reasoning="x",
        )
        original = self._clear_cache_for(pick_up_resolver_prompt, PickUpDiscreteResolutionSchema)
        try:
            with patch("llmr.workflows.nodes.resolver.default_llm") as mock_llm:
                mock_llm.with_structured_output.return_value = RunnableLambda(
                    lambda _: schema_obj
                )
                g1 = _build_resolver_graph(
                    pick_up_resolver_prompt, PickUpDiscreteResolutionSchema
                )
                g2 = _build_resolver_graph(
                    pick_up_resolver_prompt, PickUpDiscreteResolutionSchema
                )
        finally:
            self._restore_cache(pick_up_resolver_prompt, PickUpDiscreteResolutionSchema, original)

        assert g1 is g2  # cache hit → same object

    def test_place_resolver_node_success(self):
        from llmr.workflows.nodes.resolver import _build_resolver_graph
        from llmr.workflows.prompts.place import place_resolver_prompt

        schema_obj = PlaceDiscreteResolutionSchema(
            arm="RIGHT", reasoning="Right arm is free."
        )
        original = self._clear_cache_for(place_resolver_prompt, PlaceDiscreteResolutionSchema)
        try:
            with patch("llmr.workflows.nodes.resolver.default_llm") as mock_llm:
                mock_llm.with_structured_output.return_value = RunnableLambda(
                    lambda _: schema_obj
                )
                graph = _build_resolver_graph(
                    place_resolver_prompt, PlaceDiscreteResolutionSchema
                )
            state = {
                "messages": [],
                "world_context": "ctx",
                "known_parameters": "None",
                "parameters_to_resolve": "arm",
                "resolved_schema": None,
                "error": None,
            }
            result = graph.invoke(state)
        finally:
            self._restore_cache(place_resolver_prompt, PlaceDiscreteResolutionSchema, original)

        assert result["resolved_schema"]["arm"] == "RIGHT"


# ── recovery_resolver._build_recovery_graph / _recovery_node ─────────────────


class TestRecoveryNode:
    """Invoke the compiled recovery graph to exercise the inner _recovery_node."""

    def test_recovery_node_success(self):
        import llmr.workflows.nodes.recovery_resolver as rr_mod

        schema_obj = RecoverySchema(
            recovery_strategy="ABORT",
            failure_diagnosis="Object not reachable.",
            reasoning="No alternative.",
        )
        original = rr_mod._recovery_graph
        rr_mod._recovery_graph = None
        try:
            with patch("llmr.workflows.nodes.recovery_resolver.default_llm") as mock_llm:
                mock_llm.with_structured_output.return_value = RunnableLambda(
                    lambda _: schema_obj
                )
                graph = rr_mod._build_recovery_graph()

            state = {
                "messages": [],
                "world_context": "ctx",
                "original_instruction": "pick up milk",
                "failed_action_description": "PickUpAction",
                "error_message": "IK failed",
                "resolved_schema": None,
                "error": None,
            }
            result = graph.invoke(state)
        finally:
            rr_mod._recovery_graph = original

        assert result["error"] is None
        assert result["resolved_schema"]["recovery_strategy"] == "ABORT"

    def test_recovery_node_exception_sets_error(self):
        import llmr.workflows.nodes.recovery_resolver as rr_mod

        def _raise(_):
            raise RuntimeError("recovery LLM failed")

        original = rr_mod._recovery_graph
        rr_mod._recovery_graph = None
        try:
            with patch("llmr.workflows.nodes.recovery_resolver.default_llm") as mock_llm:
                mock_llm.with_structured_output.return_value = RunnableLambda(_raise)
                graph = rr_mod._build_recovery_graph()

            state = {
                "messages": [],
                "world_context": "ctx",
                "original_instruction": "x",
                "failed_action_description": "x",
                "error_message": "x",
                "resolved_schema": None,
                "error": None,
            }
            result = graph.invoke(state)
        finally:
            rr_mod._recovery_graph = original

        assert result["resolved_schema"] is None
        assert "recovery LLM failed" in result["error"]

    def test_recovery_graph_cached_on_second_call(self):
        import llmr.workflows.nodes.recovery_resolver as rr_mod

        schema_obj = RecoverySchema(
            recovery_strategy="REPLAN_FULL",
            revised_instruction="pick up cup with left arm",
            failure_diagnosis="x",
            reasoning="x",
        )
        original = rr_mod._recovery_graph
        rr_mod._recovery_graph = None
        try:
            with patch("llmr.workflows.nodes.recovery_resolver.default_llm") as mock_llm:
                mock_llm.with_structured_output.return_value = RunnableLambda(
                    lambda _: schema_obj
                )
                g1 = rr_mod._build_recovery_graph()
                g2 = rr_mod._build_recovery_graph()
        finally:
            rr_mod._recovery_graph = original

        assert g1 is g2