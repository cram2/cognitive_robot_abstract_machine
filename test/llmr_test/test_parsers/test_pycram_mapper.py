"""Tests for the PyCRAM action designator mapping agent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.runnables import RunnableLambda

from llmr.workflows.parsers.pycram_mapper import (
    _build_belief_state_context,
    _fetch_belief_state_data,
    action_name_selector_node,
    belief_state_context_node,
    pycram_designator_node,
    pycram_mapper_node,
)
from llmr.workflows.models.pycram_models import ActionNames, Actions, GroundedCramPlans


# ── Fixtures ──────────────────────────────────────────────────────────────────

KINEMATIC_NODES: dict = {"robot": {"type": "robot"}, "cup": {"type": "object"}}
SEMANTIC_ANNOTATIONS: dict = {"cup": {"graspable": True}}

MINIMAL_GROUNDING_STATE: dict = {
    "atomics": "pick up the cup",
    "cram_plans": "(perform (pick-up ?cup))",
    "belief_state_context": "some belief state context",
    "context": "",
    "grounded_cram_plans": ["(pick-up cup_1)"],
    "action_names": ["PickUpAction"],
    "designator_models": "",
}


def _mock_default_llm(return_value) -> MagicMock:
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = RunnableLambda(lambda _: return_value)
    return mock_llm


# ── _fetch_belief_state_data ──────────────────────────────────────────────────

class TestFetchBeliefStateData:
    def test_returns_correct_data_from_both_endpoints(self) -> None:
        mock_kn = MagicMock()
        mock_kn.json.return_value = KINEMATIC_NODES
        mock_sa = MagicMock()
        mock_sa.json.return_value = SEMANTIC_ANNOTATIONS

        with patch(
            "llmr.workflows.parsers.pycram_mapper.requests.get",
            side_effect=[mock_kn, mock_sa],
        ):
            kn, sa = _fetch_belief_state_data()

        assert kn == KINEMATIC_NODES
        assert sa == SEMANTIC_ANNOTATIONS

    def test_calls_expected_endpoints(self) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        with patch(
            "llmr.workflows.parsers.pycram_mapper.requests.get",
            return_value=mock_resp,
        ) as mock_get:
            _fetch_belief_state_data()
        urls = [call.args[0] for call in mock_get.call_args_list]
        assert any("ks_nodes_all" in url for url in urls)
        assert any("semantic_annotations" in url for url in urls)


# ── _build_belief_state_context ───────────────────────────────────────────────

class TestBuildBeliefStateContext:
    def test_contains_kinematic_and_annotation_data(self) -> None:
        context = _build_belief_state_context(KINEMATIC_NODES, SEMANTIC_ANNOTATIONS)
        assert "robot" in context
        assert "graspable" in context

    def test_contains_sdt_overview(self) -> None:
        context = _build_belief_state_context(KINEMATIC_NODES, SEMANTIC_ANNOTATIONS)
        assert "Semantic Digital Twin" in context

    def test_returns_string_for_empty_inputs(self) -> None:
        assert isinstance(_build_belief_state_context({}, {}), str)


# ── belief_state_context_node ─────────────────────────────────────────────────

class TestBeliefStateContextNode:
    def test_returns_belief_state_context_and_grounded_plans(self) -> None:
        mock_grounded = MagicMock(spec=GroundedCramPlans)
        mock_grounded.grounded_plans = ["(pick-up cup_1)", "(navigate kitchen)"]
        mock_llm = _mock_default_llm(mock_grounded)

        with patch(
            "llmr.workflows.parsers.pycram_mapper._fetch_belief_state_data",
            return_value=(KINEMATIC_NODES, SEMANTIC_ANNOTATIONS),
        ), patch("llmr.workflows.parsers.pycram_mapper._default_llm", mock_llm):
            result = belief_state_context_node(MINIMAL_GROUNDING_STATE)

        assert "belief_state_context" in result
        assert result["grounded_cram_plans"] == ["(pick-up cup_1)", "(navigate kitchen)"]


# ── action_name_selector_node ─────────────────────────────────────────────────

class TestActionNameSelectorNode:
    def _make_action_names_mock(self, names: list) -> RunnableLambda:
        response = MagicMock(spec=ActionNames)
        response.model_names = names
        return RunnableLambda(lambda _: response)

    def test_returns_context_and_action_names(self) -> None:
        with patch(
            "llmr.workflows.parsers.pycram_mapper._openai_action_names_llm",
            self._make_action_names_mock(["PickUpAction"]),
        ):
            result = action_name_selector_node(MINIMAL_GROUNDING_STATE)

        assert "context" in result
        assert result["action_names"] == ["PickUpAction"]

    def test_context_contains_grounded_cram_plans(self) -> None:
        with patch(
            "llmr.workflows.parsers.pycram_mapper._openai_action_names_llm",
            self._make_action_names_mock([]),
        ):
            result = action_name_selector_node(MINIMAL_GROUNDING_STATE)
        assert "(pick-up cup_1)" in result["context"]

    def test_list_plans_joined_with_newlines(self) -> None:
        state = {**MINIMAL_GROUNDING_STATE, "grounded_cram_plans": ["plan_a", "plan_b"]}
        with patch(
            "llmr.workflows.parsers.pycram_mapper._openai_action_names_llm",
            self._make_action_names_mock([]),
        ):
            result = action_name_selector_node(state)
        assert "plan_a\nplan_b" in result["context"]

    def test_string_plans_used_directly(self) -> None:
        state = {**MINIMAL_GROUNDING_STATE, "grounded_cram_plans": "(pick-up cup_1)"}
        with patch(
            "llmr.workflows.parsers.pycram_mapper._openai_action_names_llm",
            self._make_action_names_mock([]),
        ):
            result = action_name_selector_node(state)
        assert "(pick-up cup_1)" in result["context"]


# ── pycram_designator_node ────────────────────────────────────────────────────

class TestPycramDesignatorNode:
    def _make_actions_runnable(self) -> RunnableLambda:
        MockActionClass = type("PickUpAction", (), {})
        mock_action = MagicMock()
        mock_action.__class__ = MockActionClass
        mock_response = MagicMock(spec=Actions)
        mock_response.models = [mock_action]
        mock_response.model_dump.return_value = {"models": [{"type": "PickUpAction"}]}
        return RunnableLambda(lambda _: mock_response)

    def test_posts_to_runner_with_correct_payload(self) -> None:
        with patch(
            "llmr.workflows.parsers.pycram_mapper._openai_actions_llm",
            self._make_actions_runnable(),
        ), patch("llmr.workflows.parsers.pycram_mapper.requests.post") as mock_post:
            result = pycram_designator_node(MINIMAL_GROUNDING_STATE)

        mock_post.assert_called_once()
        assert "runner" in mock_post.call_args.args[0]
        payload = mock_post.call_args.kwargs["json"]
        assert "model_names" in payload
        assert "models" in payload
        assert result == {}


# ── pycram_mapper_node ────────────────────────────────────────────────────────

class TestPycramMapperNode:
    def test_returns_action_names_and_passes_correct_state(self) -> None:
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"action_names": ["PickUpAction", "NavigateAction"]}

        with patch("llmr.workflows.parsers.pycram_mapper.pycram_mapper_graph", mock_graph):
            result = pycram_mapper_node({
                "instruction": "pick up the cup",
                "intents": {},
                "cram_plan_response": "(pick-up ?cup)",
            })

        assert result == {"pycram_action_names": ["PickUpAction", "NavigateAction"]}
        invoked = mock_graph.invoke.call_args[0][0]
        assert invoked["instruction"] == "pick up the cup"
        assert invoked["cram_plan_response"] == "(pick-up ?cup)"
