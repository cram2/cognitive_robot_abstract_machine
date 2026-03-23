from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llmr.task_decomposer import DecomposedPlan, TaskDecomposer


# ── DecomposedPlan ────────────────────────────────────────────────────────────


class TestDecomposedPlan:
    def test_steps_stored(self):
        plan = DecomposedPlan(steps=["pick up milk", "place milk on counter"])
        assert plan.steps == ["pick up milk", "place milk on counter"]

    def test_dependencies_default_empty(self):
        plan = DecomposedPlan(steps=["x"])
        assert plan.dependencies == {}

    def test_dependencies_stored(self):
        plan = DecomposedPlan(steps=["a", "b"], dependencies={1: [0]})
        assert plan.dependencies == {1: [0]}

    def test_dependencies_independent_per_instance(self):
        p1 = DecomposedPlan(steps=["a"])
        p2 = DecomposedPlan(steps=["b"])
        p1.dependencies[0] = [1]
        assert 0 not in p2.dependencies


# ── TaskDecomposer — helper to build a testable instance ─────────────────────


def _make_decomposer() -> tuple[TaskDecomposer, MagicMock]:
    """Return (decomposer, mock_chain) with the LLM chain replaced by a mock."""
    with patch("llmr.task_decomposer.default_llm") as mock_llm:
        mock_llm.with_structured_output.return_value = MagicMock()
        decomposer = TaskDecomposer()

    mock_chain = MagicMock()
    decomposer._chain = mock_chain
    return decomposer, mock_chain


def _atom(instruction: str, deps: list[int] | None = None):
    """Create a mock _AtomicStep-like object."""
    step = MagicMock()
    step.instruction = instruction
    step.dependencies = deps or []
    return step


def _llm_result(steps: list[MagicMock]) -> MagicMock:
    """Wrap steps in a mock _DecomposedInstructions object."""
    result = MagicMock()
    result.steps = steps
    return result


# ── TaskDecomposer.decompose ──────────────────────────────────────────────────


class TestTaskDecomposerDecompose:
    def test_single_step_passthrough(self):
        decomposer, chain = _make_decomposer()
        chain.invoke.return_value = _llm_result([_atom("pick up milk")])
        plan = decomposer.decompose("pick up milk")
        assert plan.steps == ["pick up milk"]
        assert plan.dependencies == {}

    def test_two_steps_with_dependency(self):
        decomposer, chain = _make_decomposer()
        chain.invoke.return_value = _llm_result(
            [
                _atom("grab the bottle", deps=[]),
                _atom("put the bottle on the shelf", deps=[0]),
            ]
        )
        plan = decomposer.decompose("grab the bottle and put it on the shelf")
        assert len(plan.steps) == 2
        assert plan.dependencies == {1: [0]}

    def test_two_independent_steps_no_deps(self):
        # Use two different action verbs so the single-verb guard doesn't fire.
        # "pick" and "grab" are both in _PICKUP_VERBS; they count as 2 distinct
        # verb tokens once the instruction words are lowercased into a set.
        decomposer, chain = _make_decomposer()
        chain.invoke.return_value = _llm_result(
            [
                _atom("pick up the cup", deps=[]),
                _atom("grab the plate", deps=[]),
            ]
        )
        plan = decomposer.decompose("pick up the cup and grab the plate")
        assert len(plan.steps) == 2
        assert plan.dependencies == {}

    def test_single_verb_instruction_collapsed_to_one_step(self):
        """Guard: single action verb → at most one step, even if LLM returns two."""
        decomposer, chain = _make_decomposer()
        # LLM incorrectly returns two steps for a single-verb instruction
        chain.invoke.return_value = _llm_result(
            [
                _atom("pick up the milk", deps=[]),
                _atom("place the milk somewhere", deps=[0]),
            ]
        )
        plan = decomposer.decompose("pick up the milk")
        # Guard collapses to the best-match step
        assert len(plan.steps) == 1
        assert "milk" in plan.steps[0].lower()

    def test_duplicate_steps_are_deduplicated(self):
        decomposer, chain = _make_decomposer()
        chain.invoke.return_value = _llm_result(
            [
                _atom("pick up the milk", deps=[]),
                _atom("pick up the milk", deps=[]),  # duplicate
            ]
        )
        plan = decomposer.decompose("pick up the milk")
        assert len(plan.steps) == 1

    def test_out_of_range_dependency_is_dropped(self):
        decomposer, chain = _make_decomposer()
        chain.invoke.return_value = _llm_result(
            [
                _atom("grab bottle", deps=[]),
                _atom("put bottle on shelf", deps=[0, 99]),  # 99 is out of range
            ]
        )
        plan = decomposer.decompose("grab bottle and put it on shelf")
        assert 1 in plan.dependencies
        assert 99 not in plan.dependencies[1]

    def test_self_dependency_is_dropped(self):
        decomposer, chain = _make_decomposer()
        chain.invoke.return_value = _llm_result(
            [
                _atom("pick up cup", deps=[0]),  # self-dependency
            ]
        )
        plan = decomposer.decompose("pick up cup")
        assert plan.dependencies == {}

    def test_llm_exception_falls_back_to_original(self):
        decomposer, chain = _make_decomposer()
        chain.invoke.side_effect = RuntimeError("LLM unavailable")
        plan = decomposer.decompose("pick up the cup")
        assert plan.steps == ["pick up the cup"]

    def test_empty_steps_from_llm_falls_back(self):
        decomposer, chain = _make_decomposer()
        chain.invoke.return_value = _llm_result([])
        plan = decomposer.decompose("pick up the cup")
        assert plan.steps == ["pick up the cup"]

    def test_whitespace_only_steps_are_filtered(self):
        decomposer, chain = _make_decomposer()
        chain.invoke.return_value = _llm_result(
            [
                _atom("  ", deps=[]),
                _atom("pick up cup", deps=[]),
            ]
        )
        plan = decomposer.decompose("pick up cup")
        assert "pick up cup" in plan.steps
        assert "" not in plan.steps
        assert "  " not in plan.steps
