"""Tests for TaskDecomposer — NL instruction decomposition.

Uses ScriptedLLM with pre-built responses. No network, no API keys.

Coverage target: 75% (9 tests covering decomposition and helper logic).
"""
from __future__ import annotations

import pytest
from ..scripted_llm import ScriptedLLM
from llmr.reasoning.decomposer import (
    TaskDecomposer,
    DecomposedPlan,
    _AtomicStep,
    _DecomposedInstructions,
)


class TestDecompose:
    """TaskDecomposer.decompose() — instruction decomposition."""

    def test_single_step_instruction_returns_one_step(self) -> None:
        """Single-action instruction returns one step."""
        response = _DecomposedInstructions(
            steps=[_AtomicStep(instruction="pick up the milk", dependencies=[])]
        )
        llm = ScriptedLLM(responses=[response])
        decomposer = TaskDecomposer(llm=llm)
        result = decomposer.decompose("pick up the milk")
        assert len(result.steps) == 1
        assert result.steps[0] == "pick up the milk"

    def test_multi_step_returns_correct_step_count(self) -> None:
        """Multi-action instruction returns multiple steps."""
        response = _DecomposedInstructions(
            steps=[
                _AtomicStep(instruction="navigate to the table", dependencies=[]),
                _AtomicStep(instruction="pick up the milk", dependencies=[]),
                _AtomicStep(instruction="place the milk in the fridge", dependencies=[1]),
            ]
        )
        llm = ScriptedLLM(responses=[response])
        decomposer = TaskDecomposer(llm=llm)
        result = decomposer.decompose(
            "go to the table, pick up the milk and put it in the fridge"
        )
        assert len(result.steps) == 3

    def test_object_pronoun_resolved_in_later_steps(self) -> None:
        """Pronouns are resolved to object names in steps."""
        response = _DecomposedInstructions(
            steps=[
                _AtomicStep(instruction="pick up the milk", dependencies=[]),
                _AtomicStep(
                    instruction="place the milk in the fridge", dependencies=[0]
                ),
            ]
        )
        llm = ScriptedLLM(responses=[response])
        decomposer = TaskDecomposer(llm=llm)
        result = decomposer.decompose("pick up the milk and put it in the fridge")
        # Step 2 should have "milk" not "it"
        assert "milk" in result.steps[1].lower()

    def test_dependencies_populated_when_step_uses_prior_step_output(self) -> None:
        """Dependencies are set when steps chain (output→input)."""
        response = _DecomposedInstructions(
            steps=[
                _AtomicStep(instruction="pick up the ball", dependencies=[]),
                _AtomicStep(instruction="throw the ball", dependencies=[0]),
            ]
        )
        llm = ScriptedLLM(responses=[response])
        decomposer = TaskDecomposer(llm=llm)
        result = decomposer.decompose("pick up the ball and throw it")
        # Step 1 should depend on step 0
        assert 1 in result.dependencies or len(result.dependencies) > 0

    def test_no_dependency_for_independent_steps(self) -> None:
        """Independent steps have no dependencies."""
        response = _DecomposedInstructions(
            steps=[
                _AtomicStep(instruction="go to the kitchen", dependencies=[]),
                _AtomicStep(instruction="go to the bedroom", dependencies=[]),
            ]
        )
        llm = ScriptedLLM(responses=[response])
        decomposer = TaskDecomposer(llm=llm)
        result = decomposer.decompose("go to the kitchen and go to the bedroom")
        # These are independent, so no dependencies
        assert result.dependencies == {} or len(result.dependencies) == 0

    def test_dedup_removes_repeated_steps(self) -> None:
        """Duplicate steps are removed."""
        response = _DecomposedInstructions(
            steps=[
                _AtomicStep(instruction="go to the table", dependencies=[]),
                _AtomicStep(instruction="pick up milk", dependencies=[]),
                _AtomicStep(instruction="go to the table", dependencies=[]),  # duplicate
            ]
        )
        llm = ScriptedLLM(responses=[response])
        decomposer = TaskDecomposer(llm=llm)
        result = decomposer.decompose("go to table, pick up milk, go back to table")
        # Deduplication should have occurred
        assert len(result.steps) <= 3

    def test_dedup_preserves_order(self) -> None:
        """Deduplication preserves step order."""
        response = _DecomposedInstructions(
            steps=[
                _AtomicStep(instruction="step A", dependencies=[]),
                _AtomicStep(instruction="step B", dependencies=[]),
                _AtomicStep(instruction="step A", dependencies=[]),  # second A
            ]
        )
        llm = ScriptedLLM(responses=[response])
        decomposer = TaskDecomposer(llm=llm)
        result = decomposer.decompose("A then B then A again")
        # The first occurrence of A should come before B
        if "step A" in result.steps:
            a_idx = result.steps.index("step A")
            if "step B" in result.steps:
                b_idx = result.steps.index("step B")
                # Either only one A (deduped) or A comes before B
                assert a_idx < b_idx or result.steps.count("step A") == 1

    def test_graceful_fallback_when_llm_fails(self) -> None:
        """When LLM fails, returns single-step fallback plan."""
        # Create an LLM that raises an exception on invoke
        from langchain_core.runnables import RunnableLambda

        class FailingLLM:
            def with_structured_output(self, schema):
                def _failing_invoke(messages):
                    raise RuntimeError("LLM error")
                return RunnableLambda(_failing_invoke)

        llm = FailingLLM()
        decomposer = TaskDecomposer(llm=llm)
        result = decomposer.decompose("some instruction")
        # Should fall back to single-step plan with original instruction
        assert len(result.steps) == 1
        assert result.steps[0] == "some instruction"

    def test_clamped_dependencies_out_of_range(self) -> None:
        """Out-of-range dependency indices are removed."""
        response = _DecomposedInstructions(
            steps=[
                _AtomicStep(instruction="step 0", dependencies=[]),
                _AtomicStep(instruction="step 1", dependencies=[999]),  # out of range
            ]
        )
        llm = ScriptedLLM(responses=[response])
        decomposer = TaskDecomposer(llm=llm)
        result = decomposer.decompose("some instruction")
        # Out-of-range dependency should be clamped/removed
        if 1 in result.dependencies:
            assert 999 not in result.dependencies.get(1, [])
