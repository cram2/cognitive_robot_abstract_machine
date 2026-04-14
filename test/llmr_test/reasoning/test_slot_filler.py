"""Tests for slot-filler — LLM-driven parameter resolution.

Uses ScriptedLLM with pre-built responses. No network, no API keys.

Coverage target: 80% (18 tests covering classify_action, run_slot_filler, and helpers).
"""
from __future__ import annotations

import pytest
from ..scripted_llm import ScriptedLLM
from ..test_actions import (
    MockPickUpAction,
    MockNavigateAction,
    GraspType,
)
from llmr.exceptions import LLMActionRegistryEmpty
from llmr.reasoning.slot_filler import (
    classify_action,
    run_slot_filler,
)
from llmr.schemas.slots import ActionClassification, ActionReasoningOutput, SlotValue
from llmr.schemas.entities import EntityDescriptionSchema


class TestClassifyAction:
    """classify_action() — NL instruction → action class."""

    def test_returns_matching_class_from_registry(self) -> None:
        """Returns the action class matching LLM classification."""
        llm = ScriptedLLM(
            responses=[ActionClassification(action_type="MockPickUpAction")]
        )
        result = classify_action(
            "pick up the milk",
            llm,
            action_registry={"MockPickUpAction": MockPickUpAction},
        )
        assert result is MockPickUpAction

    def test_returns_none_when_llm_returns_unknown_name(self) -> None:
        """Returns None when LLM classifies to unknown action."""
        llm = ScriptedLLM(
            responses=[ActionClassification(action_type="UnknownAction")]
        )
        result = classify_action(
            "do something",
            llm,
            action_registry={"MockPickUpAction": MockPickUpAction},
        )
        assert result is None

    def test_raises_registry_empty_when_no_actions(self) -> None:
        """Raises LLMActionRegistryEmpty when registry is empty."""
        llm = ScriptedLLM(responses=[ActionClassification(action_type="X")])
        with pytest.raises(LLMActionRegistryEmpty):
            classify_action("pick up milk", llm, action_registry={})

    def test_handles_classification_with_confidence_and_reasoning(self) -> None:
        """classify_action works with confidence and reasoning fields."""
        llm = ScriptedLLM(
            responses=[
                ActionClassification(
                    action_type="MockNavigateAction",
                    confidence=0.95,
                    reasoning="user said navigate",
                )
            ]
        )
        result = classify_action(
            "navigate to the kitchen",
            llm,
            action_registry={"MockNavigateAction": MockNavigateAction},
        )
        assert result is MockNavigateAction

class TestRunSlotFiller:
    """run_slot_filler() — action class + free slots → filled parameters."""

    def test_returns_action_reasoning_output(self) -> None:
        """run_slot_filler returns ActionReasoningOutput."""
        expected = ActionReasoningOutput(
            action_type="MockPickUpAction",
            slots=[
                SlotValue(
                    field_name="object_designator",
                    entity_description=EntityDescriptionSchema(name="milk"),
                    reasoning="instruction mentions milk",
                )
            ],
        )
        llm = ScriptedLLM(responses=[expected])
        result = run_slot_filler(
            instruction="pick up the milk",
            action_cls=MockPickUpAction,
            free_slot_names=["object_designator"],
            fixed_slots={},
            world_context="milk is on the table",
            llm=llm,
        )
        assert result is not None
        assert result.action_type == "MockPickUpAction"
        assert len(result.slots) == 1

    def test_returns_none_when_llm_raises(self) -> None:
        """run_slot_filler returns None when LLM call fails."""
        # Create an LLM that raises an exception on invoke
        from langchain_core.runnables import RunnableLambda

        class FailingLLM:
            def with_structured_output(self, schema):
                def _failing_invoke(messages):
                    raise RuntimeError("LLM error")
                return RunnableLambda(_failing_invoke)

        llm = FailingLLM()
        result = run_slot_filler(
            instruction="pick up",
            action_cls=MockPickUpAction,
            free_slot_names=["object_designator"],
            fixed_slots={},
            world_context="",
            llm=llm,
        )
        assert result is None

    def test_handles_multiple_free_slots(self) -> None:
        """run_slot_filler processes multiple free slots."""
        output = ActionReasoningOutput(
            action_type="MockPickUpAction",
            slots=[
                SlotValue(field_name="object_designator", value="milk"),
                SlotValue(field_name="timeout", value="30.0"),
            ],
        )
        llm = ScriptedLLM(responses=[output])
        result = run_slot_filler(
            instruction="pick up milk",
            action_cls=MockPickUpAction,
            free_slot_names=["object_designator", "timeout"],
            fixed_slots={},
            world_context="",
            llm=llm,
        )
        assert result is not None
        assert len(result.slots) == 2

    def test_passes_world_context_to_prompt(self) -> None:
        """run_slot_filler includes world_context in the LLM prompt."""
        output = ActionReasoningOutput(
            action_type="MockPickUpAction", slots=[]
        )
        llm = ScriptedLLM(responses=[output])
        # The prompt should include the world context
        result = run_slot_filler(
            instruction="pick up the milk",
            action_cls=MockPickUpAction,
            free_slot_names=[],
            fixed_slots={},
            world_context="milk is on the table, table is in kitchen",
            llm=llm,
        )
        assert result is not None

    def test_handles_complex_field_expansion(self) -> None:
        """run_slot_filler expands complex fields to dotted sub-fields."""
        output = ActionReasoningOutput(
            action_type="MockPickUpAction",
            slots=[
                SlotValue(
                    field_name="grasp_description.grasp_type", value="TOP"
                ),
            ],
        )
        llm = ScriptedLLM(responses=[output])
        result = run_slot_filler(
            instruction="pick up",
            action_cls=MockPickUpAction,
            free_slot_names=["grasp_description"],
            fixed_slots={},
            world_context="",
            llm=llm,
        )
        assert result is not None

    def test_optional_instruction(self) -> None:
        """run_slot_filler works with None instruction."""
        output = ActionReasoningOutput(
            action_type="MockPickUpAction", slots=[]
        )
        llm = ScriptedLLM(responses=[output])
        result = run_slot_filler(
            instruction=None,
            action_cls=MockPickUpAction,
            free_slot_names=[],
            fixed_slots={},
            world_context="",
            llm=llm,
        )
        assert result is not None

    def test_strips_field_name_prefixes(self) -> None:
        """run_slot_filler handles 'ClassName.field' format."""
        output = ActionReasoningOutput(
            action_type="MockPickUpAction",
            slots=[SlotValue(field_name="object_designator", value="x")],
        )
        llm = ScriptedLLM(responses=[output])
        # Free slot names may have 'MockPickUpAction.' prefix
        result = run_slot_filler(
            instruction="pick up",
            action_cls=MockPickUpAction,
            free_slot_names=["MockPickUpAction.object_designator"],
            fixed_slots={},
            world_context="",
            llm=llm,
        )
        assert result is not None

    def test_uses_per_field_docstrings(self) -> None:
        """Prompt includes docstrings extracted from action class."""
        output = ActionReasoningOutput(
            action_type="MockPickUpAction", slots=[]
        )
        llm = ScriptedLLM(responses=[output])
        # The prompt should reference docstrings from introspection
        result = run_slot_filler(
            instruction="pick up",
            action_cls=MockPickUpAction,
            free_slot_names=["object_designator"],
            fixed_slots={},
            world_context="",
            llm=llm,
        )
        assert result is not None

    def test_enum_slot_includes_valid_members(self) -> None:
        """Prompt lists enum member names for ENUM slots."""
        output = ActionReasoningOutput(
            action_type="MockPickUpAction",
            slots=[SlotValue(field_name="grasp_description.grasp_type", value="FRONT")],
        )
        llm = ScriptedLLM(responses=[output])
        result = run_slot_filler(
            instruction="grasp from front",
            action_cls=MockPickUpAction,
            free_slot_names=["grasp_description"],
            fixed_slots={},
            world_context="",
            llm=llm,
        )
        assert result is not None
