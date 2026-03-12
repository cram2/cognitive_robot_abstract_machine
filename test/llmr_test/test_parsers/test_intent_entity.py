"""Tests for the ReflectiveParser intent and entity parsing agent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.runnables import RunnableLambda

from llmr.workflows.parsers.intent_entity import ReflectiveParser, _INTENT_REQUIRED_ROLES
from llmr.workflows.models.intent_entity_models import (
    Instruction,
    InstructionList,
    IntentType,
    Metadata,
    Roles,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_valid_instruction(
    intent: IntentType = IntentType.PICK,
    action_id: str = "action_abc12345",
) -> Instruction:
    """Create a valid instruction with all required roles for the given intent."""
    roles = Roles(patient="cup", destination_location="table")
    return Instruction(
        intent=intent,
        atomic_instruction="pick up the cup",
        action_id=action_id,
        roles=roles,
        metadata=Metadata(confidence=0.9),
    )


def _make_instruction_bypass_validation(**kwargs) -> Instruction:
    """Create an Instruction bypassing model validators (for testing edge cases)."""
    defaults = {
        "action_id": "action_abc12345",
        "intent": IntentType.PICK,
        "atomic_instruction": "pick up the cup",
        "roles": Roles(),
        "metadata": Metadata(),
    }
    defaults.update(kwargs)
    return Instruction.model_construct(**defaults)


def _make_parser_with_mock_llm(enable_reflection: bool = False) -> ReflectiveParser:
    """Create a ReflectiveParser with a mocked LLM."""
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = MagicMock()
    with patch(
        "llmr.workflows.parsers.intent_entity.ReflectiveParser._build_llm",
        return_value=mock_llm,
    ):
        return ReflectiveParser(enable_reflection=enable_reflection)


# ── _assign_action_id ─────────────────────────────────────────────────────────

class TestAssignActionId:
    def test_assigns_id_when_none(self) -> None:
        inst = _make_instruction_bypass_validation(action_id=None)
        result = ReflectiveParser._assign_action_id(inst)
        assert result.action_id is not None
        assert result.action_id.startswith("action_")

    def test_assigns_id_when_starts_with_capital_A(self) -> None:
        inst = _make_instruction_bypass_validation(action_id="A001")
        result = ReflectiveParser._assign_action_id(inst)
        assert result.action_id.startswith("action_")
        assert not result.action_id.startswith("A0")

    def test_preserves_valid_id(self) -> None:
        inst = _make_instruction_bypass_validation(action_id="action_custom123")
        result = ReflectiveParser._assign_action_id(inst)
        assert result.action_id == "action_custom123"

    def test_generated_id_has_correct_format(self) -> None:
        inst = _make_instruction_bypass_validation(action_id=None)
        result = ReflectiveParser._assign_action_id(inst)
        # format: "action_" + 8 hex chars = 15 chars total
        assert len(result.action_id) == len("action_") + 8

    def test_empty_string_id_gets_replaced(self) -> None:
        inst = _make_instruction_bypass_validation(action_id="")
        result = ReflectiveParser._assign_action_id(inst)
        assert result.action_id.startswith("action_")


# ── _check_missing_roles ──────────────────────────────────────────────────────

class TestCheckMissingRoles:
    def setup_method(self) -> None:
        with patch.object(ReflectiveParser, "__init__", lambda self, *a, **kw: None):
            self.parser = ReflectiveParser.__new__(ReflectiveParser)

    def test_no_missing_roles_when_all_present(self) -> None:
        inst = _make_instruction_bypass_validation(
            intent=IntentType.POUR,
            roles=Roles(patient="water", destination_location="glass"),
        )
        assert self.parser._check_missing_roles(inst) == []

    def test_detects_missing_patient_for_pick(self) -> None:
        inst = _make_instruction_bypass_validation(
            intent=IntentType.PICK,
            roles=Roles(patient=None),
        )
        missing = self.parser._check_missing_roles(inst)
        assert "patient" in missing

    def test_detects_missing_destination_for_pour(self) -> None:
        inst = _make_instruction_bypass_validation(
            intent=IntentType.POUR,
            roles=Roles(patient="water", destination_location=None),
        )
        missing = self.parser._check_missing_roles(inst)
        assert "destination_location" in missing

    def test_unknown_intent_has_no_required_roles(self) -> None:
        inst = _make_instruction_bypass_validation(
            intent=IntentType.STIR,
            roles=Roles(),
        )
        assert self.parser._check_missing_roles(inst) == []

    def test_all_required_roles_present_returns_empty(self) -> None:
        inst = _make_instruction_bypass_validation(
            intent=IntentType.PLACE,
            roles=Roles(patient="cup", destination_location="table"),
        )
        assert self.parser._check_missing_roles(inst) == []


# ── _append_reflection_feedback ───────────────────────────────────────────────

class TestAppendReflectionFeedback:
    def setup_method(self) -> None:
        with patch.object(ReflectiveParser, "__init__", lambda self, *a, **kw: None):
            self.parser = ReflectiveParser.__new__(ReflectiveParser)

    def test_appends_feedback_to_existing_comments(self) -> None:
        inst = _make_valid_instruction()
        inst.metadata.comments = "initial comment"
        result = self.parser._append_reflection_feedback(inst, ["missing patient"])
        assert "REFLECTION FEEDBACK" in result.metadata.comments
        assert "missing patient" in result.metadata.comments
        assert "initial comment" in result.metadata.comments

    def test_creates_metadata_when_none(self) -> None:
        inst = _make_instruction_bypass_validation(metadata=None)
        result = self.parser._append_reflection_feedback(inst, ["missing role"])
        assert result.metadata is not None
        assert "missing role" in result.metadata.comments

    def test_multiple_feedback_items_joined_by_semicolon(self) -> None:
        inst = _make_valid_instruction()
        inst.metadata = Metadata()
        result = self.parser._append_reflection_feedback(
            inst, ["missing patient", "low confidence"]
        )
        assert "missing patient" in result.metadata.comments
        assert "low confidence" in result.metadata.comments


# ── reflect ───────────────────────────────────────────────────────────────────

class TestReflect:
    def setup_method(self) -> None:
        with patch.object(ReflectiveParser, "__init__", lambda self, *a, **kw: None):
            self.parser = ReflectiveParser.__new__(ReflectiveParser)

    def test_no_feedback_when_roles_complete(self) -> None:
        # POUR requires patient + destination_location — both present in _make_valid_instruction
        inst = _make_valid_instruction(intent=IntentType.POUR)
        result = self.parser.reflect(InstructionList(instructions=[inst]))
        comments = result.instructions[0].metadata.comments or ""
        assert "REFLECTION FEEDBACK" not in comments

    def test_returns_instruction_list(self) -> None:
        inst = _make_valid_instruction()
        result = self.parser.reflect(InstructionList(instructions=[inst]))
        assert isinstance(result, InstructionList)

    def test_all_instructions_returned(self) -> None:
        insts = [
            _make_valid_instruction(action_id="action_aaa11111"),
            _make_valid_instruction(action_id="action_bbb22222"),
        ]
        result = self.parser.reflect(InstructionList(instructions=insts))
        assert len(result.instructions) == 2

    def test_reflect_preserves_metadata_when_no_issues(self) -> None:
        inst = _make_valid_instruction()
        inst.metadata.comments = "existing note"
        result = self.parser.reflect(InstructionList(instructions=[inst]))
        assert result.instructions[0].metadata.comments == "existing note"


# ── parse_initial ─────────────────────────────────────────────────────────────

class TestParseInitial:
    def test_returns_instruction_list(self) -> None:
        parser = _make_parser_with_mock_llm()
        inst = _make_valid_instruction()
        instruction_list = InstructionList(instructions=[inst])
        # Use RunnableLambda so coerce_to_runnable treats it as a proper Runnable
        parser.structured_output_llm = RunnableLambda(lambda _: instruction_list)
        result = parser.parse_initial("pick up the cup")
        assert isinstance(result, InstructionList)
        assert len(result.instructions) == 1

    def test_assigns_action_ids_to_all(self) -> None:
        parser = _make_parser_with_mock_llm()
        # Use valid instruction with action_id starting with "A" — will be replaced
        inst = Instruction(
            intent=IntentType.PICK,
            atomic_instruction="pick up the cup",
            action_id="A001",
            roles=Roles(patient="cup"),
        )
        parser.structured_output_llm = RunnableLambda(
            lambda _: InstructionList(instructions=[inst])
        )
        result = parser.parse_initial("pick up the cup")
        assert result.instructions[0].action_id.startswith("action_")

    def test_raises_runtime_error_on_llm_failure(self) -> None:
        parser = _make_parser_with_mock_llm()

        def _raise(_):
            raise Exception("LLM error")

        parser.structured_output_llm = RunnableLambda(_raise)
        with pytest.raises(RuntimeError, match="Parsing failed"):
            parser.parse_initial("pick up the cup")


# ── reiterate ─────────────────────────────────────────────────────────────────

class TestReiterate:
    def test_returns_instruction_list(self) -> None:
        parser = _make_parser_with_mock_llm()
        inst = _make_valid_instruction()
        instruction_list = InstructionList(instructions=[inst])
        # reiterate calls self.structured_output_llm.invoke(prompt_str) directly
        mock_so_llm = MagicMock()
        mock_so_llm.invoke.return_value = instruction_list
        parser.structured_output_llm = mock_so_llm
        result = parser.reiterate("pick up the cup", instruction_list)
        assert isinstance(result, InstructionList)

    def test_calls_structured_output_llm_with_feedback_prompt(self) -> None:
        parser = _make_parser_with_mock_llm()
        inst = _make_valid_instruction()
        instruction_list = InstructionList(instructions=[inst])
        mock_so_llm = MagicMock()
        mock_so_llm.invoke.return_value = instruction_list
        parser.structured_output_llm = mock_so_llm
        parser.reiterate("original instruction", instruction_list)
        mock_so_llm.invoke.assert_called_once()
        prompt_arg = mock_so_llm.invoke.call_args[0][0]
        assert "original instruction" in prompt_arg


# ── parse ─────────────────────────────────────────────────────────────────────

class TestParse:
    def test_returns_instructions_key(self) -> None:
        parser = _make_parser_with_mock_llm()
        inst = _make_valid_instruction()
        instruction_list = InstructionList(instructions=[inst])
        parser.structured_output_llm = RunnableLambda(lambda _: instruction_list)
        result = parser.parse("pick up the cup")
        assert "instructions" in result

    def test_returns_error_key_on_llm_failure(self) -> None:
        parser = _make_parser_with_mock_llm()

        def _raise(_):
            raise Exception("network error")

        parser.structured_output_llm = RunnableLambda(_raise)
        result = parser.parse("pick up the cup")
        assert "error" in result
        assert result["instructions"] == []

    def test_reiterate_called_when_reflection_enabled(self) -> None:
        parser = _make_parser_with_mock_llm(enable_reflection=True)
        inst = _make_valid_instruction()
        instruction_list = InstructionList(instructions=[inst])
        parser.structured_output_llm = RunnableLambda(lambda _: instruction_list)
        with patch.object(parser, "reiterate", return_value=instruction_list) as mock_reiterate:
            parser.parse("pick up the cup")
        mock_reiterate.assert_called_once()

    def test_reiterate_not_called_when_reflection_disabled(self) -> None:
        parser = _make_parser_with_mock_llm(enable_reflection=False)
        inst = _make_valid_instruction()
        instruction_list = InstructionList(instructions=[inst])
        parser.structured_output_llm = RunnableLambda(lambda _: instruction_list)
        with patch.object(parser, "reiterate") as mock_reiterate:
            parser.parse("pick up the cup")
        mock_reiterate.assert_not_called()

    def test_instructions_are_serialized_as_dicts(self) -> None:
        parser = _make_parser_with_mock_llm()
        inst = _make_valid_instruction()
        instruction_list = InstructionList(instructions=[inst])
        parser.structured_output_llm = RunnableLambda(lambda _: instruction_list)
        result = parser.parse("pick up the cup")
        assert isinstance(result["instructions"][0], dict)


# ── _INTENT_REQUIRED_ROLES constant ──────────────────────────────────────────

class TestIntentRequiredRoles:
    def test_pour_requires_patient_and_destination(self) -> None:
        assert "patient" in _INTENT_REQUIRED_ROLES[IntentType.POUR]
        assert "destination_location" in _INTENT_REQUIRED_ROLES[IntentType.POUR]

    def test_pick_requires_only_patient(self) -> None:
        assert _INTENT_REQUIRED_ROLES[IntentType.PICK] == ["patient"]

    def test_place_requires_patient_and_destination(self) -> None:
        assert "patient" in _INTENT_REQUIRED_ROLES[IntentType.PLACE]
        assert "destination_location" in _INTENT_REQUIRED_ROLES[IntentType.PLACE]

    def test_stir_not_in_required_roles(self) -> None:
        assert IntentType.STIR not in _INTENT_REQUIRED_ROLES
