"""Tests for intent/entity Pydantic models."""

import pytest
from pydantic import ValidationError

from llmr.workflows.models.intent_entity_models import (
    Instruction,
    InstructionList,
    IntentType,
    Metadata,
    Roles,
)


def _make_instruction(**kwargs) -> dict:
    base = {
        "action_id": "action_abc12345",
        "intent": IntentType.PICK,
        "roles": {"patient": "apple"},
    }
    base.update(kwargs)
    return base


class TestIntentType:
    def test_all_values(self) -> None:
        assert IntentType.POUR == "Pouring"
        assert IntentType.CUT == "Cutting"
        assert IntentType.PICK == "PickingUp"
        assert IntentType.PLACE == "Placing"
        assert IntentType.OPEN == "Opening"
        assert IntentType.CLOSE == "Closing"
        assert IntentType.PULL == "Pulling"
        assert IntentType.STIR == "Stirring"
        assert IntentType.MIX == "Mixing"
        assert IntentType.COOL == "Cooling"


class TestMetadataConfidenceValidator:
    def test_normalises_percentage_confidence(self) -> None:
        m = Metadata(confidence=85.0)
        assert m.confidence == pytest.approx(0.85)

    def test_clamps_negative_confidence_to_zero(self) -> None:
        m = Metadata(confidence=-0.5)
        assert m.confidence == pytest.approx(0.0)

    def test_valid_confidence_unchanged(self) -> None:
        m = Metadata(confidence=0.75)
        assert m.confidence == pytest.approx(0.75)

    def test_none_confidence_stays_none(self) -> None:
        m = Metadata(confidence=None)
        assert m.confidence is None

    def test_confidence_clamped_above_one_after_normalisation(self) -> None:
        # 200 -> 2.0 after /100, then clamped to 1.0
        m = Metadata(confidence=200.0)
        assert m.confidence == pytest.approx(1.0)


class TestInstructionRoleValidation:
    def test_pour_requires_patient_and_destination(self) -> None:
        with pytest.raises(ValidationError):
            Instruction(
                **_make_instruction(
                    intent=IntentType.POUR,
                    roles={"patient": "water"},  # missing destination_location
                )
            )

    def test_pick_requires_patient(self) -> None:
        with pytest.raises(ValidationError):
            Instruction(
                **_make_instruction(
                    intent=IntentType.PICK,
                    roles={},  # missing patient
                )
            )

    def test_valid_pick_instruction(self) -> None:
        inst = Instruction(
            **_make_instruction(
                intent=IntentType.PICK,
                roles={"patient": "apple"},
            )
        )
        assert inst.roles.patient == "apple"

    def test_valid_pour_instruction(self) -> None:
        inst = Instruction(
            **_make_instruction(
                intent=IntentType.POUR,
                roles={"patient": "water", "destination_location": "glass"},
            )
        )
        assert inst.roles.destination_location == "glass"

    def test_stir_intent_no_required_roles(self) -> None:
        inst = Instruction(
            **_make_instruction(
                intent=IntentType.STIR,
                roles={},
            )
        )
        assert inst.intent == IntentType.STIR


class TestInstructionList:
    def test_empty_list(self) -> None:
        il = InstructionList(instructions=[])
        assert il.instructions == []

    def test_multiple_instructions(self) -> None:
        inst_data = _make_instruction()
        il = InstructionList(instructions=[inst_data])
        assert len(il.instructions) == 1
