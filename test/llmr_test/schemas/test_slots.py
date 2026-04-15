"""Tests for SlotValue, ActionReasoningOutput, ActionClassification.

Pure Pydantic validation — no fixtures needed.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from llmr.schemas.entities import EntityDescriptionSchema
from llmr.schemas.slots import (
    SlotValue,
    ActionReasoningOutput,
    ActionClassification,
)


class TestSlotValue:
    """SlotValue Pydantic model — single resolved slot."""

    def test_entity_slot_with_description(self) -> None:
        """Entity slot with entity_description populated."""
        slot = SlotValue(
            field_name="object_designator",
            entity_description=EntityDescriptionSchema(name="milk"),
            reasoning="instruction mentions milk",
        )
        assert slot.field_name == "object_designator"
        assert slot.entity_description.name == "milk"
        assert slot.value is None
        assert slot.reasoning == "instruction mentions milk"

    def test_primitive_slot_with_value(self) -> None:
        """Primitive slot with value string."""
        slot = SlotValue(
            field_name="timeout", value="30.0", reasoning="reasonable timeout"
        )
        assert slot.field_name == "timeout"
        assert slot.value == "30.0"
        assert slot.entity_description is None

    def test_field_name_required(self) -> None:
        """field_name is required."""
        with pytest.raises(ValidationError):
            SlotValue(value="some_value")

    def test_value_and_entity_description_both_optional(self) -> None:
        """Both value and entity_description can be None."""
        slot = SlotValue(field_name="some_field")
        assert slot.value is None
        assert slot.entity_description is None

    def test_reasoning_defaults_to_empty_string(self) -> None:
        """reasoning field defaults to empty string."""
        slot = SlotValue(field_name="field", value="val")
        assert slot.reasoning == ""

    def test_dotted_field_name_for_complex_subfield(self) -> None:
        """Dotted field names represent complex sub-fields."""
        slot = SlotValue(
            field_name="grasp_description.grasp_type",
            value="FRONT",
            reasoning="user said front-facing grasp",
        )
        assert slot.field_name == "grasp_description.grasp_type"
        assert slot.value == "FRONT"


class TestActionReasoningOutput:
    """ActionReasoningOutput Pydantic model."""

    def test_valid_output_with_slots(self) -> None:
        """Valid output with action_type and slots list."""
        output = ActionReasoningOutput(
            action_type="PickUpAction",
            slots=[
                SlotValue(
                    field_name="object_designator",
                    entity_description=EntityDescriptionSchema(name="milk"),
                ),
                SlotValue(field_name="timeout", value="30.0"),
            ],
        )
        assert output.action_type == "PickUpAction"
        assert len(output.slots) == 2
        assert output.overall_reasoning == ""

    def test_action_type_required(self) -> None:
        """action_type is required."""
        with pytest.raises(ValidationError):
            ActionReasoningOutput(slots=[])

    def test_overall_reasoning_defaults_to_empty(self) -> None:
        """overall_reasoning defaults to empty string."""
        output = ActionReasoningOutput(action_type="NavigateAction", slots=[])
        assert output.overall_reasoning == ""

    def test_slots_can_be_empty(self) -> None:
        """slots list can be empty (though unusual)."""
        output = ActionReasoningOutput(action_type="SomeAction", slots=[])
        assert output.slots == []

    def test_complex_nested_slots(self) -> None:
        """Multiple slots including complex sub-fields."""
        output = ActionReasoningOutput(
            action_type="PickUpAction",
            slots=[
                SlotValue(
                    field_name="object_designator",
                    entity_description=EntityDescriptionSchema(name="milk"),
                ),
                SlotValue(field_name="grasp_description.grasp_type", value="TOP"),
                SlotValue(
                    field_name="grasp_description.manipulator",
                    entity_description=EntityDescriptionSchema(name="left_gripper"),
                ),
            ],
            overall_reasoning="found milk and chose top grasp with left gripper",
        )
        assert len(output.slots) == 3
        assert output.overall_reasoning == "found milk and chose top grasp with left gripper"


class TestActionClassification:
    """ActionClassification Pydantic model."""

    def test_confidence_defaults_to_one(self) -> None:
        """confidence field defaults to 1.0."""
        clf = ActionClassification(action_type="PickUpAction")
        assert clf.confidence == 1.0

    def test_confidence_clamped_between_zero_and_one(self) -> None:
        """confidence must be in [0.0, 1.0]."""
        clf = ActionClassification(action_type="X", confidence=0.75)
        assert clf.confidence == 0.75

    def test_confidence_zero_accepted(self) -> None:
        """confidence=0.0 is valid."""
        clf = ActionClassification(action_type="X", confidence=0.0)
        assert clf.confidence == 0.0

    def test_confidence_one_accepted(self) -> None:
        """confidence=1.0 is valid."""
        clf = ActionClassification(action_type="X", confidence=1.0)
        assert clf.confidence == 1.0

    def test_confidence_out_of_range_rejected(self) -> None:
        """confidence > 1.0 is rejected."""
        with pytest.raises(ValidationError):
            ActionClassification(action_type="X", confidence=1.5)

    def test_reasoning_defaults_to_empty(self) -> None:
        """reasoning field defaults to empty string."""
        clf = ActionClassification(action_type="NavigateAction")
        assert clf.reasoning == ""

    def test_action_type_required(self) -> None:
        """action_type is required."""
        with pytest.raises(ValidationError):
            ActionClassification(confidence=0.9)

    def test_full_classification_output(self) -> None:
        """Complete classification with all fields."""
        clf = ActionClassification(
            action_type="PickUpAction",
            confidence=0.95,
            reasoning="user said pick up the milk",
        )
        assert clf.action_type == "PickUpAction"
        assert clf.confidence == 0.95
        assert clf.reasoning == "user said pick up the milk"
