"""Tests for :mod:`llmr.schemas` — Pydantic LLM I/O models.

Covers :class:`EntityDescriptionSchema`, :class:`SlotValue`,
:class:`ActionReasoningOutput`, and :class:`ActionClassification`.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llmr.schemas import (
    ActionClassification,
    ActionReasoningOutput,
    EntityDescriptionSchema,
    SlotValue,
)


class TestEntityDescriptionSchema:
    """EntityDescriptionSchema Pydantic model."""

    def test_all_fields_accepted(self) -> None:
        """All fields (name, semantic_type, spatial_context, attributes) accepted."""
        schema = EntityDescriptionSchema(
            name="milk bottle",
            semantic_type="FoodItem",
            spatial_context="on the kitchen counter",
            attributes={"color": "white", "size": "medium"},
        )
        assert schema.name == "milk bottle"
        assert schema.semantic_type == "FoodItem"
        assert schema.spatial_context == "on the kitchen counter"
        assert schema.attributes == {"color": "white", "size": "medium"}

    def test_only_name_required(self) -> None:
        """Only name is required; others default to None."""
        schema = EntityDescriptionSchema(name="table")
        assert schema.name == "table"
        assert schema.semantic_type is None
        assert schema.spatial_context is None
        assert schema.attributes is None

    def test_missing_name_raises_validation_error(self) -> None:
        """Missing name field raises ValidationError."""
        with pytest.raises(ValidationError):
            EntityDescriptionSchema(semantic_type="Surface")

    def test_attributes_defaults_to_none(self) -> None:
        """Attributes field defaults to None when not provided."""
        schema = EntityDescriptionSchema(name="cup")
        assert schema.attributes is None

    def test_round_trip_json(self) -> None:
        """Schema can be serialized to JSON and reconstructed."""
        original = EntityDescriptionSchema(
            name="red cup", semantic_type="Container", attributes={"color": "red"}
        )
        json_str = original.model_dump_json()
        reconstructed = EntityDescriptionSchema.model_validate_json(json_str)
        assert reconstructed.name == original.name
        assert reconstructed.semantic_type == original.semantic_type
        assert reconstructed.attributes == original.attributes

    def test_extra_fields_ignored(self) -> None:
        """Extra fields are ignored (not rejected) by Pydantic."""
        schema = EntityDescriptionSchema(name="object", unknown_field="ignored")
        assert schema.name == "object"
        assert not hasattr(schema, "unknown_field")


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
        assert (
            output.overall_reasoning
            == "found milk and chose top grasp with left gripper"
        )


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
