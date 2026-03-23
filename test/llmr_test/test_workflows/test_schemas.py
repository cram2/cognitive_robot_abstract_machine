from __future__ import annotations

import pytest
from pydantic import ValidationError

from llmr.workflows.schemas.common import EntityDescriptionSchema
from llmr.workflows.schemas.pick_up import (
    GraspParamsSchema,
    PickUpDiscreteResolutionSchema,
    PickUpSlotSchema,
)
from llmr.workflows.schemas.place import PlaceDiscreteResolutionSchema, PlaceSlotSchema
from llmr.workflows.schemas.recovery import RecoverySchema
from llmr.workflows.states.all_states import (
    DiscreteResolutionState,
    RecoveryState,
    SlotFillingState,
)


class TestEntityDescriptionSchema:
    def test_required_name(self):
        desc = EntityDescriptionSchema(name="milk")
        assert desc.name == "milk"

    def test_optional_fields_default_none(self):
        desc = EntityDescriptionSchema(name="cup")
        assert desc.semantic_type is None
        assert desc.spatial_context is None
        assert desc.attributes is None

    def test_all_fields(self):
        desc = EntityDescriptionSchema(
            name="red cup",
            semantic_type="Cup",
            spatial_context="on the table",
            attributes={"color": "red"},
        )
        assert desc.semantic_type == "Cup"
        assert desc.spatial_context == "on the table"
        assert desc.attributes == {"color": "red"}

    def test_missing_name_raises(self):
        with pytest.raises(ValidationError):
            EntityDescriptionSchema()


class TestGraspParamsSchema:
    def test_all_defaults_none(self):
        p = GraspParamsSchema()
        assert p.approach_direction is None
        assert p.vertical_alignment is None
        assert p.rotate_gripper is None

    def test_valid_approach_direction(self):
        p = GraspParamsSchema(approach_direction="FRONT")
        assert p.approach_direction == "FRONT"

    def test_invalid_approach_direction_raises(self):
        with pytest.raises(ValidationError):
            GraspParamsSchema(approach_direction="DIAGONAL")

    def test_valid_vertical_alignment(self):
        p = GraspParamsSchema(vertical_alignment="TOP")
        assert p.vertical_alignment == "TOP"

    def test_invalid_vertical_alignment_raises(self):
        with pytest.raises(ValidationError):
            GraspParamsSchema(vertical_alignment="MIDDLE")

    def test_rotate_gripper_bool(self):
        p = GraspParamsSchema(rotate_gripper=True)
        assert p.rotate_gripper is True


class TestPickUpSlotSchema:
    def test_action_type_fixed(self, entity_description):
        s = PickUpSlotSchema(object_description=entity_description)
        assert s.action_type == "PickUpAction"

    def test_action_type_cannot_override(self, entity_description):
        with pytest.raises(ValidationError):
            PickUpSlotSchema(action_type="PlaceAction", object_description=entity_description)

    def test_arm_defaults_none(self, entity_description):
        s = PickUpSlotSchema(object_description=entity_description)
        assert s.arm is None

    def test_arm_valid_values(self, entity_description):
        for arm in ("LEFT", "RIGHT", "BOTH"):
            s = PickUpSlotSchema(object_description=entity_description, arm=arm)
            assert s.arm == arm

    def test_model_dump_round_trip(self, entity_description):
        s = PickUpSlotSchema(object_description=entity_description, arm="LEFT")
        restored = PickUpSlotSchema.model_validate(s.model_dump())
        assert restored.arm == "LEFT"
        assert restored.action_type == "PickUpAction"


class TestPlaceSlotSchema:
    def test_action_type_fixed(self, entity_description):
        s = PlaceSlotSchema(
            object_description=entity_description,
            target_description=EntityDescriptionSchema(name="counter"),
        )
        assert s.action_type == "PlaceAction"

    def test_arm_defaults_none(self, entity_description):
        s = PlaceSlotSchema(
            object_description=entity_description,
            target_description=EntityDescriptionSchema(name="counter"),
        )
        assert s.arm is None

    def test_model_dump_round_trip(self, entity_description):
        s = PlaceSlotSchema(
            object_description=entity_description,
            target_description=EntityDescriptionSchema(name="counter"),
            arm="RIGHT",
        )
        restored = PlaceSlotSchema.model_validate(s.model_dump())
        assert restored.arm == "RIGHT"
        assert restored.action_type == "PlaceAction"


class TestPickUpDiscreteResolutionSchema:
    def test_all_fields(self):
        s = PickUpDiscreteResolutionSchema(
            arm="LEFT",
            approach_direction="FRONT",
            vertical_alignment="TOP",
            rotate_gripper=False,
            reasoning="Object is to the left.",
        )
        assert s.arm == "LEFT"
        assert s.approach_direction == "FRONT"
        assert s.vertical_alignment == "TOP"
        assert s.rotate_gripper is False

    def test_invalid_arm_raises(self):
        with pytest.raises(ValidationError):
            PickUpDiscreteResolutionSchema(
                arm="BOTH",
                approach_direction="FRONT",
                vertical_alignment="TOP",
                rotate_gripper=False,
                reasoning="x",
            )

    def test_invalid_approach_direction_raises(self):
        with pytest.raises(ValidationError):
            PickUpDiscreteResolutionSchema(
                arm="LEFT",
                approach_direction="UP",
                vertical_alignment="TOP",
                rotate_gripper=False,
                reasoning="x",
            )

    def test_invalid_vertical_alignment_raises(self):
        with pytest.raises(ValidationError):
            PickUpDiscreteResolutionSchema(
                arm="LEFT",
                approach_direction="FRONT",
                vertical_alignment="MIDDLE",
                rotate_gripper=False,
                reasoning="x",
            )


class TestPlaceDiscreteResolutionSchema:
    def test_construction(self):
        s = PlaceDiscreteResolutionSchema(arm="RIGHT", reasoning="Right arm is free.")
        assert s.arm == "RIGHT"

    def test_invalid_arm_raises(self):
        with pytest.raises(ValidationError):
            PlaceDiscreteResolutionSchema(arm="BOTH", reasoning="x")


class TestRecoverySchema:
    def test_replan_full_with_revised_instruction(self):
        s = RecoverySchema(
            recovery_strategy="REPLAN_FULL",
            revised_instruction="Pick up the cup with the left arm.",
            failure_diagnosis="Arm collision detected.",
            reasoning="Switch to left arm.",
        )
        assert s.recovery_strategy == "REPLAN_FULL"
        assert s.revised_instruction == "Pick up the cup with the left arm."

    def test_abort_without_revised_instruction(self):
        s = RecoverySchema(
            recovery_strategy="ABORT",
            failure_diagnosis="Object not reachable.",
            reasoning="No recovery possible.",
        )
        assert s.recovery_strategy == "ABORT"
        assert s.revised_instruction is None

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValidationError):
            RecoverySchema(
                recovery_strategy="RETRY",
                failure_diagnosis="x",
                reasoning="x",
            )


class TestStateTypedDicts:
    def test_slot_filling_state_keys(self):
        state = SlotFillingState(
            messages=[],
            instruction="pick up milk",
            world_context="",
            slot_schema=None,
            grounded_body_indices=None,
            error=None,
        )
        assert state["instruction"] == "pick up milk"
        assert state["error"] is None

    def test_discrete_resolution_state_keys(self):
        state = DiscreteResolutionState(
            messages=[],
            world_context="ctx",
            known_parameters="arm=LEFT",
            parameters_to_resolve="approach_direction",
            resolved_schema=None,
            error=None,
        )
        assert state["world_context"] == "ctx"

    def test_recovery_state_keys(self):
        state = RecoveryState(
            messages=[],
            world_context="ctx",
            original_instruction="pick up milk",
            failed_action_description="PickUpAction(arm=RIGHT)",
            error_message="IK failed",
            resolved_schema=None,
            error=None,
        )
        assert state["error_message"] == "IK failed"
