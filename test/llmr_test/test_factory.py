"""Tests for factory functions — nl_plan, nl_sequential, resolve_match, resolve_params.

Uses ScriptedLLM with pre-built responses. Real SymbolGraph cleared via autouse.
"""
from __future__ import annotations

import pytest

from .scripted_llm import ScriptedLLM
from .test_actions import MockPickUpAction

from llmr.factory import (
    _fully_underspecified,
    _get_required_schema_fields,
    _get_settable_fields,
    resolve_params,
)
from llmr.exceptions import LLMUnresolvedRequiredFields
from llmr.schemas.slots import ActionReasoningOutput, SlotValue
from krrood.entity_query_language.query.match import Match
from krrood.symbol_graph.symbol_graph import Symbol


class MockBody(Symbol):
    def __init__(self, name: str):
        self.name = name


def _free_field_names(match) -> set[str]:
    return {
        attr.attribute_name
        for attr in match.matches_with_variables
        if attr.assigned_variable._value_ is ...
    }


class TestFullyUnderspecified:
    """_fully_underspecified() — Match construction with free slots."""

    def test_returns_match_with_free_required_fields(self) -> None:
        """Returns Match with required fields set to ...."""
        match = _fully_underspecified(MockPickUpAction)
        assert _free_field_names(match) == {"object_designator"}

    def test_match_with_all_required_fields(self) -> None:
        """All required fields are marked as free."""
        match = _fully_underspecified(MockPickUpAction)
        assert match.kwargs == {"object_designator": ...}

    def test_optional_fields_excluded_from_match(self) -> None:
        """Optional fields are not included in the Match."""
        match = _fully_underspecified(MockPickUpAction)
        assert "grasp_description" not in _free_field_names(match)
        assert "timeout" not in _free_field_names(match)

    def test_internal_fields_skipped(self) -> None:
        """Internal fields (id, plan_node) are skipped."""
        match = _fully_underspecified(MockPickUpAction)
        assert "id" not in _free_field_names(match)
        assert "plan_node" not in _free_field_names(match)


class TestGetRequiredSchemaFields:
    """_get_required_schema_fields() — introspection-based required field discovery."""

    def test_returns_only_non_optional_fields(self) -> None:
        """Returns list of required (non-optional) field names."""
        fields = _get_required_schema_fields(MockPickUpAction)
        assert fields == ["object_designator"]

    def test_returns_none_on_introspection_failure(self) -> None:
        """Returns None if introspection fails."""
        # Pass a non-dataclass to trigger failure
        result = _get_required_schema_fields(dict)
        assert result is None

    def test_skips_internal_fields(self) -> None:
        """Internal fields (id, plan_node) are skipped."""
        fields = _get_required_schema_fields(MockPickUpAction)
        assert fields == ["object_designator"]

    def test_uses_introspector_and_excludes_optional_fields(self) -> None:
        """Uses PycramIntrospector and excludes optional fields from required list."""
        fields = _get_required_schema_fields(MockPickUpAction)
        assert fields is not None
        assert "object_designator" in fields
        # grasp_description and timeout are optional — must not appear
        assert "grasp_description" not in fields
        assert "timeout" not in fields


class TestGetSettableFields:
    """_get_settable_fields() — field discovery fallback."""

    def test_returns_all_non_underscore_fields(self) -> None:
        """Returns all public (non-underscore) fields."""
        fields = _get_settable_fields(MockPickUpAction)
        # Should include all public fields
        assert len(fields) > 0
        assert all(not f.startswith("_") for f in fields)

    def test_skips_id_and_plan_node(self) -> None:
        """id and plan_node are skipped."""
        fields = _get_settable_fields(MockPickUpAction)
        assert "id" not in fields
        assert "plan_node" not in fields

    def test_handles_non_dataclass(self) -> None:
        """Handles non-dataclass objects with __init__ signature."""
        fields = _get_settable_fields(dict)
        # dict should return empty or valid signature params
        assert isinstance(fields, list)


class TestResolveParams:
    """resolve_params() — standalone parameter resolution (no context/execution)."""

    def test_returns_concrete_action_instance(self) -> None:
        """resolve_params returns a concrete action instance."""
        output = ActionReasoningOutput(
            action_type="MockPickUpAction",
            slots=[
                SlotValue(field_name="timeout", value="12.5")
            ],
        )
        llm = ScriptedLLM(responses=[output])
        milk = MockBody("milk")

        match = Match(MockPickUpAction)(object_designator=milk, timeout=...)

        result = resolve_params(match, llm=llm, strict_required=False)

        assert result == MockPickUpAction(object_designator=milk, timeout=12.5)

    def test_does_not_require_context(self) -> None:
        """resolve_params does not require a PyCRAM Context."""
        output = ActionReasoningOutput(action_type="MockPickUpAction", slots=[])
        llm = ScriptedLLM(responses=[output])
        milk = MockBody("milk")

        match = Match(MockPickUpAction)(object_designator=milk)

        result = resolve_params(match, llm=llm)

        assert result == MockPickUpAction(object_designator=milk)

    def test_accepts_custom_instructions(self) -> None:
        """resolve_params accepts instruction parameter for context."""
        output = ActionReasoningOutput(action_type="MockPickUpAction", slots=[])
        llm = ScriptedLLM(responses=[output])
        milk = MockBody("milk")

        match = Match(MockPickUpAction)(object_designator=milk)

        result = resolve_params(
            match,
            llm=llm,
            instruction="pick up the milk",
        )

        assert result == MockPickUpAction(object_designator=milk)

    def test_accepts_groundable_type(self) -> None:
        """resolve_params accepts groundable_type parameter."""
        output = ActionReasoningOutput(action_type="MockPickUpAction", slots=[])
        llm = ScriptedLLM(responses=[output])
        milk = MockBody("milk")

        match = Match(MockPickUpAction)(object_designator=milk)

        result = resolve_params(
            match,
            llm=llm,
            groundable_type=Symbol,
        )

        assert result == MockPickUpAction(object_designator=milk)

    def test_strict_required_mode(self) -> None:
        """resolve_params respects strict_required parameter."""
        output = ActionReasoningOutput(action_type="MockPickUpAction", slots=[])
        llm = ScriptedLLM(responses=[output])

        match = Match(MockPickUpAction)(object_designator=...)

        with pytest.raises(LLMUnresolvedRequiredFields) as exc_info:
            resolve_params(
                match,
                llm=llm,
                strict_required=True,
            )

        assert exc_info.value.unresolved_fields == ["object_designator"]


class TestMatchConstruction:
    """Basic Match construction for action classes."""

    def test_match_for_simple_action(self) -> None:
        """Can construct a Match for a simple action."""
        match = Match(MockPickUpAction)
        assert match is not None

    def test_match_with_free_fields(self) -> None:
        """Can construct a Match with free fields (...)."""
        match = Match(MockPickUpAction)(object_designator=...)
        assert _free_field_names(match) == {"object_designator"}

    def test_match_with_fixed_slots(self) -> None:
        """Can construct a Match with fixed slot values."""
        from test.llmr_test.test_actions import MockGraspDescription, GraspType

        grasp = MockGraspDescription(grasp_type=GraspType.FRONT)
        match = Match(MockPickUpAction)(grasp_description=grasp)
        assert match.kwargs == {"grasp_description": grasp}
