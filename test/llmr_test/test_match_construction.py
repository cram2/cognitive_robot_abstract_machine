"""Tests for KRROOD Match construction helpers."""
from __future__ import annotations

from dataclasses import dataclass

from krrood.entity_query_language.query.match import Match

from llmr.match_construction import required_match

from .test_actions import MockGraspDescription, MockPickUpAction


@dataclass
class MockRequiredComplexAction:
    grasp_description: MockGraspDescription


def _free_field_names(match) -> set[str]:
    return {
        attr.attribute_name
        for attr in match.matches_with_variables
        if attr.assigned_variable._value_ is ...
    }


def test_required_match_marks_required_fields_free() -> None:
    match = required_match(MockPickUpAction)

    assert _free_field_names(match) == {"object_designator"}
    assert match.kwargs == {"object_designator": ...}


def test_required_match_excludes_optional_fields() -> None:
    match = required_match(MockPickUpAction)

    assert "grasp_description" not in _free_field_names(match)
    assert "timeout" not in _free_field_names(match)


def test_required_match_skips_internal_fields() -> None:
    match = required_match(MockPickUpAction)

    assert "id" not in _free_field_names(match)
    assert "plan_node" not in _free_field_names(match)


def test_required_complex_field_is_nested_match() -> None:
    match = required_match(MockRequiredComplexAction)
    grasp_match = match.kwargs["grasp_description"]

    assert isinstance(grasp_match, Match)
    assert grasp_match.kwargs == {"grasp_type": ...}
    assert _free_field_names(match) == {"grasp_type"}
