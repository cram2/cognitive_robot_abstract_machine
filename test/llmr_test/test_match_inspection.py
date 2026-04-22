"""Tests for KRROOD Match inspection helpers."""
from __future__ import annotations

from dataclasses import dataclass

from krrood.entity_query_language.factories import variable_from
from krrood.entity_query_language.query.match import Match
from krrood.symbol_graph.symbol_graph import Symbol

from llmr.match_inspection import assigned_variable_value, match_bindings


class MockBody(Symbol):
    def __init__(self, name: str):
        self.name = name


@dataclass
class MockPickUpAction:
    object_designator: Symbol


def test_assigned_variable_value_resolves_selectable_to_first_value() -> None:
    milk = MockBody("milk")
    juice = MockBody("juice")
    unresolved = object()

    result = assigned_variable_value(variable_from([milk, juice]), unresolved)

    assert result is milk


def test_assigned_variable_value_returns_unresolved_when_selectable_is_empty() -> None:
    unresolved = object()

    result = assigned_variable_value(variable_from([]), unresolved)

    assert result is unresolved


def test_match_bindings_collects_prompt_names_and_values() -> None:
    milk = MockBody("milk")
    unresolved = object()

    bindings = match_bindings(
        Match(MockPickUpAction)(object_designator=milk),
        unresolved=unresolved,
    )

    assert len(bindings) == 1
    assert bindings[0].attribute_name == "object_designator"
    assert bindings[0].prompt_name == "object_designator"
    assert bindings[0].value is milk
    assert not bindings[0].is_free
