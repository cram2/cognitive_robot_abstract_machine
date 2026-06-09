"""
Unit tests for the condition component (``grammar/conditions/``): the shared recognizers
and the comparator predicate surface form.

The recognizers are pure structural predicates (no rendering); the predicate form is
exercised end-to-end through the verbalizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from krrood.entity_query_language.factories import variable
from krrood.entity_query_language.verbalization.grammar.conditions.recognition import (
    attribute_names,
    is_bool_attr_chain,
    references,
    single_hop_attr,
)
from krrood.entity_query_language.verbalization.verbalizer import verbalize_expression


@dataclass
class _Task:
    completed: bool


@dataclass
class _Robot:
    battery: int
    tasks: List[_Task]


def test_single_hop_attr_recognizes_subject_attribute():
    r = variable(_Robot, [])
    attr = single_hop_attr(r.battery, r)
    assert attr is not None and attr._attribute_name_ == "battery"


def test_single_hop_attr_rejects_multi_hop_and_other_subject():
    r = variable(_Robot, [])
    other = variable(_Robot, [])
    assert single_hop_attr(r.tasks[0].completed, r) is None  # multi-hop
    assert single_hop_attr(r.battery, other) is None  # different subject


def test_is_bool_attr_chain_only_for_bool_terminal():
    r = variable(_Robot, [])
    assert is_bool_attr_chain(r.tasks[0].completed) is True
    assert is_bool_attr_chain(r.battery) is False


def test_attribute_names_walks_the_chain():
    r = variable(_Robot, [])
    assert attribute_names(r.battery) == ["battery"]


def test_references_detects_the_subject():
    r = variable(_Robot, [])
    assert references(r.battery, r) is True


def test_predicate_form_end_to_end():
    r = variable(_Robot, [])
    text = verbalize_expression(r.battery > 50)
    assert "battery" in text
    assert "greater than" in text
    assert "50" in text
