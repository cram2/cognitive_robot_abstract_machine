"""
Unit tests for the condition-form registry — the open-closed seam that maps a folded WHERE
conjunct to its surface form and :class:`SurfacePosition`. The registry is *total*: a standalone fallback
catches anything no specific form fits, so selection never returns ``None`` and there is no
residual special case.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.entity_query_language.factories import variable
from krrood.entity_query_language.verbalization.grammar.conditions.placement import (
    ConditionForm,
    Placement,
    SurfacePosition,
    StandaloneForm,
    SuperlativeForm,
    WhosePredicateForm,
    WhoseRangeForm,
)
from krrood.entity_query_language.verbalization.microplanning.coordination import (
    fold_range_pairs,
)


@dataclass
class _Thing:
    battery: int
    salary: int
    other: int


def test_each_form_declares_its_slot():
    """Every form fixes where its output lands — the assembler buckets by this slot."""
    assert SuperlativeForm.position is SurfacePosition.SELECTION_MODIFIER
    assert WhoseRangeForm.position is SurfacePosition.WHOSE
    assert WhosePredicateForm.position is SurfacePosition.WHOSE
    assert StandaloneForm.position is SurfacePosition.STANDALONE


def test_attribute_predicate_selects_the_whose_form():
    """A single-hop, non-boolean attribute comparator → the whose-predicate form (slot WHOSE)."""
    thing = variable(_Thing, [])
    form = ConditionForm.most_applicable(
        Placement(item=thing.battery > 50, subject=thing)
    )
    assert form is WhosePredicateForm
    assert form.position is SurfacePosition.WHOSE


def test_range_fold_selects_the_whose_range_form():
    """A folded lower/upper bound pair on a subject attribute → the whose-range form (slot WHOSE)."""
    thing = variable(_Thing, [])
    (folded,) = fold_range_pairs([thing.salary > 100, thing.salary < 200])
    form = ConditionForm.most_applicable(Placement(item=folded, subject=thing))
    assert form is WhoseRangeForm
    assert form.position is SurfacePosition.WHOSE


def test_non_foldable_conjunct_falls_back_to_standalone():
    """A conjunct that fits no specific form (here: its right side references the subject) falls to
    the standalone fallback — the registry's totality, no ``None``, no residual special case.
    """
    thing = variable(_Thing, [])
    form = ConditionForm.most_applicable(
        Placement(item=thing.battery > thing.salary, subject=thing)
    )
    assert form is StandaloneForm
    assert form.position is SurfacePosition.STANDALONE


def test_selection_is_total():
    """Even an unrelated comparator selects a form (the standalone fallback) — never ``None``."""
    thing = variable(_Thing, [])
    other = variable(_Thing, [])
    form = ConditionForm.most_applicable(
        Placement(item=other.battery > 1, subject=thing)
    )
    assert form is StandaloneForm
