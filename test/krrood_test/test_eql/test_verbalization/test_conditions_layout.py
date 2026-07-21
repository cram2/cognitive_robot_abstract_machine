"""
The condition grammar reads as cohesive, clearly-named concern units (Phase 4): the
predicate realization (``predication``), the subject slot placement (``placement``), and
the WHERE-subject resolution (``subject``).

This test pins the public surface of that layout so the names do not drift back to the
old scattered ``transforms`` / ``operator_phrase`` / ``forms`` / ``restriction`` files.
"""

from __future__ import annotations

import importlib

import pytest


def test_predication_owns_predicate_realization():
    predication = importlib.import_module(
        "krrood.entity_query_language.verbalization.grammar.conditions.predication"
    )
    for name in (
        "PredicateTransform",
        "comparator_operator",
        "coindexed_operator",
        "render_absence",
    ):
        assert hasattr(predication, name), f"predication is missing {name}"


def test_placement_owns_subject_slot_placement():
    placement = importlib.import_module(
        "krrood.entity_query_language.verbalization.grammar.conditions.placement"
    )
    for name in (
        "ConditionForm",
        "SurfacePosition",
        "Placement",
        "place",
        "as_subject_restrictions",
    ):
        assert hasattr(placement, name), f"placement is missing {name}"


def test_subject_owns_where_subject_resolution():
    subject = importlib.import_module(
        "krrood.entity_query_language.verbalization.grammar.conditions.subject"
    )
    for name in ("RestrictionSubjectRule", "restriction_subject"):
        assert hasattr(subject, name), f"subject is missing {name}"


@pytest.mark.parametrize(
    "removed_module",
    ["transforms", "operator_phrase", "forms", "restriction"],
)
def test_old_scattered_modules_are_gone(removed_module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(
            f"krrood.entity_query_language.verbalization.grammar.conditions.{removed_module}"
        )
