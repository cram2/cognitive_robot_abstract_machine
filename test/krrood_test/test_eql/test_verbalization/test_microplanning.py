"""
Standalone unit tests for the microplanning services and the coordination module.

Each service is exercised in isolation (no verbalizer pipeline) so that the
single-responsibility behaviour split out of the former god-context is pinned
directly.
"""

from __future__ import annotations

import pytest

from krrood.entity_query_language.factories import variable
from krrood.entity_query_language.verbalization.fragments.base import (
    flatten_fragment_to_plain_text,
    WordFragment,
)
from krrood.entity_query_language.verbalization.microplanning.binding_scope import (
    BindingScope,
)
from krrood.entity_query_language.verbalization.microplanning.config import RenderConfig
from krrood.entity_query_language.verbalization.microplanning.coordination import (
    build_between,
    has_pair,
)
from krrood.entity_query_language.verbalization.fragments.features import Definiteness
from krrood.entity_query_language.verbalization.microplanning.referring import (
    ReferringExpressions,
)
from krrood.entity_query_language.verbalization.rendering.determiner_processor import (
    DeterminerProcessor,
)
from krrood.entity_query_language.verbalization.vocabulary.english import RangePhrases


class Robot:
    """Minimal type used to give test variables a clean type name."""


# ── ReferringExpressions ─────────────────────────────────────────────────────


def test_noun_for_parts_first_then_subsequent_mention():
    refer = ReferringExpressions()
    var = variable(Robot, domain=[])

    definiteness, label = refer.noun_for_parts(var)
    assert (definiteness, label) == (Definiteness.INDEFINITE, "Robot")

    # Second mention of the same variable becomes definite.
    definiteness, label = refer.noun_for_parts(var)
    assert (definiteness, label) == (Definiteness.DEFINITE, "Robot")


def test_noun_for_parts_numbered_variable_takes_no_article():
    var = variable(Robot, domain=[])
    refer = ReferringExpressions(disambiguation_map={var._id_: "Robot 2"})

    definiteness, label = refer.noun_for_parts(var)
    assert (definiteness, label) == (Definiteness.BARE, "Robot 2")


def test_seen_reference_is_none_until_mentioned_then_definite():
    refer = ReferringExpressions()
    var = variable(Robot, domain=[])

    assert refer.seen_reference(var) is None
    refer.noun_for_parts(var)  # records the mention
    # seen_reference returns a NounPhrase spec; the determiner phase realises "the Robot"
    reference = DeterminerProcessor().process(refer.seen_reference(var))
    assert flatten_fragment_to_plain_text(reference) == "the Robot"


def test_pronoun_only_for_current_unnumbered_seen_subject():
    refer = ReferringExpressions()
    var = variable(Robot, domain=[])

    # Not yet a subject / not yet seen → no pronoun.
    assert refer.pronoun_for(var) is None

    refer.noun_for_parts(var)  # mark seen
    refer.push_subject(var)  # make it the current subject
    assert flatten_fragment_to_plain_text(refer.pronoun_for(var)) == "its"

    refer.pop_subject()
    assert refer.pronoun_for(var) is None


def test_pronoun_suppressed_for_numbered_variable():
    var = variable(Robot, domain=[])
    refer = ReferringExpressions(disambiguation_map={var._id_: "Robot 2"})
    refer.noun_for_parts(var)
    refer.push_subject(var)
    assert refer.pronoun_for(var) is None


# ── BindingScope ─────────────────────────────────────────────────────────────


def test_binding_scope_frame_collects_deferred_in_order():
    scope = BindingScope()
    scope.push_constraint_frame()
    scope.defer_constraint("a")
    scope.defer_constraint("b")
    assert scope.pop_constraint_frame() == ["a", "b"]


def test_binding_scope_defer_without_frame_is_noop():
    scope = BindingScope()
    scope.defer_constraint("a")  # no frame open
    assert scope.pop_constraint_frame() == []


def test_binding_scope_frames_nest():
    scope = BindingScope()
    scope.push_constraint_frame()
    scope.defer_constraint("outer")
    scope.push_constraint_frame()
    scope.defer_constraint("inner")
    assert scope.pop_constraint_frame() == ["inner"]
    assert scope.pop_constraint_frame() == ["outer"]


# ── RenderConfig ─────────────────────────────────────────────────────────────


def test_query_depth_scope_increments_and_restores():
    config = RenderConfig()
    assert config.query_depth == 0
    with config.query_depth_scope():
        assert config.query_depth == 1
        with config.query_depth_scope():
            assert config.query_depth == 2
        assert config.query_depth == 1
    assert config.query_depth == 0


def test_compact_predicates_scope_restores_previous_even_nested():
    config = RenderConfig()
    assert config.compact_predicates is False
    with config.compact_predicates_scope():
        assert config.compact_predicates is True
        with config.compact_predicates_scope():
            assert config.compact_predicates is True
        assert config.compact_predicates is True
    assert config.compact_predicates is False


# ── Coordination ─────────────────────────────────────────────────────────────


def test_build_between_standard_and_compact_forms():
    left, lo, hi = WordFragment("x"), WordFragment("1"), WordFragment("10")

    standard = build_between(left, lo, hi, compact=False)
    assert (
        flatten_fragment_to_plain_text(standard)
        == f"x {RangePhrases.IS_BETWEEN.text} 1, and 10"
    )

    compact = build_between(left, lo, hi, compact=True)
    assert (
        flatten_fragment_to_plain_text(compact)
        == f"x {RangePhrases.BETWEEN.text} 1, and 10"
    )


def test_has_pair_false_without_complementary_bounds():
    assert has_pair([]) is False
    assert has_pair([WordFragment("not a comparator")]) is False
