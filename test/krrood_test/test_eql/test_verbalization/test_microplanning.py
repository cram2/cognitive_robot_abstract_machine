"""
Standalone unit tests for the microplanning services and the coordination module.

Each service is exercised in isolation (no verbalizer pipeline) so that the single-
responsibility behaviour split out of the former god-context is pinned directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from krrood.entity_query_language.factories import variable
from krrood.entity_query_language.verbalization.fragments.base import (
    flatten_fragment_to_plain_text,
    WordFragment,
)
from krrood.entity_query_language.verbalization.microplanning.binding_scope import (
    BindingScope,
)
from krrood.entity_query_language.verbalization.microplanning.config import (
    RenderConfiguration,
)
from krrood.entity_query_language.verbalization.microplanning.coordination import (
    build_between,
    CoindexedFold,
    coindexed_natural_parts,
    coindexed_signature,
    fold_coindexed_groups,
    fold_range_pairs,
    has_pair,
    reduce_conjuncts,
)
from krrood.entity_query_language.verbalization.fragments.features import Definiteness
from krrood.entity_query_language.verbalization.microplanning.referring import (
    ReferringExpressions,
)
from krrood.entity_query_language.verbalization.vocabulary.english import RangePhrases


class Robot:
    """
    Minimal type used to give test variables a clean type name.
    """


# ── ReferringExpressions ─────────────────────────────────────────────────────


def test_noun_for_parts_returns_first_mention_form_and_records():
    refer = ReferringExpressions()
    var = variable(Robot, domain=[])

    noun_form = refer.noun_for_parts(var)
    assert (noun_form.definiteness, noun_form.label) == (
        Definiteness.INDEFINITE,
        "Robot",
    )
    assert var._id_ in refer.seen  # mention recorded for the build-time services

    # The subsequent-mention downgrade now lives in the CoreferenceProcessor, so
    # noun_for_parts keeps returning the first-mention (indefinite) form.
    noun_form = refer.noun_for_parts(var)
    assert (noun_form.definiteness, noun_form.label) == (
        Definiteness.INDEFINITE,
        "Robot",
    )


def test_noun_for_parts_numbered_variable_takes_no_article():
    var = variable(Robot, domain=[])
    refer = ReferringExpressions(disambiguation_map={var._id_: "Robot 2"})

    noun_form = refer.noun_for_parts(var)
    assert (noun_form.definiteness, noun_form.label) == (Definiteness.BARE, "Robot 2")


def test_noun_for_parts_records_the_mention():
    refer = ReferringExpressions()
    var = variable(Robot, domain=[])
    assert var._id_ not in refer.seen
    refer.noun_for_parts(var)  # records the mention (for cross-build seeding)
    assert var._id_ in refer.seen


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


# ── RenderConfiguration ─────────────────────────────────────────────────────────────


def test_query_depth_scope_increments_and_restores():
    configuration = RenderConfiguration()
    assert configuration.query_depth == 0
    with configuration.query_depth_scope():
        assert configuration.query_depth == 1
        with configuration.query_depth_scope():
            assert configuration.query_depth == 2
        assert configuration.query_depth == 1
    assert configuration.query_depth == 0


# ── Coordination ─────────────────────────────────────────────────────────────


def test_build_between_standard_and_compact_forms():
    left, lo, hi = WordFragment("x"), WordFragment("1"), WordFragment("10")

    standard = build_between(left, lo, hi, compact=False)
    assert (
        flatten_fragment_to_plain_text(standard)
        == f"x is {RangePhrases.BETWEEN.text} 1 and 10"
    )

    compact = build_between(left, lo, hi, compact=True)
    assert (
        flatten_fragment_to_plain_text(compact)
        == f"x {RangePhrases.BETWEEN.text} 1 and 10"
    )


def test_has_pair_false_without_complementary_bounds():
    assert has_pair([]) is False
    assert has_pair([WordFragment("not a comparator")]) is False


# ── Co-indexed folding ───────────────────────────────────────────────────────


@dataclass
class _Date:
    month: int
    year: int
    day: int


@dataclass
class _Period:
    begin: _Date
    end: _Date


@dataclass
class _Inner:
    z: _Date


@dataclass
class _Outer:
    x: _Date
    y: _Inner


def _period() -> "object":
    return variable(_Period, domain=None)


def test_coindexed_signature_groups_same_prefixes_and_operator():
    p = _period()
    sig_month, term_month = coindexed_signature(p.begin.month == p.end.month)
    sig_year, term_year = coindexed_signature(p.begin.year == p.end.year)
    assert sig_month == sig_year  # same prefixes + operator → one group
    assert term_month == ("month", _Date) and term_year == ("year", _Date)


def test_coindexed_signature_rejects_non_foldable_and_mismatches():
    p = _period()
    assert coindexed_signature(p.begin.month != p.end.month) is None  # ne not foldable
    assert coindexed_signature(p.begin.month == p.end.year) is None  # leaves differ
    assert coindexed_signature(p.begin.month == 5) is None  # right side not a chain


def test_fold_coindexed_groups_reduces_group_to_one_fold():
    p = _period()
    items = [p.begin.month == p.end.month, p.begin.year == p.end.year]
    folded = fold_coindexed_groups(items)
    assert len(folded) == 1
    [fold] = folded
    assert isinstance(fold, CoindexedFold)
    assert fold.terminals == [("month", _Date), ("year", _Date)]


def test_fold_coindexed_groups_preserves_order_of_unrelated_conjuncts():
    p = _period()
    unrelated = p.begin.day == 1  # right side is a literal → never folds
    items = [p.begin.month == p.end.month, unrelated, p.begin.year == p.end.year]
    folded = fold_coindexed_groups(items)
    # The fold sits at the group's first index; the unrelated conjunct keeps its place.
    assert isinstance(folded[0], CoindexedFold)
    assert folded[1] is unrelated
    assert len(folded) == 2


def test_lone_co_indexed_comparison_is_not_folded():
    p = _period()
    items = [p.begin.month == p.end.month]
    assert fold_coindexed_groups(items) == items  # a group of one says itself


def test_reduce_conjuncts_is_a_no_op_without_a_co_indexed_group():
    p = _period()
    corpus = [
        [],
        [p.begin.month == p.end.month],  # lone co-indexed comparison
        [p.begin.month != p.end.month, p.begin.year != p.end.year],  # ne excluded
        [p.begin.month == p.end.month, p.begin.year > p.end.year],  # mixed operators
    ]
    for conjuncts in corpus:
        assert reduce_conjuncts(conjuncts) == fold_range_pairs(conjuncts)


def test_coindexed_natural_parts_for_sibling_equality():
    p = _period()
    [fold] = fold_coindexed_groups(
        [p.begin.month == p.end.month, p.begin.year == p.end.year]
    )
    parts = coindexed_natural_parts(fold)
    assert parts is not None
    assert parts.left_hop == ("begin", _Period) and parts.right_hop == ("end", _Period)


def test_coindexed_natural_parts_none_for_non_sibling_prefixes():
    o = variable(_Outer, domain=None)
    [fold] = fold_coindexed_groups([o.x.month == o.y.z.month, o.x.year == o.y.z.year])
    assert coindexed_natural_parts(fold) is None  # x vs y.z are not siblings → faithful
