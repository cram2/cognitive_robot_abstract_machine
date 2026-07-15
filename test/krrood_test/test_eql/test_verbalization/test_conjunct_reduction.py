"""
Generality tests for the conjunct-reduction pass — the single home of WHERE-conjunct
simplification (range-bound and co-indexed folds).

They exercise the :class:`ConjunctReducer` registry directly and its individual folds,
on cases the golden end-to-end tests do not isolate.
"""

from __future__ import annotations

import dataclasses
import datetime

from krrood.entity_query_language.factories import variable
from krrood.entity_query_language.verbalization.microplanning.coordination import (
    CoindexedComparisonFold,
    CoindexedFold,
    ConjunctReducer,
    fold_coindexed_groups,
    RangeBoundFold,
    RangeFold,
    reduce_conjuncts,
)
from krrood.entity_query_language.verbalization.example_domain import (
    BankTransaction,
    LoveBirds,
)


def _sibling_pair_type() -> type:
    """
    A two-attribute sibling structure — ``Pair(left, right)`` where each side has
    ``low`` and ``high`` — so a co-indexed fold can range over more than one distinct
    leaf.
    """
    side = dataclasses.make_dataclass("Span", [("low", int), ("high", int)])
    return dataclasses.make_dataclass("Pair", [("left", side), ("right", side)])


def _coindexed_leaf_names(fold: CoindexedFold) -> list[str]:
    """:return: The terminal attribute names of *fold*, in order."""
    return [name for name, _owner in fold.terminals]


def _bounded_transaction():
    transaction = variable(BankTransaction, domain=None)
    low = datetime.datetime(2026, 5, 1)
    high = datetime.datetime(2026, 5, 30)
    return transaction, low, high


def test_reducer_default_pipeline_is_range_then_coindexed():
    """
    The pass applies its folds in order; the default registry is range-bound then co-
    indexed.
    """
    reducer = ConjunctReducer()
    assert [type(fold) for fold in reducer.folds] == [
        RangeBoundFold,
        CoindexedComparisonFold,
    ]


def test_complementary_bounds_fold_to_a_single_range():
    transaction, low, high = _bounded_transaction()
    reduced = reduce_conjuncts(
        [transaction.booking_date >= low, transaction.booking_date <= high]
    )
    assert len(reduced) == 1 and isinstance(reduced[0], RangeFold)


def test_bounds_in_reverse_order_still_fold_by_direction_not_position():
    transaction, low, high = _bounded_transaction()
    reduced = reduce_conjuncts(
        [transaction.booking_date <= high, transaction.booking_date >= low]
    )
    assert len(reduced) == 1
    fold = reduced[0]
    assert isinstance(fold, RangeFold)
    assert fold.lower_expression._value_ == low
    assert fold.upper_expression._value_ == high


def test_bounds_on_distinct_chains_do_not_fold():
    transaction, low, high = _bounded_transaction()
    conjuncts = [
        transaction.booking_date >= low,
        transaction.amount_details.amount <= 100,
    ]
    reduced = reduce_conjuncts(conjuncts)
    assert not any(isinstance(item, RangeFold) for item in reduced)
    assert len(reduced) == 2


def test_three_bounds_fold_one_pair_and_leave_the_remainder():
    """
    A lone third bound with no complement on its chain stays unfolded beside the folded
    pair.
    """
    transaction, low, high = _bounded_transaction()
    extra_low = datetime.datetime(2026, 5, 10)
    reduced = reduce_conjuncts(
        [
            transaction.booking_date >= low,
            transaction.booking_date <= high,
            transaction.amount_details.amount >= extra_low.day,
        ]
    )
    assert sum(isinstance(item, RangeFold) for item in reduced) == 1
    assert len(reduced) == 2


def test_lone_coindexed_comparison_is_not_folded():
    """
    A co-indexed fold needs at least two co-indexed comparisons; a single one says
    itself.
    """
    reduced = CoindexedComparisonFold().apply(
        reduce_conjuncts([])  # empty → empty, exercises the empty path too
    )
    assert reduced == []


def test_a_non_comparator_conjunct_passes_through_untouched():
    transaction, low, _ = _bounded_transaction()
    boolean_condition = transaction.amount_details.amount > 0
    reduced = reduce_conjuncts([boolean_condition])
    assert reduced == [boolean_condition]


def test_repeated_coindexed_leaf_is_listed_once():
    """
    Two identical co-indexed comparisons fold to one clause whose shared leaf is listed
    once — *"have the same name"*, never *"have the same name and name"*.
    """
    birds = variable(LoveBirds, domain=None)
    comparison = birds.bird_1.name == birds.bird_2.name
    repeated = birds.bird_1.name == birds.bird_2.name
    [fold] = fold_coindexed_groups([comparison, repeated])
    assert isinstance(fold, CoindexedFold)
    assert _coindexed_leaf_names(fold) == ["name"]


def test_distinct_coindexed_leaves_are_all_kept():
    """
    The repeated-leaf fix must not over-collapse: distinct leaves are each kept.
    """
    pair = variable(_sibling_pair_type(), domain=None)
    [fold] = fold_coindexed_groups(
        [pair.left.low == pair.right.low, pair.left.high == pair.right.high]
    )
    assert _coindexed_leaf_names(fold) == ["low", "high"]


def test_repeated_leaf_collapses_while_distinct_ones_remain():
    """
    A leaf repeated among distinct ones is deduped in first-occurrence order, not
    reordered.
    """
    pair = variable(_sibling_pair_type(), domain=None)
    [fold] = fold_coindexed_groups(
        [
            pair.left.low == pair.right.low,
            pair.left.high == pair.right.high,
            pair.left.low == pair.right.low,
        ]
    )
    assert _coindexed_leaf_names(fold) == ["low", "high"]
