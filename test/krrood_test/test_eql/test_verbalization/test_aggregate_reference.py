"""
Tests for how a computed quantity is referred back to.

An aggregate is named in full when it is introduced and shortened to its bare
aggregation word on a later mention (*"the sum of incomes of Statements"* -> *"the
sum"*). The word identifies the quantity only while it is the only aggregate of its
kind: once a query selects two sums, the short form names either one, so each is
described in full at every mention instead. An aggregate is told apart by what it
aggregates, not by a determiner, which is why it is spelled out rather than given the
*"another …"* of same-noun disambiguation.
"""

from __future__ import annotations

from dataclasses import dataclass

import krrood.entity_query_language.factories as eql
from krrood.entity_query_language.factories import a, set_of, variable
from krrood.entity_query_language.verbalization.microplanning.referring import (
    ReferringExpressions,
)
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression
from krrood.entity_query_language.verbalization.vocabulary.english import Aggregations
from krrood.patterns.role import Role

# %% mimic domain


@dataclass(eq=False)
class Money(Role[float]):
    """
    An amount of money, the leaf an aggregate sums over.
    """


@dataclass
class Period:
    """
    The span a statement covers, giving the queries below a grouping key.
    """

    month: int
    """
    The month the statement covers, used as the grouping key.
    """


@dataclass
class Statement:
    """
    An entity with two separately aggregable amounts, so one query can select two sums.
    """

    period: Period
    """
    The span the amounts below are taken over.
    """

    income: Money
    """
    The amount taken in over the period.
    """

    expenses: Money
    """
    The amount paid out over the period, aggregated by the same word as :attr:`income`.
    """


# %% two aggregates of one kind


def test_a_ranked_sum_is_described_in_full_when_another_sum_is_selected():
    """
    The ranking frame names the sum it ranks by, so the body's mention of that same sum
    is a repeat.

    It stays spelled out, because a bare *"the sum"* there would read as the expenses
    sum named right beside it.
    """
    statement = variable(Statement, domain=None)
    income = eql.sum(statement.income)
    expenses = eql.sum(statement.expenses)
    query = a(
        set_of(statement.period.month, income, expenses)
        .grouped_by(statement.period.month)
        .ordered_by(income, descending=True)
        .limit(1)
    )
    assert verbalize_expression(query) == (
        "For the month with the highest sum of incomes of Statements, "
        "report the month, the sum of incomes of Statements, "
        "and the sum of expenses of Statements"
    )


def test_an_ordered_by_sum_is_described_in_full_when_another_sum_is_selected():
    """
    Without a limit the repeat mention lands in the trailing *"ordered by …"* clause,
    where proximity makes a bare *"the sum"* read as the expenses sum immediately before
    it.
    """
    statement = variable(Statement, domain=None)
    income = eql.sum(statement.income)
    expenses = eql.sum(statement.expenses)
    query = a(
        set_of(statement.period.month, income, expenses)
        .grouped_by(statement.period.month)
        .ordered_by(income, descending=True)
    )
    assert verbalize_expression(query) == (
        "For each month, report the sum of incomes of Statements "
        "and the sum of expenses of Statements "
        "ordered by the sum of incomes of Statements from highest to lowest"
    )


# %% the only aggregate of its kind still shortens


def test_a_lone_sum_is_shortened_on_its_repeat_mention():
    """
    Nothing else is a sum, so the word identifies the quantity and the repeat mention
    shortens to it.
    """
    statement = variable(Statement, domain=None)
    income = eql.sum(statement.income)
    query = a(
        set_of(statement.period.month, income)
        .grouped_by(statement.period.month)
        .ordered_by(income, descending=True)
        .limit(1)
    )
    assert verbalize_expression(query) == (
        "For the month with the highest sum of incomes of Statements, "
        "report the month and the sum"
    )


def test_a_sum_is_shortened_alongside_an_aggregate_of_another_kind():
    """
    Two aggregates collide only when they share an aggregation word: a sum beside an
    average is still the only sum, so it shortens.
    """
    statement = variable(Statement, domain=None)
    income = eql.sum(statement.income)
    expenses = eql.average(statement.expenses)
    query = a(
        set_of(statement.period.month, income, expenses)
        .grouped_by(statement.period.month)
        .ordered_by(income, descending=True)
        .limit(1)
    )
    assert verbalize_expression(query) == (
        "For the month with the highest sum of incomes of Statements, "
        "report the month, the sum, "
        "and the average of expenses of Statements"
    )


# %% the pre-scan behind it


def test_an_aggregation_two_aggregates_share_is_recorded_as_shared():
    """
    Two sums over different chains are two aggregates of one aggregation, so that
    aggregation is recorded as shared.
    """
    statement = variable(Statement, domain=None)
    query = a(
        set_of(
            statement.period.month,
            eql.sum(statement.income),
            eql.sum(statement.expenses),
        ).grouped_by(statement.period.month)
    )
    referring = ReferringExpressions.from_expression(query)
    assert referring.shared_aggregations == {Aggregations.SUM}


def test_an_aggregation_only_one_aggregate_uses_is_not_shared():
    """
    A sum and an average are one aggregate each, so neither aggregation is shared.
    """
    statement = variable(Statement, domain=None)
    query = a(
        set_of(
            statement.period.month,
            eql.sum(statement.income),
            eql.average(statement.expenses),
        ).grouped_by(statement.period.month)
    )
    referring = ReferringExpressions.from_expression(query)
    assert referring.shared_aggregations == set()
