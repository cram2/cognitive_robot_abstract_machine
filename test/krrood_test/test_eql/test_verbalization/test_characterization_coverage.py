"""
Characterization (golden) tests that widen the black-box safety net before the verbalization
refactor (:doc:`/eql/developer/verbalization_refactor_plan`).

They pin the *current* rendered English for behaviours the existing suite under-covers — multi-way
coordination/folding, mixed nesting, double negation, and plural-subject agreement across the whole
realisation pipeline — so a structural refactor that must preserve output has a tight regression
net. They assert through the public API (``verbalize_expression``) only, so they survive any internal
restructuring.

.. note::
    These record behaviour as-is, not as-desired. Where the current output is itself a simplification
    opportunity (e.g. double negation is *not* collapsed), that is captured as a fact to preserve
    until a phase deliberately changes it (with its own test).
"""

from __future__ import annotations

import krrood.entity_query_language.factories as eql
from krrood.entity_query_language.factories import (
    a,
    an,
    and_,
    entity,
    not_,
    or_,
    set_of,
    variable,
)
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression
from krrood.entity_query_language.verbalization.example_domain import (
    BankTransaction,
    Employee,
    Robot,
)


def test_three_co_owned_selection_attributes_fold_into_one_genitive():
    """Three contiguous attributes of one owner share a single genitive, not three repetitions."""
    employee = variable(Employee, domain=None)
    assert (
        verbalize_expression(
            a(set_of(employee.name, employee.department, employee.salary))
        )
        == "Find the name, department, and salary of an Employee"
    )


def test_range_fold_coexists_with_an_unrelated_conjunct():
    """A folded lower/upper bound pair on one chain folds while a sibling conjunct stays standalone."""
    robot = variable(Robot, domain=None)
    assert (
        verbalize_expression(
            an(
                entity(robot).where(
                    robot.battery > 10, robot.battery < 90, robot.operational == True
                )
            )
        )
        == "Find a Robot whose battery is between 10 and 90, such that the Robot is operational"
    )


def test_and_nested_inside_or_keeps_both_connectives():
    """A conjunction nested inside a disjunction renders both connectives without flattening across."""
    robot = variable(Robot, domain=None)
    assert verbalize_expression(
        an(
            entity(robot).where(
                or_(
                    and_(robot.battery > 50, robot.operational == True),
                    robot.battery < 5,
                )
            )
        )
    ) == (
        "Find a Robot such that either its battery is greater than 50, "
        "and the Robot is operational, or its battery is less than 5"
    )


def test_double_negation_is_not_currently_simplified():
    """``not_(not_(x))`` is rendered literally today (a future simplification pass may collapse it)."""
    robot = variable(Robot, domain=None)
    assert (
        verbalize_expression(not_(not_(robot.battery > 5)))
        == "not (the battery of a Robot is not greater than 5)"
    )


def test_superlative_aggregation_in_where_collapses_to_a_modifier():
    """``chain == max(other.chain)`` over a distinct same-type variable reads as *"with the maximum …"*."""
    transaction = variable(BankTransaction, domain=None)
    other = variable(BankTransaction, domain=None)
    assert (
        verbalize_expression(
            an(
                entity(transaction).where(
                    transaction.amount_details.amount
                    == an(entity(eql.max(other.amount_details.amount)))
                )
            )
        )
        == "Find a BankTransaction with the maximum amount"
    )


def test_plural_subject_agreement_runs_through_the_whole_pipeline():
    """An ordered, filtered listing exercises every realisation pass: bare-plural determiner, noun and
    copula pluralisation, ``the`` → ``their`` pronominalisation, and comma orthography.
    """
    employee = variable(Employee, domain=None)
    assert verbalize_expression(
        an(entity(employee).where(employee.salary > 5).ordered_by(employee.salary))
    ) == (
        "Report Employees whose salaries are greater than 5, "
        "ordered by their salaries from lowest to highest"
    )
