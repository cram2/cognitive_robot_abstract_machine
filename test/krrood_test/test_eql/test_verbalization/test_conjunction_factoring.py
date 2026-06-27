"""
Tests for conjunction factoring: an AND whose conjuncts are all value comparisons on the *same bare
variable* is said as a restrictive relative clause — *"an Integer that is greater than 1, less than
10, and is not 5"* — the subject and the lead copula stated once and only the predicate tails
coordinated. Scoped to a bare variable so the relative pronoun attaches unambiguously; an
attribute-chain conjunction keeps its per-clause surface.
"""

from __future__ import annotations

from krrood.entity_query_language.factories import and_, variable
from krrood.entity_query_language.verbalization.example_domain import Robot
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression


def test_bare_variable_conjuncts_factor_into_a_relative_clause():
    """A complementary bound pair folds to *between* inside the clause; the inequality keeps its tail."""
    x = variable(int, [])
    assert (
        verbalize_expression(and_(x > 1, x < 10, x != 5))
        == "an Integer that is between 1 and 10 and is not 5"
    )


def test_complementary_bounds_fold_to_between_in_the_clause():
    x = variable(int, [])
    assert (
        verbalize_expression(and_(x > 1, x < 10)) == "an Integer that is between 1 and 10"
    )


def test_unpaired_bounds_stay_spelled_out():
    """Without a complementary pair the comparisons stay as operator-and-value tails."""
    x = variable(int, [])
    assert (
        verbalize_expression(and_(x > 1, x > 3, x != 9))
        == "an Integer that is greater than 1, greater than 3, and is not 9"
    )


def test_inequality_tail_keeps_its_copula():
    """A ``!=`` tail re-introduces the copula (*"is not 5"*) since the negation cannot share the
    affirmative lead *"is"*."""
    x = variable(int, [])
    assert "and is not 5" in verbalize_expression(and_(x > 1, x != 5))


def test_attribute_chain_conjunction_is_not_factored():
    """An attribute-chain subject keeps per-clause rendering — a *"that"* clause would dangle off the
    owner noun (*"the battery of a Robot that …"*)."""
    robot = variable(Robot, [])
    assert (
        verbalize_expression(and_(robot.battery > 50, robot.name == "x"))
        == "the battery of a Robot is greater than 50, and the name of the Robot is 'x'"
    )
