"""
Tests for disjunction factoring: an OR whose disjuncts are all value comparisons on the *same*
subject chain is said with the subject and copula stated once and only the predicate tails
coordinated — *"the battery of a Robot is either greater than 50 or less than 10"* — rather than
repeating *"the battery of … is …"* per disjunct.
"""

from __future__ import annotations

from krrood.entity_query_language.factories import or_, variable
from krrood.entity_query_language.verbalization.example_domain import Robot
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression


def test_two_disjuncts_on_one_subject_factor_the_subject():
    robot = variable(Robot, [])
    assert (
        verbalize_expression(or_(robot.battery > 50, robot.battery < 10))
        == "the battery of a Robot is either greater than 50 or less than 10"
    )


def test_three_disjuncts_on_one_subject_factor_with_oxford_comma():
    robot = variable(Robot, [])
    assert (
        verbalize_expression(
            or_(robot.battery > 50, robot.battery < 10, robot.battery == 30)
        )
        == "the battery of a Robot is either greater than 50, less than 10, or 30"
    )


def test_disjuncts_on_different_subjects_are_not_factored():
    robot = variable(Robot, [])
    assert (
        verbalize_expression(or_(robot.battery > 50, robot.name == "x"))
        == "either the battery of a Robot is greater than 50, "
        "or the name of the Robot is 'x'"
    )
