"""
Tests for equality-induced referent unification and relational-identity collapse.

An ``==`` constraint that identifies a variable with a relational hop (``m.assigned_to == r``) is
said as the active relational predicate (*"it is assigned to a Mission"*), and the two referents are
counted as one entity so the subject reads *"a Robot"* rather than *"Robot 1"* / *"Robot 2"*.
"""

from __future__ import annotations

from krrood.entity_query_language.factories import an, entity, variable
from krrood.entity_query_language.verbalization.example_domain import Mission, Robot
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression


def test_relational_identity_collapses_to_active_predicate():
    """When the identified entity also carries a restriction, the binding and the restriction fold
    into one relative clause on the subject noun (the restriction nested onto the entity).
    """
    robot, mission = variable(Robot, []), variable(Mission, [])
    query = an(entity(robot).where(mission.assigned_to == robot, mission.priority > 2))
    assert verbalize_expression(query) == (
        "Find a Robot that is assigned to a Mission with priority greater than 2"
    )


def test_relational_identity_holds_either_operand_order():
    robot, mission = variable(Robot, []), variable(Mission, [])
    assert (
        verbalize_expression(an(entity(robot).where(robot == mission.assigned_to)))
        == "Find a Robot such that it is assigned to a Mission"
    )


def test_distinct_same_type_referents_are_still_numbered_without_an_identity():
    first, second = variable(Robot, []), variable(Robot, [])
    assert (
        verbalize_expression(an(entity(first).where(first.battery > second.battery)))
        == "Find Robot 1 whose battery is greater than the battery of Robot 2"
    )
