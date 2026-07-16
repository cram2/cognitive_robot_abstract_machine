"""
Tests for nesting a related entity's restriction into the relative clause that
introduces it.

When a query binds a secondary entity through a relational attribute of the subject
(``robot.assigned_to == mission``) and separately restricts that entity
(``mission.priority > 2``), the two conditions fold into one relative clause on the
subject noun — *"Find a Robot that is assigned to a Mission with priority greater than
2"* — instead of the disjointed *"whose assigned_to is a Mission, such that the priority
of the Mission is greater than 2"*.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import an, entity, variable
from krrood.entity_query_language.verbalization.grammar.conditions.scoping import (
    RelationalBindingFold,
    bind_relational_entities,
)
from krrood.entity_query_language.verbalization.example_domain import (
    Mission as DomainMission,
    Robot as DomainRobot,
)
from krrood.entity_query_language.verbalization.microplanning.coordination import (
    reduce_conjuncts,
)
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression


@dataclass
class Mission:
    """
    A mission with a scalar priority — the bound secondary entity in these tests.
    """

    priority: int


@dataclass
class Base:
    """
    A base named for nesting-depth tests.
    """

    name: str


@dataclass
class Robot:
    """
    A robot related to a mission (agentive-free ``assigned_to``) — the query subject.
    """

    assigned_to: Mission


@dataclass
class Book:
    """
    A book with an agentive ``owned_by`` relation, to check the passive subject-relative
    form.
    """

    owned_by: Base


# ── end-to-end rendering ──────────────────────────────────────────────────────


def test_binding_with_order_restriction_nests_with_compact_with():
    robot, mission = variable(Robot, []), variable(Mission, [])
    query = an(entity(robot).where(robot.assigned_to == mission, mission.priority > 2))
    assert (
        verbalize_expression(query)
        == "Find a Robot that is assigned to a Mission with priority greater than 2"
    )


def test_bare_binding_without_restriction():
    robot, mission = variable(Robot, []), variable(Mission, [])
    query = an(entity(robot).where(robot.assigned_to == mission))
    assert verbalize_expression(query) == "Find a Robot that is assigned to a Mission"


def test_equality_restriction_falls_back_to_whose():
    """
    A non-order comparison on the bound entity keeps the ``whose … is …`` clause.
    """
    robot, base = variable(Book, []), variable(Base, [])
    query = an(entity(robot).where(robot.owned_by == base, base.name == "HQ"))
    text = verbalize_expression(query)
    assert "that is owned by a Base" in text
    assert "whose name is 'HQ'" in text


def test_agentive_binding_stays_passive():
    """
    The subject is the patient of an agentive relation, so the clause is passive, not
    active.
    """
    book, base = variable(Book, []), variable(Base, [])
    query = an(entity(book).where(book.owned_by == base))
    assert verbalize_expression(query) == "Find a Book that is owned by a Base"


# ── reverse direction (relation on the other entity, pointing at the subject) ──


def test_reverse_binding_nests_when_entity_has_restriction():
    """
    Ichumuh's example: the relation sits on the other entity (``mission.assigned_to ==
    robot``) and that entity carries a restriction — it folds into one relative clause
    on the subject.
    """
    robot, mission = variable(DomainRobot, []), variable(DomainMission, [])
    query = an(entity(robot).where(mission.assigned_to == robot, mission.priority > 2))
    assert (
        verbalize_expression(query)
        == "Find a Robot that is assigned to a Mission with priority greater than 2"
    )


def test_bare_reverse_binding_keeps_referent_unification():
    """
    A reverse binding with no restriction on the entity keeps its existing *"it is
    assigned to a Mission"* rendering — only a restriction to nest triggers the fold.
    """
    robot, mission = variable(DomainRobot, []), variable(DomainMission, [])
    query = an(entity(robot).where(mission.assigned_to == robot))
    assert (
        verbalize_expression(query)
        == "Find a Robot such that it is assigned to a Mission"
    )


# ── negative guards (unchanged behaviour) ─────────────────────────────────────


def test_entity_restricted_but_unbound_stays_standalone():
    """
    Without a binding relation from the subject, the entity's restriction is not nested.
    """
    robot, mission = variable(Robot, []), variable(Mission, [])
    query = an(entity(robot).where(mission.priority > 2))
    text = verbalize_expression(query)
    assert "that is assigned to" not in text


# ── the pure grouping analysis ────────────────────────────────────────────────


def test_bind_relational_entities_collapses_binding_and_restriction():
    robot, mission = variable(Robot, []), variable(Mission, [])
    grouped = bind_relational_entities(
        reduce_conjuncts([robot.assigned_to == mission, mission.priority > 2]), robot
    )
    folds = [item for item in grouped if isinstance(item, RelationalBindingFold)]
    assert len(folds) == 1
    assert (
        isinstance(folds[0].entity, Variable) and folds[0].entity._id_ == mission._id_
    )
    assert len(folds[0].nested) == 1
    # The binding and the restriction are the only conjuncts, both consumed into the fold.
    assert len(grouped) == 1


def test_bind_relational_entities_noop_without_binding():
    robot, mission = variable(Robot, []), variable(Mission, [])
    items = reduce_conjuncts([mission.priority > 2])
    assert bind_relational_entities(items, robot) == items
