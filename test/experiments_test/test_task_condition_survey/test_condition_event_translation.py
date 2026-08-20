"""
Representing an EQL condition as a random event, and the cost of doing so.

The conditions here are built directly rather than parsed, so each test pins one
property of the representation -- how a conjunction, a disjunction or a negation turns
into terms -- independently of how any task suite happens to phrase it.
"""

import pytest
from krrood.entity_query_language.factories import and_, exists, for_all, not_, or_
from krrood.parametrization.exceptions import WhereExpressionIsFirstOrder
from random_events.product_algebra import Event

from experiments.random_events_experiments.task_condition_survey.condition_event_translation import (
    ConditionEventTranslator,
    ConditionNotFullyRead,
)
from experiments.random_events_experiments.task_condition_survey.task_conditions import (
    ConstantCondition,
    SceneObject,
    UnreadCondition,
)

from .test_task_conditions import condition, relation


def translated(expression) -> Event:
    """
    :param expression: The EQL condition to represent.
    :return: The event representing it.
    """
    return ConditionEventTranslator().translate(condition(expression))


# %% terms a condition needs


def test_conjunction_of_relations_needs_one_term():
    """
    Requiring several relations at once is a single box in the product space, however
    many relations it requires.
    """
    event = translated(and_(relation("a"), relation("b"), relation("c")))

    assert len(event.simple_sets) == 1
    assert len(event.simple_sets[0].variables) == 3


def test_disjunction_of_relations_needs_one_term_per_alternative():
    """
    Accepting any of several relations needs one term per alternative, since the product
    algebra represents a union as a disjunction of boxes.
    """
    event = translated(or_(relation("a"), relation("b")))

    assert len(event.simple_sets) == 2


def test_every_term_spans_every_variable():
    """
    A relation an operand does not mention is still carried by it, since the product
    algebra combines events by the variables they already carry.
    """
    event = translated(or_(relation("a"), relation("b")))

    for simple_set in event.simple_sets:
        assert {variable.name for variable in simple_set.variables} == {"a", "b"}


def test_negation_of_a_relation_holds_exactly_where_it_fails():
    """
    Negating a condition gives the event holding wherever the condition does not, so the
    two are disjoint and together cover everything.
    """
    holds = translated(relation("a"))
    fails = translated(not_(relation("a")))

    assert (holds & fails).is_empty()
    assert not (holds | fails).is_empty()
    assert (holds | fails).complement().is_empty()


def test_repeated_relation_is_one_variable():
    """
    A condition naming the same relation of the same objects twice constrains one
    variable, not two, so sharing a proposition is what lets its terms merge.
    """
    event = translated(and_(relation("a"), relation("a")))

    assert len(event.simple_sets[0].variables) == 1


def test_one_relation_of_two_objects_is_two_variables():
    """
    A relation stated of different objects states different conditions, so the objects
    have to reach the representation rather than being collapsed into the relation's
    name.
    """
    event = translated(
        and_(
            relation("a", stated_of="(cup)"),
            relation("a", stated_of="(bowl)"),
        )
    )

    assert len(event.simple_sets[0].variables) == 2


# %% conditions that cannot be represented


def test_unread_condition_is_refused_rather_than_guessed():
    """
    A condition with a part the survey never read constrains something unknown, so it is
    refused rather than represented as though the part were absent.
    """
    with pytest.raises(ConditionNotFullyRead):
        translated(and_(relation("a"), UnreadCondition.found("unread")))


def test_first_order_condition_is_refused_rather_than_instantiated():
    """
    A condition quantifying over a collection states something about objects rather than
    about a fixed set of propositions, and a propositional algebra represents no such
    statement, so it is refused instead of being instantiated over a chosen number of
    objects.
    """
    with pytest.raises(WhereExpressionIsFirstOrder):
        translated(for_all(SceneObject.provided_by_the_scene(), relation("a")))


def test_first_order_condition_is_refused_wherever_it_is_quantified():
    """
    Quantifying a part of a condition makes the whole condition first order, so a
    quantification buried under other operators is refused just as a bare one is.
    """
    with pytest.raises(WhereExpressionIsFirstOrder):
        translated(
            and_(
                relation("a"),
                not_(exists(SceneObject.provided_by_the_scene(), relation("b"))),
            )
        )


def test_first_order_condition_has_no_fixed_variable_set():
    """
    A condition quantifying over an unknown collection has no fixed set of variables, so
    asking for them is refused.
    """
    quantified = condition(for_all(SceneObject.provided_by_the_scene(), relation("a")))

    with pytest.raises(WhereExpressionIsFirstOrder):
        ConditionEventTranslator.variables_of(quantified)


# %% size of a condition's representation


def test_alternatives_on_private_relation_pairs_multiply_the_terms():
    """
    Alternatives each requiring their own pair of relations share no variable, so nothing
    merges: the product algebra holds a union as disjoint terms, and the representation
    grows with the number of alternatives rather than with their size.
    """
    alternative_count = 3
    alternatives = condition(
        or_(
            *(
                and_(relation(f"a{index}"), relation(f"b{index}"))
                for index in range(alternative_count)
            )
        )
    )

    translator = ConditionEventTranslator()

    assert translator.simple_set_count(alternatives) == 2**alternative_count - 1


def test_shared_relations_keep_the_representation_small():
    """
    Alternatives built from the same relations let their terms merge, so a condition
    written as a union of overlapping requirements stays a single term.
    """
    overlapping = condition(
        or_(
            and_(relation("a"), relation("b")),
            and_(relation("b"), relation("a")),
        )
    )

    translator = ConditionEventTranslator()

    assert translator.simple_set_count(overlapping) == 1


def test_outcome_settled_outright_costs_nothing_to_require():
    """
    A branch settling the outcome outright constrains no variable, so requiring it
    alongside a relation costs the same as requiring the relation alone.
    """
    settled = condition(and_(relation("a"), ConstantCondition.ALWAYS.expression))

    assert ConditionEventTranslator().simple_set_count(settled) == 1
