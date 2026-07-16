"""
Unit tests for the core structural/semantic expression helpers and the query-algebra
accessors extracted out of the verbalization planners.

These exercise query-algebra facts (chain roots, group keys, aggregated selections)
directly on the EQL expressions, without going through the natural-language generation
stack.
"""

import datetime
from dataclasses import dataclass

from krrood.entity_query_language.core.expression_structure import (
    chain_root,
    is_temporal,
    root_variable_ids,
    walk_chain,
)
from krrood.entity_query_language.core.variable import Literal
from krrood.entity_query_language.factories import set_of, sum, variable
from krrood.entity_query_language.query.query import Entity

from ..dataset.semantic_world_like_classes import Body


@dataclass
class Appointment:
    """
    A domain class with both a date-only and a datetime field for temporality tests.
    """

    due_date: datetime.date
    created_at: datetime.datetime


def test_walk_chain_on_plain_variable_is_empty_with_self_root():
    """
    A non-``MappedVariable`` walks to an empty chain rooted at itself.
    """
    body = variable(Body, [])
    chain, root = walk_chain(body)
    assert chain == []
    assert root is body


def test_walk_chain_on_attribute_yields_chain_and_variable_root():
    """
    ``body.size`` walks to a one-hop chain whose root is the ``body`` variable.
    """
    body = variable(Body, [])
    chain, root = walk_chain(body.size)
    assert [step._attribute_name_ for step in chain] == ["size"]
    assert root is body


def test_chain_root_of_attribute_is_the_base_variable():
    body = variable(Body, [])
    assert chain_root(body.size) is body
    assert chain_root(body) is body


def test_root_variable_ids_collects_distinct_variable_roots():
    """
    Attributes of the same variable share its root; distinct variables contribute
    distinct ids; expressions whose root is not a variable contribute nothing.
    """
    body = variable(Body, [])
    other = variable(Body, [])
    assert root_variable_ids([body.size, body.name]) == {body._id_}
    assert root_variable_ids([body.size, other.name]) == {body._id_, other._id_}
    assert root_variable_ids([]) == set()


def test_group_key_root_ids_are_the_keys_chain_roots():
    """
    ``group_key_root_ids`` holds the keys' chain-root ids — distinct from
    ``ids_of_variables_to_group_by`` (the keys' own ids) when a key is a chain.
    """
    body = variable(Body, [])
    grouped = (
        set_of(body.name, sum(body.size)).grouped_by(body.name)._grouped_by_expression_
    )
    assert grouped.group_key_root_ids == {body._id_}
    assert grouped.ids_of_variables_to_group_by == (body.name._id_,)


def test_aggregated_selections_excludes_group_keys():
    """
    A selection that is itself a group key is dropped; the aggregated selection is kept.
    """
    body = variable(Body, [])
    other = variable(Body, [])
    query = set_of(body, sum(other.size)).grouped_by(body)
    grouped = query._grouped_by_expression_
    aggregated = query.aggregated_selections(grouped.group_key_root_ids)
    assert aggregated == [sum(other.size)]


def test_entity_aggregated_selections_falls_back_to_generic_for_plain_selection():
    """
    An Entity whose selection is not an InstantiatedVariable uses the generic per-
    selection rule (its single selection, kept when it is not a group key).
    """
    body = variable(Body, [])
    selection = sum(body.size)
    entity_query = Entity(_selected_variables_=(selection,))
    assert entity_query.aggregated_selections(set()) == [selection]


def test_is_temporal_recognizes_datetime_value_and_attribute():
    """
    A ``datetime`` literal and a ``datetime``-typed attribute are temporal.
    """
    appointment = variable(Appointment, [])
    assert is_temporal(Literal(_value_=datetime.datetime(2026, 5, 1)))
    assert is_temporal(appointment.created_at)


def test_is_temporal_recognizes_date_value_and_attribute():
    """
    A plain ``date`` literal and a ``date``-typed attribute are temporal too — ``date``
    is the base class of ``datetime``, and the temporal comparators (*"before"*, *"no
    later than"*) read naturally for date-only fields.
    """
    appointment = variable(Appointment, [])
    assert is_temporal(Literal(_value_=datetime.date(2026, 5, 1)))
    assert is_temporal(appointment.due_date)


def test_is_temporal_is_false_for_non_temporal_value_and_attribute():
    """
    A non-date value and a non-date attribute are not temporal.
    """
    body = variable(Body, [])
    assert not is_temporal(Literal(_value_=5))
    assert not is_temporal(body.size)
