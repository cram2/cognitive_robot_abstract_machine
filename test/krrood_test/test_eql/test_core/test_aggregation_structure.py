"""
Tests for :mod:`krrood.entity_query_language.query.aggregation_structure` — the pure structural
queries about aggregation subqueries and result quantifiers, relocated out of the verbalization
package (they are query-algebra facts, usable by any consumer).
"""

from __future__ import annotations

import krrood.entity_query_language.factories as eql
from krrood.entity_query_language.factories import an, entity, variable
from krrood.entity_query_language.query.aggregation_structure import (
    aggregation_leaf_attribute,
    aggregation_source_root,
    is_aggregation_subquery,
    is_calculation_value,
    is_collapsible_aggregation_subquery,
    selected_aggregator,
    unwrap_result_quantifiers,
)

from ...dataset.department_and_employee import Employee


def test_unwrap_result_quantifiers_strips_the_wrapper():
    inner = entity(eql.max(variable(int, domain=[1, 2, 3])))
    assert unwrap_result_quantifiers(an(inner)) is inner


def test_unwrap_result_quantifiers_is_identity_on_a_bare_expression():
    bare = variable(int, domain=[1, 2, 3])
    assert unwrap_result_quantifiers(bare) is bare


def test_selected_aggregator_present_only_for_an_aggregation_entity():
    var = variable(int, domain=[1, 2, 3])
    assert selected_aggregator(entity(eql.max(var))) is not None
    assert selected_aggregator(entity(var)) is None


def test_is_aggregation_subquery_distinguishes_aggregation_from_plain():
    var = variable(int, domain=[1, 2, 3])
    assert is_aggregation_subquery(entity(eql.max(var)))
    assert not is_aggregation_subquery(entity(var))


def test_is_calculation_value_sees_through_result_quantifiers():
    var = variable(int, domain=[1, 2, 3])
    assert is_calculation_value(an(entity(eql.max(var))))
    assert not is_calculation_value(var)


def test_unconstrained_aggregation_subquery_is_collapsible():
    var = variable(int, domain=[1, 2, 3])
    assert is_collapsible_aggregation_subquery(entity(eql.max(var)))


def test_aggregation_source_root_and_leaf_attribute_of_a_chain_aggregate():
    employee = variable(Employee, domain=None)
    aggregate = entity(eql.max(employee.salary))
    assert aggregation_source_root(aggregate)._id_ == employee._id_
    leaf = aggregation_leaf_attribute(aggregate)
    assert leaf is not None and leaf._attribute_name_ == "salary"
