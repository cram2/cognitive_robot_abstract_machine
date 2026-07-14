"""
Tests for the shared owner-grouping aggregation primitives in the coordination home —
the single mechanism both the match construction-pattern grouping (all same-owner
attributes of a position) and the ``set_of`` selection grouping (a consecutive run of
attributes sharing one owner) build on.
"""

from __future__ import annotations

from krrood.entity_query_language.core.mapped_variable import Attribute
from krrood.entity_query_language.factories import variable
from krrood.entity_query_language.verbalization.microplanning.coordination import (
    OwnerGroup,
    group_by_owner,
    group_consecutive_by_owner,
)
from krrood.entity_query_language.verbalization.example_domain import (
    Department,
    Employee,
)


def _attribute_owner(expression):
    return (
        (expression._child_, expression) if isinstance(expression, Attribute) else None
    )


def test_group_by_owner_collects_all_same_owner_in_first_seen_order():
    employee = variable(Employee, domain=None)
    department = variable(Department, domain=None)
    items = [employee.department, department.name, employee.salary]  # emp, dept, emp
    groups, ungrouped = group_by_owner(items, _attribute_owner)
    assert [group.owner._id_ for group in groups] == [employee._id_, department._id_]
    assert len(groups[0].items) == 2  # department and salary, even though not adjacent
    assert ungrouped == []


def test_group_by_owner_returns_unclassified_items_separately():
    employee = variable(Employee, domain=None)
    items = [employee.department, employee]  # the bare variable does not classify
    groups, ungrouped = group_by_owner(items, _attribute_owner)
    assert len(groups) == 1 and ungrouped == [employee]


def test_group_consecutive_only_folds_adjacent_same_owner_runs():
    employee = variable(Employee, domain=None)
    department = variable(Department, domain=None)
    items = [employee.department, employee.salary, department.name, employee.name]
    result = group_consecutive_by_owner(items, _attribute_owner)
    assert isinstance(result[0], OwnerGroup) and len(result[0].items) == 2
    assert result[1] is items[2]  # dept.name — different owner, stays a raw singleton
    assert result[2] is items[3]  # emp.name — non-adjacent to the run, stays raw


def test_group_consecutive_passes_a_lone_item_through_unchanged():
    employee = variable(Employee, domain=None)
    items = [employee.department]
    assert group_consecutive_by_owner(items, _attribute_owner) == [employee.department]
