"""
Tests for
:meth:`~krrood.entity_query_language.core.mapped_variable.Attribute.number_like_field`.

Consolidates the "does this attribute resolve to a numeric type" check that used to be
duplicated by callers into a single, reusable method on the attribute itself.
"""

import pytest

from krrood.entity_query_language.exceptions import (
    AmbiguousQueryAttribute,
    NotNumberLikeFieldError,
)
from krrood.entity_query_language.factories import entity, set_of, variable

from ...dataset.department_and_employee import Employee


def test_number_like_field_resolves_a_numeric_field():
    field = variable(Employee, domain=[]).salary.number_like_field()
    assert field._attribute_name_ == "salary"


def test_number_like_field_rejects_a_non_numeric_field():
    with pytest.raises(NotNumberLikeFieldError):
        variable(Employee, domain=[]).name.number_like_field()


def test_number_like_field_rejects_a_missing_field():
    with pytest.raises(NotNumberLikeFieldError):
        variable(Employee, domain=[]).does_not_exist.number_like_field()


def test_number_like_field_resolves_through_a_single_variable_query():
    """
    A query's own type does not resolve directly -- what it selects lives on its
    selected variable instead -- so ``number_like_field`` must follow that selection.
    """
    query = entity(variable(Employee, domain=[]))
    field = query.salary.number_like_field()
    assert field._attribute_name_ == "salary"


def test_number_like_field_rejects_a_query_selecting_multiple_variables():
    query = set_of(variable(Employee, domain=[]), variable(Employee, domain=[]))
    with pytest.raises(AmbiguousQueryAttribute):
        query.salary.number_like_field()
