"""
Tests for arithmetic operations in the Entity Query Language.

Arithmetic is written with the normal Python operators on EQL variables (``a + b``, ``-x`` …) and is
computed with Python's numeric operators; the operation nodes compose with comparison, filtering, and
aggregation like any other value.
"""

import operator
import statistics

import pytest

import krrood.entity_query_language.factories as eql
from krrood.entity_query_language.factories import (
    an,
    entity,
    variable,
)

from ...dataset.department_and_employee import Department, Employee


# --- each operator matches plain Python semantics ----------------------------

@pytest.mark.parametrize(
    "python_operator",
    [
        operator.add,
        operator.sub,
        operator.mul,
        operator.truediv,
        operator.floordiv,
        operator.mod,
        operator.pow,
    ],
)
def test_binary_operator_matches_python(python_operator):
    left_values = [10, 20, 30]
    right_value = 4
    left = variable(int, domain=left_values)
    query = an(entity(python_operator(left, right_value)))
    assert query.tolist() == [python_operator(value, right_value) for value in left_values]


def test_unary_negation():
    numbers = variable(int, domain=[1, -2, 3])
    assert an(entity(-numbers)).tolist() == [-1, 2, -3]


# --- reflected / non-commutative operators preserve operand order ------------

def test_reflected_non_commutative_operators():
    """
    A literal to the left of a variable (``10 - numbers``) must go through the operator's reflected
    dunder (``__rsub__``, ``__rtruediv__``, ...) rather than the variable's normal one, and preserve
    the literal-then-variable operand order - these operators give a different result the other way
    round, so a wrong order would be caught by the expected values below.
    """
    numbers = variable(int, domain=[2, 4])
    assert an(entity(10 - numbers)).tolist() == [8, 6]
    assert an(entity(10 / numbers)).tolist() == [5, 2.5]
    assert an(entity(20 // numbers)).tolist() == [10, 5]
    assert an(entity(10 % numbers)).tolist() == [0, 2]
    assert an(entity(2 ** numbers)).tolist() == [4, 16]


def test_literal_operand_in_either_position():
    """
    ``+`` and ``*`` are commutative, so a literal operand must produce the same result whether it is
    on the variable's right (the normal dunder, e.g. ``__add__``) or left (the reflected dunder, e.g.
    ``__radd__``) - both call paths must be exercised, not just one.
    """
    numbers = variable(int, domain=[2, 3])
    assert an(entity(numbers + 10)).tolist() == [12, 13]
    assert an(entity(10 + numbers)).tolist() == [12, 13]
    assert an(entity(numbers * 10)).tolist() == [20, 30]
    assert an(entity(10 * numbers)).tolist() == [20, 30]


# --- composes with the rest of EQL -------------------------------------------

def test_chained_arithmetic():
    numbers = variable(int, domain=[1, 2, 3])
    assert an(entity((numbers + 1) * 2)).tolist() == [4, 6, 8]


def test_comparison_of_arithmetic_filters_results():
    department = Department(name="research")
    employees = [
        Employee(name="ada", department=department, salary=1000, starting_salary=900),
        Employee(name="bob", department=department, salary=3000, starting_salary=500),
    ]
    employee = variable(Employee, domain=employees)
    query = an(entity(employee).where((employee.salary - employee.starting_salary) > 1000))
    assert [result.name for result in query.tolist()] == ["bob"]


def test_aggregation_over_arithmetic():
    numbers = variable(int, domain=[1, 2, 3])
    assert eql.sum(numbers + 1).tolist()[0] == sum(value + 1 for value in [1, 2, 3])
    assert eql.average(numbers * 2).tolist()[0] == statistics.mean([2, 4, 6])


def test_division_by_zero_raises():
    numerator = variable(int, domain=[1])
    denominator = variable(int, domain=[0])
    with pytest.raises(ZeroDivisionError):
        an(entity(numerator / denominator)).tolist()
