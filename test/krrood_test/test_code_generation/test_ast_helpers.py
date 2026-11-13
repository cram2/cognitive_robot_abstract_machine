"""Tests for :mod:`krrood.code_generation.ast_helpers`."""

from __future__ import annotations

import ast
import operator
from enum import Enum

import pytest

from krrood.code_generation import ast_helpers


class Colour(Enum):
    """An enum whose members the generated expressions reference."""

    RED = "red"


# %% literal


@pytest.mark.parametrize("value", [Colour.RED, True, 3, 2.5, "text", None])
def test_literal_evaluates_back_to_its_value(value):
    """The expression evaluates to the value it was built from."""
    source = ast_helpers.unparse_expression(ast_helpers.literal(value))

    assert eval(source, {Colour.__name__: Colour}) == value


# %% comparison operators


@pytest.mark.parametrize(
    "operation, operator_class",
    [
        (operator.eq, ast.Eq),
        (operator.ne, ast.NotEq),
        (operator.lt, ast.Lt),
        (operator.le, ast.LtE),
        (operator.gt, ast.Gt),
        (operator.ge, ast.GtE),
    ],
)
def test_comparison_operator_class_renders_the_operation(operation, operator_class):
    """Each comparison function maps to the class that renders it."""
    assert ast_helpers.comparison_operator_class(operation) is operator_class


def test_comparison_operator_class_is_none_for_a_function_that_is_no_comparison():
    """A function that is no comparison has no comparison-operator class."""
    assert ast_helpers.comparison_operator_class(operator.add) is None
