"""
Tests for :mod:`krrood.code_generation.ast_helpers`.
"""

from __future__ import annotations

import ast
import operator

import pytest

from krrood.code_generation.ast_helpers import (
    ast_call,
    ast_comparison_operator_class,
    ast_load_name,
    ast_unparse_expression,
)


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
def test_ast_comparison_operator_class_renders_the_operation(operation, operator_class):
    """
    Each comparison function maps to the class that renders it.
    """
    assert ast_comparison_operator_class(operation) is operator_class


def test_ast_comparison_operator_class_is_none_for_a_function_that_is_no_comparison():
    """
    A function that is no comparison has no comparison-operator class.
    """
    assert ast_comparison_operator_class(operator.add) is None


# %% calls


def test_ast_call_passes_positional_then_keyword_arguments():
    """
    The call lists its positional arguments before its keyword arguments.
    """
    call = ast_call("build", [ast_load_name("part")], {"size": ast_load_name("large")})

    assert ast_unparse_expression(call) == "build(part, size=large)"
