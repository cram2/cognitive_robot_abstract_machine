"""
Selecting the statements of a condition by whether they hold.
"""

from krrood.entity_query_language.factories import (
    and_,
    get_conditioned_statements,
    get_false_statements,
    get_true_statements,
    variable,
)

from ...dataset.example_classes import KRROODPosition
from ...dataset.value_comparisons import IsGreaterThan


def accept_every_result(results) -> bool:
    """
    :param results: The results of evaluating a statement.
    :return: Always true, so every statement is selected.
    """
    return True


# %% selecting statements by identity


def test_conditioned_statements_keep_every_operand_satisfying_the_condition():
    position = variable(KRROODPosition, [KRROODPosition(0.0, 1.0, 0.0)])
    left, right = position.x + 1, position.y + 1

    statements = get_conditioned_statements(
        IsGreaterThan(left, right), accept_every_result
    )

    assert [statement._id_ for statement in statements] == [left._id_, right._id_]


def test_attributes_a_statement_takes_are_not_statements_of_it():
    position = variable(KRROODPosition, [KRROODPosition(0.0, 1.0, 0.0)])

    statements = get_conditioned_statements(
        IsGreaterThan(position.x, position.y), accept_every_result
    )

    assert statements == []


# %% selecting statements by truth


def test_false_statements_are_the_statements_that_do_not_hold():
    position = variable(KRROODPosition, [KRROODPosition(2.0, 0.0, 0.0)])
    false_statement = IsGreaterThan(position.y, position.x)
    true_statement = IsGreaterThan(position.x, position.y)

    statements = get_false_statements(and_(false_statement, true_statement))

    assert [statement._id_ for statement in statements] == [false_statement._id_]


def test_true_statements_are_the_statements_that_hold():
    position = variable(KRROODPosition, [KRROODPosition(2.0, 0.0, 0.0)])
    false_statement = IsGreaterThan(position.y, position.x)
    true_statement = IsGreaterThan(position.x, position.y)

    statements = get_true_statements(and_(false_statement, true_statement))

    assert [statement._id_ for statement in statements] == [true_statement._id_]
