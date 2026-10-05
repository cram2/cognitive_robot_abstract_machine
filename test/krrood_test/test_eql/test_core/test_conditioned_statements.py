"""
Selecting the statements of a condition by whether they hold.
"""

from krrood.entity_query_language.factories import (
    and_,
    get_false_statements,
    get_true_statements,
    not_,
    or_,
    variable,
)

from ...dataset.example_classes import KRROODPosition
from ...dataset.value_comparisons import IsGreaterThan

# %% which expressions are statements


def test_a_statement_on_its_own_is_a_statement_of_itself():
    position = variable(KRROODPosition, [KRROODPosition(0.0, 1.0, 0.0)])
    comparison = IsGreaterThan(position.x, position.y)

    statements = get_false_statements(comparison)

    assert [statement._id_ for statement in statements] == [comparison._id_]


def test_attributes_a_statement_takes_are_not_statements_of_it():
    position = variable(KRROODPosition, [KRROODPosition(1.0, 2.0, 0.0)])

    statements = get_true_statements(IsGreaterThan(position.x, position.y))

    assert statements == []


def test_calculated_operands_of_a_statement_are_not_statements_of_it():
    position = variable(KRROODPosition, [KRROODPosition(0.0, 1.0, 0.0)])

    statements = get_true_statements(IsGreaterThan(position.x + 1, position.y + 1))

    assert statements == []


def test_comparisons_a_statement_takes_as_values_are_not_statements_of_it():
    position = variable(KRROODPosition, [KRROODPosition(2.0, 0.0, 0.0)])
    comparison = IsGreaterThan(position.x > 1.0, position.y > 1.0)

    statements = get_true_statements(comparison)

    assert [statement._id_ for statement in statements] == [comparison._id_]


def test_statements_of_nested_conjunctions_are_the_conjuncts():
    position = variable(KRROODPosition, [KRROODPosition(2.0, 2.0, 0.0)])
    large_x = IsGreaterThan(position.x, 1.0)
    large_y = IsGreaterThan(position.y, 1.0)
    large_z = IsGreaterThan(position.z, 1.0)

    statements = get_true_statements(and_(large_x, large_y, large_z))

    assert [statement._id_ for statement in statements] == [
        large_x._id_,
        large_y._id_,
    ]


def test_a_false_first_conjunct_of_nested_conjunctions_is_a_false_statement():
    position = variable(KRROODPosition, [KRROODPosition(0.0, 2.0, 2.0)])
    large_x = IsGreaterThan(position.x, 1.0)
    large_y = IsGreaterThan(position.y, 1.0)
    large_z = IsGreaterThan(position.z, 1.0)

    statements = get_false_statements(and_(large_x, large_y, large_z))

    assert [statement._id_ for statement in statements] == [large_x._id_]


def test_a_negation_that_does_not_hold_is_a_false_statement():
    position = variable(KRROODPosition, [KRROODPosition(2.0, 0.0, 0.0)])
    negation = not_(IsGreaterThan(position.x, 1.0))

    statements = get_false_statements(negation)

    assert [statement._id_ for statement in statements] == [negation._id_]


def test_what_a_holding_negation_negates_is_not_a_false_statement():
    """
    A negation that holds is satisfied by what it negates being false, so that is not
    reported as a statement that could not be satisfied.
    """
    position = variable(KRROODPosition, [KRROODPosition(2.0, 0.0, 0.0)])
    large_x = IsGreaterThan(position.x, 1.0)
    large_y = IsGreaterThan(position.y, 1.0)
    large_z = IsGreaterThan(position.z, 1.0)

    statements = get_false_statements(and_(large_x, not_(large_y), large_z))

    assert [statement._id_ for statement in statements] == [large_z._id_]


def test_what_a_negation_negates_is_not_a_true_statement():
    position = variable(KRROODPosition, [KRROODPosition(2.0, 2.0, 0.0)])
    large_x = IsGreaterThan(position.x, 1.0)
    large_y = IsGreaterThan(position.y, 1.0)

    statements = get_true_statements(and_(large_y, not_(large_x)))

    assert [statement._id_ for statement in statements] == [large_y._id_]


def test_nothing_inside_a_negation_is_a_statement():
    position = variable(KRROODPosition, [KRROODPosition(0.0, 2.0, 0.0)])
    negation = not_(or_(IsGreaterThan(position.x, 1.0), IsGreaterThan(position.y, 1.0)))

    statements = get_false_statements(negation)

    assert [statement._id_ for statement in statements] == [negation._id_]


def test_every_alternative_that_does_not_hold_is_a_false_statement():
    position = variable(KRROODPosition, [KRROODPosition(0.0, 0.0, 0.0)])
    large_x = IsGreaterThan(position.x, 1.0)
    large_y = IsGreaterThan(position.y, 1.0)

    statements = get_false_statements(or_(large_x, large_y))

    assert [statement._id_ for statement in statements] == [
        large_x._id_,
        large_y._id_,
    ]


# %% selecting statements by truth


def test_false_statements_are_the_statements_that_do_not_hold():
    position = variable(KRROODPosition, [KRROODPosition(2.0, 0.0, 0.0)])
    false_statement = IsGreaterThan(position.y, position.x)
    true_statement = IsGreaterThan(position.x, position.y)

    statements = get_false_statements(and_(true_statement, false_statement))

    assert [statement._id_ for statement in statements] == [false_statement._id_]


def test_true_statements_are_the_statements_that_hold():
    position = variable(KRROODPosition, [KRROODPosition(2.0, 0.0, 0.0)])
    false_statement = IsGreaterThan(position.y, position.x)
    true_statement = IsGreaterThan(position.x, position.y)

    statements = get_true_statements(and_(true_statement, false_statement))

    assert [statement._id_ for statement in statements] == [true_statement._id_]


# %% statements judged together with the values the other conditions bind


def test_false_statements_hold_for_no_value_the_other_conditions_allow():
    """
    A statement that holds for some value of a variable is still false when it holds for
    none of the values the other conditions allow.
    """
    position = variable(
        KRROODPosition,
        [KRROODPosition(2.0, 0.0, 0.0), KRROODPosition(0.0, 2.0, 0.0)],
    )
    large_x = IsGreaterThan(position.x, 1.0)
    large_y = IsGreaterThan(position.y, 1.0)

    statements = get_false_statements(and_(large_x, large_y))

    assert [statement._id_ for statement in statements] == [large_y._id_]


def test_true_statements_hold_for_a_value_the_other_conditions_allow():
    """
    A statement holds only when it holds for a value the other conditions allow.
    """
    position = variable(
        KRROODPosition,
        [KRROODPosition(2.0, 0.0, 0.0), KRROODPosition(0.0, 2.0, 0.0)],
    )
    large_x = IsGreaterThan(position.x, 1.0)
    large_y = IsGreaterThan(position.y, 1.0)

    statements = get_true_statements(and_(large_x, large_y))

    assert [statement._id_ for statement in statements] == [large_x._id_]


def test_statements_skipped_after_a_false_statement_are_neither_true_nor_false():
    """
    A statement that is never evaluated, because an earlier one ruled out every value,
    is not reported as false, nor as true.
    """
    position = variable(KRROODPosition, [KRROODPosition(0.0, 0.0, 0.0)])
    large_x = IsGreaterThan(position.x, 1.0)
    large_y = IsGreaterThan(position.y, 1.0)
    condition = and_(large_x, large_y)

    false_statements = get_false_statements(condition)
    true_statements = get_true_statements(condition)

    assert [statement._id_ for statement in false_statements] == [large_x._id_]
    assert true_statements == []
