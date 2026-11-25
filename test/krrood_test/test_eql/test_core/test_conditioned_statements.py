"""
Selecting the statements of a condition by whether they hold.
"""

import pytest

from krrood.entity_query_language.evaluation import ConditionEvaluation
from krrood.entity_query_language.factories import (
    and_,
    evaluate_condition,
    get_false_statements,
    get_true_statements,
    literal,
    not_,
    or_,
    variable,
)
from krrood.entity_query_language.ground_term import AttributeTerm

from ...dataset.example_classes import KRROODPosition
from ...dataset.term_readers import AssignedValueReader
from ...dataset.value_comparisons import IsGreaterThan

# %% which expressions are statements


def test_a_statement_on_its_own_is_a_statement_of_itself():
    position = variable(KRROODPosition, [KRROODPosition(x=0.0, y=1.0, z=0.0)])
    comparison = IsGreaterThan(value=position.x, bound=position.y)

    statements = get_false_statements(comparison)

    assert [statement._id_ for statement in statements] == [comparison._id_]


def test_attributes_a_statement_takes_are_not_statements_of_it():
    position = variable(KRROODPosition, [KRROODPosition(x=1.0, y=2.0, z=0.0)])

    statements = get_true_statements(IsGreaterThan(value=position.x, bound=position.y))

    assert statements == []


def test_calculated_operands_of_a_statement_are_not_statements_of_it():
    position = variable(KRROODPosition, [KRROODPosition(x=0.0, y=1.0, z=0.0)])

    statements = get_true_statements(
        IsGreaterThan(value=position.x + 1, bound=position.y + 1)
    )

    assert statements == []


def test_comparisons_a_statement_takes_as_values_are_not_statements_of_it():
    position = variable(KRROODPosition, [KRROODPosition(x=2.0, y=0.0, z=0.0)])
    comparison = IsGreaterThan(value=position.x > 1.0, bound=position.y > 1.0)

    statements = get_true_statements(comparison)

    assert [statement._id_ for statement in statements] == [comparison._id_]


def test_statements_of_nested_conjunctions_are_the_conjuncts():
    position = variable(KRROODPosition, [KRROODPosition(x=2.0, y=2.0, z=0.0)])
    large_x = IsGreaterThan(value=position.x, bound=1.0)
    large_y = IsGreaterThan(value=position.y, bound=1.0)
    large_z = IsGreaterThan(value=position.z, bound=1.0)

    statements = get_true_statements(and_(large_x, large_y, large_z))

    assert [statement._id_ for statement in statements] == [
        large_x._id_,
        large_y._id_,
    ]


def test_a_false_first_conjunct_of_nested_conjunctions_is_a_false_statement():
    position = variable(KRROODPosition, [KRROODPosition(x=0.0, y=2.0, z=2.0)])
    large_x = IsGreaterThan(value=position.x, bound=1.0)
    large_y = IsGreaterThan(value=position.y, bound=1.0)
    large_z = IsGreaterThan(value=position.z, bound=1.0)

    statements = get_false_statements(and_(large_x, large_y, large_z))

    assert [statement._id_ for statement in statements] == [large_x._id_]


def test_a_negation_that_does_not_hold_is_a_false_statement():
    position = variable(KRROODPosition, [KRROODPosition(x=2.0, y=0.0, z=0.0)])
    negation = not_(IsGreaterThan(value=position.x, bound=1.0))

    statements = get_false_statements(negation)

    assert [statement._id_ for statement in statements] == [negation._id_]


def test_what_a_holding_negation_negates_is_not_a_false_statement():
    """
    A negation that holds is satisfied by what it negates being false, so that is not
    reported as a statement that could not be satisfied.
    """
    position = variable(KRROODPosition, [KRROODPosition(x=2.0, y=0.0, z=0.0)])
    large_x = IsGreaterThan(value=position.x, bound=1.0)
    large_y = IsGreaterThan(value=position.y, bound=1.0)
    large_z = IsGreaterThan(value=position.z, bound=1.0)

    statements = get_false_statements(and_(large_x, not_(large_y), large_z))

    assert [statement._id_ for statement in statements] == [large_z._id_]


def test_what_a_negation_negates_is_not_a_true_statement():
    position = variable(KRROODPosition, [KRROODPosition(x=2.0, y=2.0, z=0.0)])
    large_x = IsGreaterThan(value=position.x, bound=1.0)
    large_y = IsGreaterThan(value=position.y, bound=1.0)

    statements = get_true_statements(and_(large_y, not_(large_x)))

    assert [statement._id_ for statement in statements] == [large_y._id_]


def test_nothing_inside_a_negation_is_a_statement():
    position = variable(KRROODPosition, [KRROODPosition(x=0.0, y=2.0, z=0.0)])
    negation = not_(
        or_(
            IsGreaterThan(value=position.x, bound=1.0),
            IsGreaterThan(value=position.y, bound=1.0),
        )
    )

    statements = get_false_statements(negation)

    assert [statement._id_ for statement in statements] == [negation._id_]


def test_every_alternative_that_does_not_hold_is_a_false_statement():
    position = variable(KRROODPosition, [KRROODPosition(x=0.0, y=0.0, z=0.0)])
    large_x = IsGreaterThan(value=position.x, bound=1.0)
    large_y = IsGreaterThan(value=position.y, bound=1.0)

    statements = get_false_statements(or_(large_x, large_y))

    assert [statement._id_ for statement in statements] == [
        large_x._id_,
        large_y._id_,
    ]


# %% selecting statements by truth


def test_false_statements_are_the_statements_that_do_not_hold():
    position = variable(KRROODPosition, [KRROODPosition(x=2.0, y=0.0, z=0.0)])
    false_statement = IsGreaterThan(value=position.y, bound=position.x)
    true_statement = IsGreaterThan(value=position.x, bound=position.y)

    statements = get_false_statements(and_(true_statement, false_statement))

    assert [statement._id_ for statement in statements] == [false_statement._id_]


def test_true_statements_are_the_statements_that_hold():
    position = variable(KRROODPosition, [KRROODPosition(x=2.0, y=0.0, z=0.0)])
    false_statement = IsGreaterThan(value=position.y, bound=position.x)
    true_statement = IsGreaterThan(value=position.x, bound=position.y)

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
        [KRROODPosition(x=2.0, y=0.0, z=0.0), KRROODPosition(x=0.0, y=2.0, z=0.0)],
    )
    large_x = IsGreaterThan(value=position.x, bound=1.0)
    large_y = IsGreaterThan(value=position.y, bound=1.0)

    statements = get_false_statements(and_(large_x, large_y))

    assert [statement._id_ for statement in statements] == [large_y._id_]


def test_true_statements_hold_for_a_value_the_other_conditions_allow():
    """
    A statement holds only when it holds for a value the other conditions allow.
    """
    position = variable(
        KRROODPosition,
        [KRROODPosition(x=2.0, y=0.0, z=0.0), KRROODPosition(x=0.0, y=2.0, z=0.0)],
    )
    large_x = IsGreaterThan(value=position.x, bound=1.0)
    large_y = IsGreaterThan(value=position.y, bound=1.0)

    statements = get_true_statements(and_(large_x, large_y))

    assert [statement._id_ for statement in statements] == [large_x._id_]


def test_statements_skipped_after_a_false_statement_are_neither_true_nor_false():
    """
    A statement that is never evaluated, because an earlier one ruled out every value,
    is not reported as false, nor as true.
    """
    position = variable(KRROODPosition, [KRROODPosition(x=0.0, y=0.0, z=0.0)])
    large_x = IsGreaterThan(value=position.x, bound=1.0)
    large_y = IsGreaterThan(value=position.y, bound=1.0)
    condition = and_(large_x, large_y)

    false_statements = get_false_statements(condition)
    true_statements = get_true_statements(condition)

    assert [statement._id_ for statement in false_statements] == [large_x._id_]
    assert true_statements == []


# %% one evaluation of a condition


def test_a_conjunction_whose_conjunct_fails_is_not_satisfied():
    """
    One evaluation of a conjunction reports it unsatisfied and names the conjunct that
    fails.
    """
    position = variable(KRROODPosition, [KRROODPosition(x=2.0, y=0.0, z=0.0)])
    large_x = IsGreaterThan(value=position.x, bound=1.0)
    large_y = IsGreaterThan(value=position.y, bound=1.0)

    evaluation = ConditionEvaluation.from_condition(condition=and_(large_x, large_y))

    assert evaluation.is_satisfied is False
    assert [statement._id_ for statement in evaluation.false_statements] == [
        large_y._id_
    ]
    assert [statement._id_ for statement in evaluation.true_statements] == [
        large_x._id_
    ]


def test_a_disjunction_with_a_false_disjunct_is_satisfied():
    """
    A false disjunct is a false statement, yet the disjunction holds.
    """
    position = variable(KRROODPosition, [KRROODPosition(x=2.0, y=0.0, z=0.0)])
    large_x = IsGreaterThan(value=position.x, bound=1.0)
    large_y = IsGreaterThan(value=position.y, bound=1.0)

    evaluation = ConditionEvaluation.from_condition(condition=or_(large_y, large_x))

    assert evaluation.is_satisfied is True
    assert [statement._id_ for statement in evaluation.false_statements] == [
        large_y._id_
    ]


def test_a_condition_is_evaluated_in_the_given_state():
    """
    The state reads the position's ``x`` as large, though the live position's is not.
    """
    position = KRROODPosition(x=0.0, y=0.0, z=0.0)
    condition = IsGreaterThan(value=literal(position).x, bound=1.0)
    state = AssignedValueReader(
        assigned_values={AttributeTerm(instance=position, attribute_name="x"): 2.0}
    )

    assert (
        ConditionEvaluation.from_condition(condition=condition).is_satisfied,
        ConditionEvaluation.from_condition(
            condition=condition, state=state
        ).is_satisfied,
    ) == (False, True)
    assert [
        statement._id_ for statement in get_true_statements(condition, state=state)
    ] == [condition._id_]


def test_whether_a_condition_holds_is_decided_in_the_given_state():
    """
    The state reads the position's ``x`` as large, though the live position's is not.
    """
    position = KRROODPosition(x=0.0, y=0.0, z=0.0)
    condition = literal(position).x > 1.0
    state = AssignedValueReader(
        assigned_values={AttributeTerm(instance=position, attribute_name="x"): 2.0}
    )

    assert (
        evaluate_condition(condition),
        evaluate_condition(condition, state=state),
    ) == (False, True)


# %% conditions that are plain booleans


@pytest.mark.parametrize("condition", [True, False])
def test_a_boolean_condition_is_satisfied_when_it_is_true(condition):
    """
    A plain boolean is evaluated like any other condition: it holds exactly when it is
    ``True``.
    """
    assert (
        ConditionEvaluation.from_condition(condition=condition).is_satisfied
        is condition
    )


def test_a_false_boolean_condition_is_its_own_false_statement():
    """
    ``False`` is the one statement of the condition, and it does not hold.
    """
    evaluation = ConditionEvaluation.from_condition(condition=False)
    assert [statement._value_ for statement in evaluation.false_statements] == [False]
