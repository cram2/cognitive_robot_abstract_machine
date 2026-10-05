"""
Rewiring the expression graph must find an expression by identity.

Comparing symbolic expressions with ``==`` builds a comparison instead of comparing
them, so a lookup that relies on it picks whichever expression comes first. The parent-
removal tests drive the private graph methods directly, because no public operation
reaches the parent-removal step while the removed expression is still recorded on both
sides.
"""

from krrood.entity_query_language.factories import case_when, variable

from ...dataset.example_classes import KRROODPosition
from ...dataset.value_comparisons import IsGreaterThan


def identifiers(expressions):
    """
    :param expressions: The expressions to identify.
    :return: The identifiers of the expressions, in order.
    """
    return [expression._id_ for expression in expressions]


# %% removing a parent


def test_detaching_a_child_keeps_its_siblings():
    position = variable(KRROODPosition, [])
    x, y = position.x, position.y
    predicate = IsGreaterThan(x, y)

    y._parent_ = None

    assert identifiers(predicate._children_) == identifiers([x])


def test_detaching_a_parent_keeps_the_other_parents():
    position = variable(KRROODPosition, [])
    other_position = variable(KRROODPosition, [])
    x, y = position.x, position.y

    y._replace_child_(position, other_position)

    assert identifiers(position._parents_) == identifiers([x])


# %% replacing a child


def test_replacing_a_child_with_its_sibling_leaves_the_sibling_once():
    position = variable(KRROODPosition, [])
    x, y = position.x, position.y
    predicate = IsGreaterThan(x, y)

    predicate._replace_child_(x, y)

    assert identifiers(predicate._children_) == identifiers([y])


def test_replacing_a_child_replaces_every_branch_of_a_case_holding_it():
    position = variable(KRROODPosition, [])
    y, z = position.y, position.z
    case = case_when(position.x > 0, y, y)

    case._replace_child_(y, z)

    assert identifiers([case.then_value, case.else_value]) == identifiers([z, z])
