from __future__ import annotations

from .core.base_expressions import OperationResult, SymbolicExpression
from .core.mapped_variable import Attribute
from .core.variable import Variable
from .factories import ConditionType
from .utils import is_iterable, T

"""
User interface (grammar & vocabulary) for entity query language.
"""
import operator

from typing_extensions import (
    Any,
    Optional,
    Union,
    Iterable,
    Type,
    Callable,
    TYPE_CHECKING,
    List,
)


def get_conditioned_statements(
    statement, condition: Callable[OperationResult, bool]
) -> List[SymbolicExpression]:
    condition_results = []
    for node in [
        s
        for s in statement._children_
        if not isinstance(s, (Variable, Attribute))
        # or type(s._predicate_type_) is PredicateType
    ]:
        node_result = node.evaluate()
        if condition(node_result):
            condition_results.append(node)
    if statement in condition_results:
        condition_results.remove(statement)

    return condition_results


def get_false_statements(statement):
    """
    The false statements of all statements of this condition.

    :return: The false statements of all statements of this condition.
    """
    return get_conditioned_statements(statement, lambda x: not x == [])


def get_true_statements(statement):
    """
    The true statements of all statements of this condition.

    :return: The true statements of this condition.
    """
    return get_conditioned_statements(statement, lambda x: x == [])


def evaluate_condition(condition: ConditionType) -> bool:
    """
    Evaluates the condition to True or False.

    :param condition: The condition to evaluate.
    :return: True if there is a possible solution, False otherwise.
    """
    if type(condition) is bool:
        return condition
    results = list(condition.evaluate())
    return any(results)
