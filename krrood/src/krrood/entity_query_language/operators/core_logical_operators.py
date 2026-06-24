"""
This module defines the core logical operators for the Entity Query Language.

It contains implementations of basic logical operations like AND, OR, and NOT.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from typing_extensions import Iterable, Optional, TYPE_CHECKING, Type

from krrood.entity_query_language.core.base_expressions import (
    TruthValueOperator,
    UnaryExpression,
    OperationResult,
    BinaryExpression,
    SymbolicExpression,
    BinaryExpression, SymbolicExpression,
)
from krrood.entity_query_language.core.variable import Literal

if TYPE_CHECKING:
    from krrood.entity_query_language.factories import ConditionType


@dataclass(eq=False, repr=False)
class LogicalOperator(TruthValueOperator, ABC):
    """
    A symbolic operation that can be used to combine multiple symbolic expressions using logical constraints on their
    truth values. Examples are conjunction (AND), disjunction (OR), negation (NOT), and conditional quantification
    (ForALL, Exists).
    """

    @property
    def _name_(self):
        return self.__class__.__name__


@dataclass(eq=False, repr=False)
class Not(LogicalOperator, UnaryExpression):
    """
    The logical negation of a symbolic expression. Its truth value is the opposite of its child's truth value. This is
    used when you want bindings that satisfy the negated condition (i.e., that doesn't satisfy the original condition).
    """

    def _evaluate__(
        self,
        sources: OperationResult,
    ) -> Iterable[OperationResult]:

        for v in self._evaluate_child_as_condition_(self._child_, sources):
            is_false = v.is_true
            yield OperationResult(v.bindings, is_false, self)


@dataclass(eq=False, repr=False)
class LogicalBinaryOperator(LogicalOperator, BinaryExpression, ABC):
    """
    Abstract base class for logical operators that take two operands (i.e. have two children) only.
    """

    def evaluate_right(self, sources: OperationResult) -> Iterable[OperationResult]:
        """
        Evaluate the right operand.

        :param sources: The current OperationResult to use during evaluation.
        :return: The new bindings after evaluating the right operand.
        """
        for right_value in self._evaluate_child_as_condition_(self.right, sources):
            is_false = right_value.is_false
            yield OperationResult(right_value.bindings, is_false, self, right_value)


@dataclass(eq=False, repr=False)
class AND(LogicalBinaryOperator):
    """
    A symbolic AND operation that can be used to combine multiple symbolic expressions.
    """

    def _evaluate__(
        self,
        sources: OperationResult,
    ) -> Iterable[OperationResult]:

        yielded: bool = False
        for left_value in self._evaluate_child_as_condition_(self.left, sources):
            is_false = left_value.is_false
            yielded = True
            if is_false:
                yield OperationResult(left_value.bindings, is_false, self, left_value)
            else:
                yield from self.evaluate_right(left_value)
        if not yielded:
            # Negation as failure: no variable value satisfied the condition. So the whole condition is False.
            yield OperationResult(sources.bindings, True, self, sources)


@dataclass(eq=False, repr=False)
class OR(LogicalBinaryOperator):
    """
    A logical OR operator that evaluates the right operand only when the left operand is False. It is like an 'ElseIf`.
    """

    def _evaluate__(
        self,
        sources: OperationResult,
    ) -> Iterable[OperationResult]:
        """
        Evaluate the left operand, if it is False, then evaluate the right operand.

        :param sources: The current OperationResult to use for evaluation.
        :return: The new bindings after evaluating the left operand (and possibly right operand).
        """
        yielded: bool = False
        for left_value in self._evaluate_child_as_condition_(self.left, sources):
            yielded = True
            if left_value.is_false:
                yield from self.evaluate_right(left_value)
            else:
                yield OperationResult(left_value.bindings, False, self, left_value)
        if not yielded:
            # Negation as failure: no variable value satisfied the condition. So the whole condition is False.
            yield OperationResult(sources.bindings, True, self, sources)


def chained_logic(
    operator: Type[LogicalBinaryOperator], *conditions: ConditionType
) -> SymbolicExpression:
    """
    A chain of logic operation over multiple conditions, e.g. cond1 | cond2 | cond3.

    :param operator: The symbolic operator to apply between the conditions.
    :param conditions: The conditions to be chained.
    """
    prev_operation = None
    for condition in conditions:
        if prev_operation is None:
            prev_operation = condition
            continue
        prev_operation = operator(prev_operation, condition)
    if not isinstance(prev_operation, SymbolicExpression):
        prev_operation = Literal(_value_=True)
    return prev_operation
