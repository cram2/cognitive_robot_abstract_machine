"""
This module defines the core logical operators for the Entity Query Language.

It contains implementations of basic logical operations like AND, OR, and NOT.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from typing_extensions import Iterable, TYPE_CHECKING, Type

from ..core.base_expressions import (
    TruthValueOperator,
    UnaryExpression,
    Bindings,
    OperationResult,
    BinaryExpression,
)

if TYPE_CHECKING:
    from ..factories import ConditionType


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
        sources: Bindings,
    ) -> Iterable[OperationResult]:

        for v in self._child_._evaluate_(sources, parent=self):
            self._is_false_ = v.is_true
            yield OperationResult(v.bindings, self._is_false_, self)


@dataclass(eq=False, repr=False)
class LogicalBinaryOperator(LogicalOperator, BinaryExpression, ABC):
    """
    Abstract base class for logical operators that take two operands (i.e. have two children) only.
    """

    def evaluate_right(self, sources: Bindings) -> Iterable[OperationResult]:
        """
        Evaluate the right operand.

        :param sources: The current bindings to use during evaluation.
        :return: The new bindings after evaluating the right operand.
        """
        for right_value in self.right._evaluate_(sources, parent=self):
            self._is_false_ = right_value.is_false
            yield OperationResult(
                right_value.bindings, self._is_false_, self, right_value
            )


@dataclass(eq=False, repr=False)
class AND(LogicalBinaryOperator):
    """
    A symbolic AND operation that can be used to combine multiple symbolic expressions.
    """

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:

        for left_value in self.left._evaluate_(sources, parent=self):
            self._is_false_ = left_value.is_false
            if self._is_false_:
                yield OperationResult(
                    left_value.bindings, self._is_false_, self, left_value
                )
            else:
                yield from self.evaluate_right(left_value.bindings)


@dataclass(eq=False, repr=False)
class OR(LogicalBinaryOperator):
    """
    A logical OR operator that evaluates the right operand only when the left operand is False. It is like an 'ElseIf`.
    """

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:
        """
        Evaluate the left operand, if it is False, then evaluate the right operand.

        :param sources: The current bindings to use for evaluation.
        :return: The new bindings after evaluating the left operand (and possibly right operand).
        """
        for left_value in self.left._evaluate_(sources, parent=self):
            if left_value.is_false:
                yield from self.evaluate_right(left_value.bindings)
            else:
                self._is_false_ = False
                yield OperationResult(
                    left_value.bindings, self._is_false_, self, left_value
                )


def chained_logic(
    operator: Type[LogicalBinaryOperator], *conditions: ConditionType
) -> LogicalOperator:
    """
    A chian of logic operation over multiple conditions, e.g. cond1 | cond2 | cond3.

    :param operator: The symbolic operator to apply between the conditions.
    :param conditions: The conditions to be chained.
    """
    prev_operation = None
    for condition in conditions:
        if prev_operation is None:
            prev_operation = condition
            continue
        prev_operation = operator(prev_operation, condition)
    return prev_operation
