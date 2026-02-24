"""
Conclusion selection operators for the Entity Query Language.

This module provides operators that control which conclusions from operands propagate, such as ExceptIf,
Alternative, and Next.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing_extensions import Optional, Iterable, TYPE_CHECKING, Self

from .conclusion import Conclusion
from ..operators.set_operations import Union as EQLUnion
from ..operators.core_logical_operators import (
    LogicalBinaryOperator,
    OR,
    AND,
    chained_logic,
    LogicalOperator,
)
from ..core.base_expressions import (
    Bindings,
    OperationResult,
    SymbolicExpression,
    TruthValueOperator,
)

if TYPE_CHECKING:
    from ..factories import ConditionType


@dataclass(eq=False)
class ConclusionSelector(TruthValueOperator, ABC):
    """
    Base class for operators that selects the conclusions to pass through from it's operands' conclusions.
    """

    @classmethod
    def create_and_update_rule_tree(
        cls,
        *conditions: ConditionType,
    ) -> Self:
        """
        Create a new RDR rule (e.g., Refinement, Alternative, Next) and add it to the current rule tree.

        Each provided condition is chained with AND, and the resulting branch is
        connected via ElseIf/Next to the current node, representing an alternative/next path.

        :param conditions: Conditions to chain with AND to create the new condition expression.
        :returns: The conditions root after attaching the new condition to the rule tree.
        """
        new_condition = chained_logic(AND, *conditions)

        current_context = cls._get_current_context_condition()

        prev_parent = current_context._parent_

        new_context = cls._create_between_two_expressions(
            current_context, new_condition
        )

        if new_context is not current_context:
            prev_parent._replace_child_(current_context, new_context)

        return new_condition

    @classmethod
    @abstractmethod
    def _get_current_context_condition(
        cls,
    ) -> SymbolicExpression:
        """
        :return: The condition that will be connected to the new condition via this ConclusionSelector.
        """
        ...

    @classmethod
    @abstractmethod
    def _create_between_two_expressions(
        cls,
        current_condition: SymbolicExpression,
        new_condition: SymbolicExpression,
    ) -> Self:
        """
        Connects the new condition by a ConclusionSelector to the current condition.

        :param current_condition: The current condition in the expression tree.
        :param new_condition: The new condition to be added to the rule tree.
        """
        ...


@dataclass(eq=False)
class Refinement(LogicalBinaryOperator, ConclusionSelector):
    """
    Conditional branch that yields left unless the right side produces values.

    This encodes an "except if" behavior: when the right condition matches,
    the left branch's conclusions/outputs are excluded; otherwise, left flows through.
    """

    right_yielded: bool = False
    """
    Whether the right branch has yielded any True results.
    """

    def _evaluate__(
        self,
        sources: Optional[Bindings] = None,
    ) -> Iterable[OperationResult]:
        """
        Evaluate the ExceptIf condition and yield the results.
        """
        for left_value in self.left._evaluate_(sources, self):
            if left_value.is_false:
                yield from self.get_operation_result_and_clear_conclusion(left_value)
                continue
            yield from self.evaluate_right(left_value.bindings)
            if not self.right_yielded:
                # If the right branch didn't yield any True values, propagate the left branch's conclusions.
                yield from self.get_operation_result_and_clear_conclusion(left_value)

    def evaluate_right(self, bindings: Bindings) -> Iterable[OperationResult]:
        """
        Evaluate the right branch of the ExceptIf condition and yield the results. In addition, update the right_yielded
         flag and the conclusion if the right branch is True.

        :param bindings: The current bindings from the left evaluation to evaluate the right branch with.
        :return: The results of evaluating the right branch.
        """
        self.right_yielded = False
        for right_value in self.right._evaluate_(bindings, parent=self):
            if right_value.is_true:
                self.right_yielded = True
            yield from self.get_operation_result_and_clear_conclusion(right_value)

    def get_operation_result_and_clear_conclusion(
        self, result: OperationResult
    ) -> Iterable[OperationResult]:
        self._is_false_ = result.is_false
        if result.is_true:
            self._conclusions_.update(result.operand._conclusions_)
        yield OperationResult(result.bindings, self._is_false_, self, result)
        self._conclusions_.clear()

    @classmethod
    def _get_current_context_condition(
        cls,
    ) -> ConditionType:
        return SymbolicExpression._current_parent_in_context_stack_()

    @classmethod
    def _create_between_two_expressions(
        cls,
        current_condition: SymbolicExpression,
        new_condition: LogicalOperator,
    ) -> Self:
        """
        Constructs a new rule from the provided rule type and the current conditions root.

        :param current_condition: The current conditions root in the expression tree.
        :param new_condition: The new condition to be added to the rule tree.
        """
        return cls(current_condition, new_condition)


@dataclass(eq=False)
class Alternative(OR, ConclusionSelector):
    """
    A conditional branch that behaves like an "else if" clause where the left branch
    is selected if it is true, otherwise the right branch is selected if it is true else
    none of the branches are selected.
    """

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:
        for output in OR._evaluate__(self, sources):
            if output.is_true:
                self._conclusions_.update(
                    output.previous_operation_result.operand._conclusions_
                )
            yield output
            self._conclusions_.clear()

    @classmethod
    def _get_current_context_condition(
        cls,
    ) -> ConditionType:
        current_context = SymbolicExpression._current_parent_in_context_stack_()
        current_context_parent = current_context._parent_
        if (
            isinstance(current_context_parent, Refinement)
            and current_context is current_context_parent.right
        ):
            return current_context
        return current_context._conditions_root_

    @classmethod
    def _create_between_two_expressions(
        cls,
        current_condition: SymbolicExpression,
        new_condition: LogicalOperator,
    ) -> Self:
        return cls(current_condition, new_condition)


@dataclass(eq=False)
class Next(EQLUnion, ConclusionSelector):
    """
    A Union conclusion selector that always evaluates all its operands and combines their results.
    """

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:
        for output in EQLUnion._evaluate__(self, sources):
            if output.is_true:
                self._conclusions_.update(
                    output.previous_operation_result.operand._conclusions_
                )
            yield output
            self._conclusions_.clear()

    @classmethod
    def _get_current_context_condition(
        cls,
    ) -> ConditionType:
        current_context = SymbolicExpression._current_parent_in_context_stack_()
        return current_context._conditions_root_

    @classmethod
    def _create_between_two_expressions(
        cls,
        current_condition: SymbolicExpression,
        new_condition: SymbolicExpression,
    ) -> Self:
        match current_condition:
            case cls():
                current_condition.add_child(new_condition)
                new_conditions_root = current_condition
            case _:
                new_conditions_root = cls((current_condition, new_condition))
        return new_conditions_root
