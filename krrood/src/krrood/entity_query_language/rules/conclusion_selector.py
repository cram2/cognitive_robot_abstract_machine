"""
Conclusion selection operators for the Entity Query Language.

This module provides operators that control which conclusions from operands propagate,
such as ExceptIf, Alternative, and Next.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing_extensions import Iterable, TYPE_CHECKING, Self, Optional

from krrood.entity_query_language.exceptions import SelfReferentialInsertionError
from krrood.entity_query_language.rules.conclusion import Conclusion
from krrood.entity_query_language.operators.set_operations import Union as EQLUnion
from krrood.entity_query_language.operators.core_logical_operators import (
    LogicalBinaryOperator,
    OR,
    AND,
    chained_logic,
    LogicalOperator,
)
from krrood.entity_query_language.core.base_expressions import (
    Bindings,
    OperationResult,
    SymbolicExpression,
    TruthValueOperator,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.factories import ConditionType


@dataclass(eq=False)
class ConclusionSelector(TruthValueOperator, ABC):
    """
    Base class for operators that selects the conclusions to pass through from it's
    operands' conclusions.
    """

    @classmethod
    def create_and_update_rule_tree(
        cls,
        *conditions: ConditionType,
    ) -> SymbolicExpression:
        """
        Create a new RDR rule (e.g., Refinement, Alternative, Next) and add it to the
        current rule tree.

        Each provided condition is chained with AND, and the resulting branch is
        connected via ElseIf/Next to the current node, representing an alternative/next
        path.

        The anchor (the node the new branch attaches to) is taken from the enclosing
        ``with`` context. For dynamic growth without a ``with`` context, use
        :meth:`insert_at` with an explicit anchor.

        :param conditions: Conditions to chain with AND to create the new condition
            expression.
        :returns: The conditions root after attaching the new condition to the rule
            tree.
        """
        return cls.insert_at(cls._get_current_context_condition(), *conditions)

    @classmethod
    def insert_at(
        cls,
        anchor: SymbolicExpression,
        *conditions: ConditionType,
    ) -> SymbolicExpression:
        """
        Attach a new branch to an explicitly given ``anchor``.

        This is the explicit-anchor counterpart of :meth:`create_and_update_rule_tree`,
        used to grow a live rule-tree DAG (e.g. when an RDR inserts a refinement or
        alternative after observing a misclassification). Conditions are chained with
        AND; the new branch is spliced in between ``anchor`` and the parent the asking
        branch reaches it by, which the enclosing ``with`` context supplies when one
        anchors on ``anchor`` and the structural parent supplies otherwise.

        Any condition already in a tree (has a ``_parent_``) is replaced with the node from
        :meth:`~krrood.entity_query_language.core.base_expressions.SymbolicExpression._node_for_new_position_`
        so splicing it in cannot corrupt the original's ``_parent_``.

        :param anchor: The existing condition node the new branch connects to.
        :param conditions: Conditions to chain with AND into the new branch.
        :returns: The newly created condition node (attach conclusions to it via ``with``).
        """
        cleaned_conditions = []
        for condition in conditions:
            if (
                isinstance(condition, SymbolicExpression)
                and condition._parent_ is not None
            ):
                condition = condition._node_for_new_position_()
            cleaned_conditions.append(condition)
        new_condition = chained_logic(AND, *cleaned_conditions)
        # A single condition returned directly by chained_logic may still carry a parent from the
        # pre-cleaning step; detach again if needed — idempotent for a parentless node.
        if (
            isinstance(new_condition, SymbolicExpression)
            and new_condition._parent_ is not None
        ):
            new_condition = new_condition._node_for_new_position_()

        # Splice above the parent the asking branch reaches the anchor by. A shared anchor
        # keeps whichever parent was attached first as its structural one, which may belong to
        # an unrelated branch entirely, so the enclosing ``with`` context decides instead and
        # the structural parent only stands in when no enclosing context anchors on it.
        anchor_context = SymbolicExpression._rule_tree_context_anchored_on_(anchor)
        previous_parent = (
            anchor._parent_ if anchor_context is None else anchor_context.owning_parent
        )

        # Only raise when the anchor is already established in a rule tree (has parents).
        # A freshly-created anchor with no parents indicates _conditions_root_ returned a node
        # from a different, unrelated part of the DAG, which is a different underlying issue —
        # not the self-referential insertion bug we guard against here.
        if new_condition is anchor and anchor._parents_:
            raise SelfReferentialInsertionError(anchor=anchor)

        new_context = cls._create_between_two_expressions(anchor, new_condition)

        if new_context is not anchor and previous_parent is not None:
            previous_parent._replace_child_(anchor, new_context)

        # The splice moved the anchor under the node just created, so the asking branch now
        # reaches it by that edge; a later edit in the same context must splice above it.
        if new_context is not anchor and anchor_context is not None:
            anchor_context.owning_parent = new_context

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

    This encodes an "except if" behavior: when the right condition matches, the left
    branch's conclusions/outputs are excluded; otherwise, left flows through.
    """

    right_yielded: bool = False
    """
    Whether the right branch has yielded any True results.
    """

    def _evaluate__(
        self,
        sources: Optional[OperationResult] = None,
    ) -> Iterable[OperationResult]:
        """
        Evaluate the ExceptIf condition and yield the results.
        """
        for left_value in self._evaluate_child_as_condition_(self.left, sources):
            if left_value.is_false:
                yield from self.get_operation_result_and_clear_conclusion(left_value)
                continue
            yield from self.evaluate_right(left_value)
            if not self.right_yielded:
                # If the right branch didn't yield any True values, propagate the left branch's conclusions.
                yield from self.get_operation_result_and_clear_conclusion(left_value)

    def evaluate_right(self, left_value: OperationResult) -> Iterable[OperationResult]:
        """
        Evaluate the right branch of the ExceptIf condition and yield the results.

        In addition, update the right_yielded flag and the conclusion if the right
        branch is True.

        :param left_value: The OperationResult from the left evaluation to evaluate the
            right branch with.
        :return: The results of evaluating the right branch.
        """
        self.right_yielded = False
        for right_value in self._evaluate_child_as_condition_(self.right, left_value):
            if right_value.is_true:
                self.right_yielded = True
            yield from self.get_operation_result_and_clear_conclusion(right_value)

    def get_operation_result_and_clear_conclusion(
        self, result: OperationResult
    ) -> Iterable[OperationResult]:
        left_branch_failed = result.operand is self.left and result.is_false
        if result.is_true:
            self._conclusions_.update(result.operand._conclusions_)
        yield self._build_operation_result_with_truth_(
            not left_branch_failed, result.bindings, result
        )
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
        Constructs a new rule from the provided rule type and the current conditions
        root.

        :param current_condition: The current conditions root in the expression tree.
        :param new_condition: The new condition to be added to the rule tree.
        """
        return cls(current_condition, new_condition)


@dataclass(eq=False)
class Alternative(OR, ConclusionSelector):
    """
    A conditional branch that behaves like an "else if" clause where the left branch is
    selected if it is true, otherwise the right branch is selected if it is true else
    none of the branches are selected.
    """

    def _evaluate__(
        self,
        sources: OperationResult,
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
    A Union conclusion selector that always evaluates all its operands and combines
    their results.
    """

    def _evaluate__(
        self,
        sources: OperationResult,
    ) -> Iterable[OperationResult]:
        for child in self._operation_children_:
            for child_result in self._evaluate_child_as_condition_(child, sources):
                output = self._build_operation_result_with_truth_(
                    child_result.is_true, child_result.bindings, child_result
                )
                if output.is_true:
                    self._conclusions_.update(child_result.operand._conclusions_)
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
