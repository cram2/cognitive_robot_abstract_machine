"""
Conclusion selection operators for the Entity Query Language.

This module provides operators that control which conclusions from operands propagate,
such as ExceptIf, Alternative, and Next.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing_extensions import Any, Iterable, TYPE_CHECKING, Self

from krrood.entity_query_language.exceptions import SelfReferentialInsertionError
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.utils import (
    cartesian_product_while_passing_the_bindings_around,
)
from krrood.patterns.classproperty import classproperty
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


@dataclass(kw_only=True, eq=False)
class ConclusionSelector(TruthValueOperator, ABC):
    """
    Base class for operators that selects the conclusions to pass through from it's
    operands' conclusions.
    """

    @classproperty
    def grows_rule_tree_inward(cls) -> bool:
        """
        Whether adding a branch of this type re-anchors it at a fixed base node, growing
        the rule tree inward, rather than wrapping the current root.
        """
        return False

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
        return cls.insert_at(*conditions, anchor=cls._get_current_context_condition())

    @classmethod
    def insert_at(
        cls,
        *conditions: ConditionType,
        anchor: SymbolicExpression,
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
        so splicing it in cannot corrupt the original's ``_parent_``. The anchor is
        cleaned the same way, by
        :meth:`~krrood.entity_query_language.core.base_expressions.SymbolicExpression._parent_outside_`:
        a node can be one rule's whole condition and part of what another rule's
        condition is written over, and the parent it has inside the incoming branch is
        not an edge this splice may re-point.

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
        # the structural parent only stands in when no enclosing context anchors on it -- and
        # then only one from outside the branch being added, since a parent inside it is part
        # of what is being spliced in rather than an edge the surrounding graph holds.
        anchor_context = SymbolicExpression._rule_tree_context_anchored_on_(
            condition=anchor
        )
        previous_parent = (
            anchor._parent_outside_(branch=new_condition)
            if anchor_context is None
            else anchor_context.owning_parent
        )

        # Only raise when the anchor is already established in a rule tree (has parents).
        # A freshly-created anchor with no parents indicates _conditions_root_ returned a node
        # from a different, unrelated part of the DAG, which is a different underlying issue —
        # not the self-referential insertion bug we guard against here.
        if new_condition is anchor and anchor._parents_:
            raise SelfReferentialInsertionError(anchor=anchor)

        new_context = cls._create_between_two_expressions(
            current_condition=anchor, new_condition=new_condition
        )

        if new_context is not anchor and previous_parent is not None:
            previous_parent._replace_child_(old_child=anchor, new_child=new_context)

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
        *,
        current_condition: SymbolicExpression,
        new_condition: SymbolicExpression,
    ) -> Self:
        """
        Connects the new condition by a ConclusionSelector to the current condition.

        :param current_condition: The current condition in the expression tree.
        :param new_condition: The new condition to be added to the rule tree.
        """
        ...


@dataclass(kw_only=True, eq=False)
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

    @classproperty
    def grows_rule_tree_inward(cls) -> bool:
        """
        Like :attr:`ConclusionSelector.grows_rule_tree_inward`, but true: a refinement
        is anchored at the rule it refines.
        """
        return True

    def _evaluate__(
        self,
        *,
        sources: OperationResult | None = None,
    ) -> Iterable[OperationResult]:
        """
        Evaluate the ExceptIf condition and yield the results.
        """
        for left_value in self._evaluate_child_as_condition_(
            child=self.left, sources=sources
        ):
            if left_value.is_false:
                yield from self.get_operation_result_and_clear_conclusion(
                    result=left_value
                )
                continue
            yield from self.evaluate_right(sources=left_value)
            if not self.right_yielded:
                # If the right branch didn't yield any True values, propagate the left branch's conclusions.
                yield from self.get_operation_result_and_clear_conclusion(
                    result=left_value
                )

    def evaluate_right(self, *, sources: OperationResult) -> Iterable[OperationResult]:
        """
        Evaluate the right branch of the ExceptIf condition and yield the results.

        In addition, update the right_yielded flag and the conclusion if the right
        branch is True.

        :param left_value: The OperationResult from the left evaluation to evaluate the
            right branch with.
        :return: The results of evaluating the right branch.
        """
        self.right_yielded = False
        for right_value in self._evaluate_child_as_condition_(
            child=self.right, sources=sources
        ):
            if right_value.is_true:
                self.right_yielded = True
            yield from self.get_operation_result_and_clear_conclusion(
                result=right_value
            )

    def get_operation_result_and_clear_conclusion(
        self, *, result: OperationResult
    ) -> Iterable[OperationResult]:
        left_branch_failed = result.operand is self.left and result.is_false
        if result.is_true:
            self._conclusions_.update(result.operand._conclusions_)
        yield self._build_operation_result_with_truth_(
            truth=not left_branch_failed, bindings=result.bindings, child_result=result
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
        *,
        current_condition: SymbolicExpression,
        new_condition: LogicalOperator,
    ) -> Self:
        """
        Constructs a new rule from the provided rule type and the current conditions
        root.

        :param current_condition: The current conditions root in the expression tree.
        :param new_condition: The new condition to be added to the rule tree.
        """
        return cls(left=current_condition, right=new_condition)


@dataclass(kw_only=True, eq=False)
class ValueCombination:
    """
    One value for each of several variables, equal to another combination only when it
    holds the very same objects in the same order, since a variable ranges over the
    objects of its domain and two equal objects are still two values.
    """

    values: tuple[Any, ...]
    """
    The value of each variable, in the order of the variables.
    """

    def __hash__(self) -> int:
        """
        :return: A hash of the identities of the values.
        """
        return hash(tuple(id(value) for value in self.values))

    def __eq__(self, other: object) -> bool:
        """
        :return: Whether *other* holds the very same objects in the same order.
        """
        return (
            isinstance(other, ValueCombination)
            and len(self.values) == len(other.values)
            and all(mine is theirs for mine, theirs in zip(self.values, other.values))
        )


@dataclass(kw_only=True)
class ConcludedValueCombinations:
    """
    The combinations of values of :attr:`Alternative._variables_shared_by_conclusions_`
    that one branch of an :class:`Alternative` already concluded for.
    """

    variables: tuple[Variable, ...]
    """
    The variables whose values make up a combination.
    """

    combinations: set[ValueCombination] = field(default_factory=set)
    """
    The combinations concluded for so far.
    """

    def _combination_in_(self, *, bindings: Bindings) -> ValueCombination:
        """
        :param bindings: Bindings holding a value for every one of :attr:`variables`.
        :return: The combination of those values.
        """
        return ValueCombination(
            values=tuple(bindings[variable._id_] for variable in self.variables)
        )

    def add(self, *, bindings: Bindings) -> None:
        """
        :param bindings: Bindings holding a value for every one of :attr:`variables`,
            whose combination is now concluded for.
        """
        self.combinations.add(self._combination_in_(bindings=bindings))

    def contains(self, *, bindings: Bindings) -> bool:
        """
        :param bindings: Bindings holding a value for every one of :attr:`variables`.
        :return: Whether their combination was already concluded for.
        """
        return self._combination_in_(bindings=bindings) in self.combinations


@dataclass(kw_only=True, eq=False)
class Alternative(OR, ConclusionSelector):
    """
    A conditional branch that behaves like an "else if" clause where the left branch is
    selected if it is true, otherwise the right branch is selected if it is true else
    none of the branches are selected.
    """

    def _evaluate__(
        self,
        *,
        sources: OperationResult,
    ) -> Iterable[OperationResult]:
        """
        Overrides :meth:`OR._evaluate__`: the right branch concludes only for the values
        of :attr:`_variables_shared_by_conclusions_` that the left branch holds for none
        of, whatever else either branch binds.

        Branches whose conclusions share no variable fall back to
        :meth:`_evaluate_for_every_failing_binding_`.
        """
        shared_variables = self._variables_shared_by_conclusions_
        if not shared_variables:
            yield from self._evaluate_for_every_failing_binding_(sources=sources)
            return
        concluded = ConcludedValueCombinations(variables=shared_variables)
        for left_value in self._evaluate_child_as_condition_(
            child=self.left, sources=sources
        ):
            if left_value.is_false:
                continue
            for combination in cartesian_product_while_passing_the_bindings_around(
                shared_variables, left_value
            ):
                concluded.add(bindings=combination.bindings)
                yield from self._result_with_conclusions_of_(
                    branch_result=left_value, bindings=combination.bindings
                )
        for right_value in self._evaluate_child_as_condition_(
            child=self.right, sources=sources
        ):
            if right_value.is_false:
                yield self._build_operation_result_with_truth_(
                    truth=False, bindings=right_value.bindings, child_result=right_value
                )
                continue
            for combination in cartesian_product_while_passing_the_bindings_around(
                shared_variables, right_value
            ):
                if concluded.contains(bindings=combination.bindings):
                    continue
                yield from self._result_with_conclusions_of_(
                    branch_result=right_value, bindings=combination.bindings
                )

    def _evaluate_for_every_failing_binding_(
        self, *, sources: OperationResult
    ) -> Iterable[OperationResult]:
        """
        Evaluate as an :class:`OR`, with the right branch tried for every result of the
        left branch that fails, from the bindings that result holds.

        :param sources: The current bindings.
        :return: The results, carrying the conclusions of the branch that holds.
        """
        for output in OR._evaluate__(self, sources=sources):
            if output.is_true:
                self._conclusions_.update(
                    output.previous_operation_result.operand._conclusions_
                )
            yield output
            self._conclusions_.clear()

    @property
    def _variables_shared_by_conclusions_(self) -> tuple[Variable, ...]:
        """
        The variables in the concluded value of every conclusion of either branch, for
        whose values the branches are alternatives to each other, in the order the first
        conclusion's value names them.
        """
        conclusions = [
            conclusion
            for node in itertools.chain([self], self._descendants_)
            for conclusion in node._conclusions_
        ]
        if not conclusions:
            return ()
        first_conclusion, *other_conclusions = conclusions
        variable_ids_of_other_conclusions = [
            {variable._id_ for variable in conclusion.variables_in_concluded_value}
            for conclusion in other_conclusions
        ]
        return tuple(
            variable
            for variable in first_conclusion.variables_in_concluded_value
            if all(
                variable._id_ in variable_ids
                for variable_ids in variable_ids_of_other_conclusions
            )
        )

    def _result_with_conclusions_of_(
        self, *, branch_result: OperationResult, bindings: Bindings
    ) -> Iterable[OperationResult]:
        """
        :param branch_result: A result of one of the two branches that holds.
        :param bindings: The bindings of that result, extended by the values of
            :attr:`_variables_shared_by_conclusions_`.
        :return: This alternative's result for it, carrying the branch's conclusions.
        """
        self._conclusions_.update(branch_result.operand._conclusions_)
        yield self._build_operation_result_with_truth_(
            truth=True, bindings=bindings, child_result=branch_result
        )
        self._conclusions_.clear()

    @classmethod
    def _get_current_context_condition(
        cls,
    ) -> ConditionType:
        """
        :return: The refinement branch the current context belongs to, together with
            the alternatives and refinements already attached to it, or the conditions
            root outside of one.
        """
        current_context = SymbolicExpression._current_parent_in_context_stack_()
        branch = current_context
        while (
            isinstance(branch._parent_, (Alternative, Refinement))
            and branch is branch._parent_.left
        ):
            branch = branch._parent_
        if isinstance(branch._parent_, Refinement) and branch is branch._parent_.right:
            return branch
        return current_context._conditions_root_

    @classmethod
    def _create_between_two_expressions(
        cls,
        *,
        current_condition: SymbolicExpression,
        new_condition: LogicalOperator,
    ) -> Self:
        return cls(left=current_condition, right=new_condition)


@dataclass(kw_only=True, eq=False)
class Next(EQLUnion, ConclusionSelector):
    """
    A Union conclusion selector that always evaluates all its operands and combines
    their results.
    """

    def _evaluate__(
        self,
        *,
        sources: OperationResult,
    ) -> Iterable[OperationResult]:
        for child in self._operation_children_:
            for child_result in self._evaluate_child_as_condition_(
                child=child, sources=sources
            ):
                output = self._build_operation_result_with_truth_(
                    truth=child_result.is_true,
                    bindings=child_result.bindings,
                    child_result=child_result,
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
        *,
        current_condition: SymbolicExpression,
        new_condition: SymbolicExpression,
    ) -> Self:
        match current_condition:
            case cls():
                current_condition.add_child(child=new_condition)
                new_conditions_root = current_condition
            case _:
                new_conditions_root = cls(
                    _operation_children_=(current_condition, new_condition)
                )
        return new_conditions_root
