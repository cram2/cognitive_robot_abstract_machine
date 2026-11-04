"""
Logical quantifiers for the Entity Query Language.

This module provides quantified conditionals such as universal (ForAll) and existential
(Exists) operators that evaluate conditions over the values of a variable.
"""

from __future__ import annotations

import uuid
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

from typing_extensions import Iterable, Iterator

from krrood.entity_query_language.core.base_expressions import (
    OperationResult,
    Selectable,
    SymbolicExpression,
)
from krrood.entity_query_language.core.variable import Literal
from krrood.entity_query_language.evaluation_context import get_evaluation_context
from krrood.entity_query_language.operators.core_logical_operators import (
    LogicalBinaryOperator,
)
from krrood.entity_query_language.query.query import Query
from krrood.entity_query_language.utils import (
    cartesian_product_while_passing_the_bindings_around,
)


@dataclass(eq=False, repr=False)
class QuantifiedConditional(LogicalBinaryOperator, ABC):
    """
    This is the super class of the universal, and existential conditional operators.

    It is a binary logical operator that has a quantified variable and a condition on
    the values of that variable.

    A variable of the quantifier that also occurs elsewhere in the enclosing query,
    whether selected or used by another condition, is outer-visible: the quantifier is
    evaluated once per binding of the outer-visible variables. A variable that occurs
    only inside the quantifier, including inside the quantified variable's own domain
    expression, is local to it and is quantified together with the quantified variable.
    """

    @property
    def variable(self):
        return self.left

    @property
    def condition(self):
        return self.right

    def _evaluate__(
        self,
        sources: OperationResult,
    ) -> Iterable[OperationResult]:
        """
        Evaluate the quantifier once per binding of its outer-visible variables.

        :param sources: The current bindings.
        :return: The results of the quantifier for every outer binding.
        """
        outer_visible_variables = self._outer_visible_variables_()
        if not outer_visible_variables:
            yield from self._evaluate_for_outer_binding_(sources)
            return
        for outer_result in cartesian_product_while_passing_the_bindings_around(
            outer_visible_variables, sources
        ):
            yield from self._evaluate_for_outer_binding_(outer_result)

    @abstractmethod
    def _evaluate_for_outer_binding_(
        self, sources: OperationResult
    ) -> Iterable[OperationResult]:
        """
        Evaluate the quantifier under one binding of its outer-visible variables.

        :param sources: The bindings, with every outer-visible variable bound.
        :return: The results of the quantifier under these bindings.
        """

    def _outer_visible_variables_(self) -> tuple[Selectable, ...]:
        """
        :return: The outer-visible variables of this quantifier in the query being
            evaluated. The quantified variable is one of them when it also occurs
            elsewhere.
        """
        evaluation_context = get_evaluation_context()
        if evaluation_context is None:
            return self._compute_outer_visible_variables_()
        cache = evaluation_context.outer_visible_variables_cache
        if self._id_ not in cache:
            cache[self._id_] = tuple(
                weakref.ref(variable)
                for variable in self._compute_outer_visible_variables_()
            )
        return tuple(reference() for reference in cache[self._id_])

    def _compute_outer_visible_variables_(self) -> tuple[Selectable, ...]:
        """
        :return: The variables of this quantifier that also occur in its enclosing
            query outside this quantifier.
        """
        scope = self._enclosing_query_()
        if scope is None:
            return ()
        ids_outside_this_quantifier = {
            expression._id_
            for expression in _expressions_in_scope_(scope._children_, {self._id_})
        }
        return tuple(
            expression
            for expression in _expressions_in_scope_((self.variable, self.condition))
            if isinstance(expression, Selectable)
            and not isinstance(expression, Literal)
            and expression._id_ in ids_outside_this_quantifier
        )

    def _enclosing_query_(self) -> Query | None:
        """
        :return: The innermost compiled query containing this quantifier: the one being
            evaluated when there are several, as after a query is rebuilt around the
            same condition, else the most recently built one. None when this quantifier
            is not inside a query.
        """
        enclosing_queries = []
        # Tracked by object identity: every build of a query reuses its identifier.
        visited_objects = set()
        frontier = list(reversed(self._parents_))
        while frontier:
            expression = frontier.pop(0)
            if id(expression) in visited_objects:
                continue
            visited_objects.add(id(expression))
            if isinstance(expression, Query):
                enclosing_queries.append(expression)
            else:
                frontier.extend(reversed(expression._parents_))
        evaluation_context = get_evaluation_context()
        evaluated_query = (
            evaluation_context.outermost_query.node if evaluation_context else None
        )
        for query in enclosing_queries:
            if query is evaluated_query:
                return query
        return enclosing_queries[0] if enclosing_queries else None


def _expressions_in_scope_(
    roots: Iterable[SymbolicExpression], excluded_ids: set[uuid.UUID] | None = None
) -> Iterator[SymbolicExpression]:
    """
    :param roots: The expressions to start from.
    :param excluded_ids: Identifiers of the expressions to neither yield nor enter.
    :return: The roots and their descendants, each once. A nested query is yielded but
        not entered, since its variables belong to its own scope.
    """
    visited_ids = set(excluded_ids or ())
    stack = list(reversed(list(roots)))
    while stack:
        expression = stack.pop()
        if expression._id_ in visited_ids:
            continue
        visited_ids.add(expression._id_)
        yield expression
        if not isinstance(expression, Query):
            stack.extend(reversed(expression._children_))


@dataclass(eq=False, repr=False)
class ForAll(QuantifiedConditional):
    """
    This operator is the universal conditional operator.

    It returns bindings that satisfy the condition for all the values of the quantified
    variable. It is efficient as it ignores the bindings that don't satisfy the
    condition.
    """

    @cached_property
    def condition_unique_variable_ids(self) -> list[uuid.UUID]:
        return [
            v._id_
            for v in self.condition._unique_variables_.difference(
                self.left._unique_variables_
            )
        ]

    def _evaluate_for_outer_binding_(
        self,
        sources: OperationResult,
    ) -> Iterable[OperationResult]:
        solution_set = None

        for variable_result in self.variable._evaluate_(sources):
            if solution_set is None:
                solution_set = self.get_all_candidate_solutions(variable_result)
            else:
                solution_set = [
                    solution
                    for solution in solution_set
                    if self.evaluate_condition(
                        OperationResult({**solution, **variable_result.bindings})
                    )
                ]
            if not solution_set:
                solution_set = []
                break

        if solution_set is None:
            # The variable has no values: the condition holds for all of them vacuously.
            yield self._build_operation_result_with_truth_(True, sources.bindings)
            return
        if not solution_set:
            # Negation as failure: the condition fails for some value of the variable.
            yield self._build_operation_result_with_truth_(False, sources.bindings)
            return
        # Yield the remaining bindings (non-universal) merged with the incoming sources
        yield from [
            self._build_operation_result_with_truth_(True, sources.bindings | solution)
            for solution in solution_set
        ]

    def get_all_candidate_solutions(self, variable_result: OperationResult):
        values_that_satisfy_condition = []
        # Evaluate the condition under this particular universal value
        for condition_result in self._evaluate_child_as_condition_(
            self.condition, variable_result
        ):
            if condition_result.is_false:
                continue
            condition_bindings = {
                k: v
                for k, v in condition_result.bindings.items()
                if k in self.condition_unique_variable_ids
            }
            values_that_satisfy_condition.append(condition_bindings)
        return values_that_satisfy_condition

    def evaluate_condition(self, sources: OperationResult) -> bool:
        for condition_result in self._evaluate_child_as_condition_(
            self.condition, sources
        ):
            return condition_result.is_true
        return False

    def _invert_(self):
        return Exists(self.variable, self.condition._invert_())


@dataclass(eq=False, repr=False)
class Exists(QuantifiedConditional):
    """
    An existential checker that checks if a condition holds for any value of the
    variable given, the benefit of this is that it returns True if the condition holds
    for any value without getting all the condition values that hold for one specific
    value of the variable.
    """

    def _evaluate_for_outer_binding_(
        self,
        sources: OperationResult,
    ) -> Iterable[OperationResult]:
        for variable_result in self._evaluate_child_as_condition_(
            self.variable, sources
        ):
            if (
                variable_result.is_false
                or self.variable._id_ not in variable_result.bindings
            ):
                continue
            variable_result = variable_result.update(sources.bindings)
            if not self._condition_holds_for_(variable_result):
                continue
            # The witness is local to the quantifier; it does not appear in the results.
            yield self._build_operation_result_with_truth_(
                True, sources.bindings, variable_result
            )
            return

        # Negation as failure: no variable value satisfied the condition.
        yield self._build_operation_result_with_truth_(False, sources.bindings)

    def _condition_holds_for_(self, variable_result: OperationResult) -> bool:
        """
        :param variable_result: A binding for this quantifier's variable.
        :return: Whether the condition is true for any evaluation under *variable_result*.
        """
        return any(
            condition_result.is_true
            for condition_result in self._evaluate_child_as_condition_(
                self.condition, variable_result
            )
        )
