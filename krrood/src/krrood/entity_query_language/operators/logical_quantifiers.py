"""
Logical quantifiers for the Entity Query Language.

This module provides quantified conditionals such as universal (ForAll) and existential
(Exists) operators that evaluate conditions over the values of a variable.
"""

from __future__ import annotations

import uuid
from abc import ABC
from dataclasses import dataclass
from functools import cached_property
from typing_extensions import Iterable

from krrood.entity_query_language.core.base_expressions import OperationResult
from krrood.entity_query_language.core.mapped_variable import MappedVariable
from krrood.entity_query_language.operators.core_logical_operators import (
    LogicalBinaryOperator,
)


@dataclass(kw_only=True, eq=False, repr=False)
class QuantifiedConditional(LogicalBinaryOperator, ABC):
    """
    This is the super class of the universal, and existential conditional operators.

    It is a binary logical operator that has a quantified variable and a condition on
    the values of that variable.
    """

    @property
    def variable(self):
        return self.left

    @property
    def condition(self):
        return self.right


@dataclass(kw_only=True, eq=False, repr=False)
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

    def _evaluate__(
        self,
        *,
        sources: OperationResult,
    ) -> Iterable[OperationResult]:
        solution_set = None

        for variable_result in self.variable._evaluate_(sources=sources):
            if solution_set is None:
                solution_set = self.get_all_candidate_solutions(
                    variable_result=variable_result
                )
            else:
                solution_set = [
                    solution
                    for solution in solution_set
                    if self.evaluate_condition(
                        sources=OperationResult(
                            bindings={**solution, **variable_result.bindings}
                        )
                    )
                ]
            if not solution_set:
                solution_set = []
                break

        # Yield the remaining bindings (non-universal) merged with the incoming sources
        yield from [
            self._build_operation_result_with_truth_(
                truth=True, bindings=sources.bindings | solution
            )
            for solution in solution_set
        ]

    def get_all_candidate_solutions(self, *, variable_result: OperationResult):
        values_that_satisfy_condition = []
        # Evaluate the condition under this particular universal value
        for condition_result in self._evaluate_child_as_condition_(
            child=self.condition, sources=variable_result
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

    def evaluate_condition(self, *, sources: OperationResult) -> bool:
        for condition_result in self._evaluate_child_as_condition_(
            child=self.condition, sources=sources
        ):
            return condition_result.is_true
        return False

    def _invert_(self):
        return Exists(left=self.variable, right=self.condition._invert_())


@dataclass(kw_only=True, eq=False, repr=False)
class Exists(QuantifiedConditional):
    """
    An existential checker that checks if a condition holds for any value of the
    variable given, the benefit of this is that it returns True if the condition holds
    for any value without getting all the condition values that hold for one specific
    value of the variable.
    """

    def _evaluate__(
        self,
        *,
        sources: OperationResult,
    ) -> Iterable[OperationResult]:
        for outer_result in self._bind_variables_the_values_are_taken_from_(sources=sources):
            yield self._quantify_over_values_in_(sources=outer_result)

    def _bind_variables_the_values_are_taken_from_(
        self, *, sources: OperationResult
    ) -> Iterable[OperationResult]:
        """
        :param sources: The bindings this quantifier is evaluated under.
        :return: One result per binding of the expression the quantified variable maps
            from, such as each lease of ``flat_variable(lease.units)``, each extending
            *sources*.
        """
        if not isinstance(self.variable, MappedVariable):
            yield sources
            return
        for outer_result in self.variable._child_._evaluate_(sources=sources):
            yield outer_result.update(other=sources.bindings)

    def _quantify_over_values_in_(self, *, sources: OperationResult) -> OperationResult:
        """
        :param sources: Bindings that fix the outer variables of this quantifier.
        :return: A true result for the first value of the quantified variable that
            satisfies the condition, or a false result with *sources* if none does.
        """
        for variable_result in self._evaluate_child_as_condition_(
            child=self.variable, sources=sources
        ):
            if (
                variable_result.is_false
                or self.variable._id_ not in variable_result.bindings
            ):
                continue
            variable_result = variable_result.update(other=sources.bindings)
            if not self._condition_holds_for_(variable_result=variable_result):
                continue
            return self._build_operation_result_with_truth_(
                truth=True,
                bindings=sources.bindings
                | {
                    id_: variable_result.bindings[id_]
                    for id_ in self._ids_of_variables_to_add_to_sources_
                    if id_ in variable_result.bindings
                },
                child_result=variable_result,
            )

        # Negation as failure: no variable value satisfied the condition.
        return self._build_operation_result_with_truth_(
            truth=False, bindings=sources.bindings
        )

    def _condition_holds_for_(self, *, variable_result: OperationResult) -> bool:
        """
        :param variable_result: A binding for this quantifier's variable.
        :return: Whether the condition is true for any evaluation under *variable_result*.
        """
        return any(
            condition_result.is_true
            for condition_result in self._evaluate_child_as_condition_(
                child=self.condition, sources=variable_result
            )
        )

    @cached_property
    def _ids_of_variables_to_add_to_sources_(self):
        """
        :return: The ids of the variables that are selected in the root query except the variable of this quantifier.
        """
        if self._root_query_ is None:
            return []
        return [
            v._id_
            for v in self._root_query_._selected_variables_
            if v._id_ != self.variable._id_
        ]
