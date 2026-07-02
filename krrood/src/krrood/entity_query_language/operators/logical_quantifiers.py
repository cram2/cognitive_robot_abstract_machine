"""
Logical quantifiers for the Entity Query Language.

This module provides quantified conditionals such as universal (ForAll) and existential (Exists) operators
that evaluate conditions over the values of a variable.
"""

from __future__ import annotations

import uuid
from abc import ABC
from dataclasses import dataclass
from functools import cached_property
from typing import List, Iterable

from krrood.entity_query_language.core.base_expressions import OperationResult
from krrood.entity_query_language.operators.core_logical_operators import (
    LogicalBinaryOperator,
)


@dataclass(eq=False, repr=False)
class QuantifiedConditional(LogicalBinaryOperator, ABC):
    """
    This is the super class of the universal, and existential conditional operators. It is a binary logical operator
    that has a quantified variable and a condition on the values of that variable.
    """

    @property
    def variable(self):
        return self.left

    @property
    def condition(self):
        return self.right


@dataclass(eq=False, repr=False)
class ForAll(QuantifiedConditional):
    """
    This operator is the universal conditional operator. It returns bindings that satisfy the condition for all the
    values of the quantified variable. It is efficient as it ignores the bindings that don't satisfy the condition.
    """

    @cached_property
    def condition_unique_variable_ids(self) -> List[uuid.UUID]:
        return [
            v._id_
            for v in self.condition._unique_variables_.difference(
                self.left._unique_variables_
            )
        ]

    def _evaluate__(
        self,
        sources: OperationResult,
    ) -> Iterable[OperationResult]:
        solution_set = None

        for var_val in self.variable._evaluate_(sources):
            if solution_set is None:
                solution_set = self.get_all_candidate_solutions(var_val)
            else:
                solution_set = [
                    sol
                    for sol in solution_set
                    if self.evaluate_condition(
                        OperationResult({**sol, **var_val.bindings})
                    )
                ]
            if not solution_set:
                solution_set = []
                break

        # Yield the remaining bindings (non-universal) merged with the incoming sources
        yield from [
            OperationResult(sources.bindings | sol, False, self) for sol in solution_set
        ]

    def get_all_candidate_solutions(self, var_val: OperationResult):
        values_that_satisfy_condition = []
        # Evaluate the condition under this particular universal value
        for condition_val in self._evaluate_child_as_condition_(
            self.condition, var_val
        ):
            if condition_val.is_false:
                continue
            condition_val_bindings = {
                k: v
                for k, v in condition_val.bindings.items()
                if k in self.condition_unique_variable_ids
            }
            values_that_satisfy_condition.append(condition_val_bindings)
        return values_that_satisfy_condition

    def evaluate_condition(self, sources: OperationResult) -> bool:
        for condition_val in self._evaluate_child_as_condition_(
            self.condition, sources
        ):
            return condition_val.is_true
        return False

    def _invert_(self):
        return Exists(self.variable, self.condition._invert_())


@dataclass(eq=False, repr=False)
class Exists(QuantifiedConditional):
    """
    An existential checker that checks if a condition holds for any value of the variable given, the benefit
    of this is that it returns True if the condition holds for any value without
    getting all the condition values that hold for one specific value of the variable.
    """

    def _evaluate__(
        self,
        sources: OperationResult,
    ) -> Iterable[OperationResult]:
        for val in self._evaluate_child_as_condition_(self.variable, sources):
            if val.is_false or self.variable._id_ not in val.bindings:
                continue
            val = val.update(sources.bindings)
            for cond_val in self._evaluate_child_as_condition_(self.condition, val):
                if cond_val.is_true:
                    yield OperationResult(
                        sources.bindings
                        | {
                            id_: val.bindings[id_]
                            for id_ in self._ids_of_variables_to_add_to_sources_
                            if id_ in val.bindings
                        },
                        is_false=False,
                        operand=self,
                        previous_operation_result=val,
                    )
                    return

        # Negation as failure: no variable value satisfied the condition.
        yield OperationResult(sources.bindings, is_false=True, operand=self)

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
