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

from ..core.base_expressions import Bindings, OperationResult
from ..operators.core_logical_operators import LogicalBinaryOperator


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
            v._binding_id_
            for v in self.condition._unique_variables_.difference(
                self.left._unique_variables_
            )
        ]

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:
        solution_set = None

        for var_val in self.variable._evaluate_(sources, parent=self):
            if solution_set is None:
                solution_set = self.get_all_candidate_solutions(var_val.bindings)
            else:
                solution_set = [
                    sol
                    for sol in solution_set
                    if self.evaluate_condition({**sol, **var_val.bindings})
                ]
            if not solution_set:
                solution_set = []
                break

        # Yield the remaining bindings (non-universal) merged with the incoming sources
        yield from [
            OperationResult({**sources, **sol}, False, self) for sol in solution_set
        ]

    def get_all_candidate_solutions(self, sources: Bindings):
        values_that_satisfy_condition = []
        # Evaluate the condition under this particular universal value
        for condition_val in self.condition._evaluate_(sources, parent=self):
            if condition_val.is_false:
                continue
            condition_val_bindings = {
                k: v
                for k, v in condition_val.bindings.items()
                if k in self.condition_unique_variable_ids
            }
            values_that_satisfy_condition.append(condition_val_bindings)
        return values_that_satisfy_condition

    def evaluate_condition(self, sources: Bindings) -> bool:
        for condition_val in self.condition._evaluate_(sources, parent=self):
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
        sources: Bindings,
    ) -> Iterable[OperationResult]:
        seen_var_values = []
        for val in self.condition._evaluate_(sources, parent=self):
            var_val = val[self.variable._binding_id_]
            if val.is_true and var_val not in seen_var_values:
                seen_var_values.append(var_val)
                yield OperationResult(val.bindings, False, self)

    def _invert_(self):
        return ForAll(self.variable, self.condition._invert_())
