"""
Set operations and cartesian-product execution for the Entity Query Language.

This module includes multi-arity union and abstract helpers to evaluate expressions via nested cartesian products.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import Iterable, Tuple, Iterator

from ..core.base_expressions import (
    MultiArityExpression,
    Bindings,
    OperationResult,
    SymbolicExpression,
)
from ..utils import cartesian_product_while_passing_the_bindings_around


@dataclass(eq=False, repr=False)
class Union(MultiArityExpression):
    """
    A symbolic union operation that can be used to evaluate multiple symbolic expressions in a sequence.
    """

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:
        yield from (
            self.get_result_and_update_truth_value(child_result)
            for child_result in itertools.chain(
                *(var._evaluate_(sources, self) for var in self._operation_children_)
            )
        )

    def get_result_and_update_truth_value(
        self, child_result: OperationResult
    ) -> OperationResult:
        self._is_false_ = child_result.is_false
        return OperationResult(
            child_result.bindings, self._is_false_, self, child_result
        )

    def add_child(self, child: SymbolicExpression) -> None:
        """
        Adds a child operand to the union operator.

        :param child: The child operand to add.
        """
        self._operation_children_ = self._operation_children_ + (child,)
        child._parent_ = self


@dataclass(eq=False, repr=False)
class PerformsCartesianProduct(SymbolicExpression, ABC):
    """
    A symbolic operation that evaluates its children in nested sequence, passing bindings from one to the next such that
    each binding has a value from each child expression. It represents a cartesian product of all child expressions.
    """

    @property
    @abstractmethod
    def _product_operands_(self) -> Tuple[SymbolicExpression, ...]:
        """
        :return: The operands of the cartesian product operation.
        """
        ...

    def _evaluate_product_(self, sources: Bindings) -> Iterator[OperationResult]:
        """
        Evaluate the symbolic expressions by generating combinations of values from their evaluation generators.

        :param sources: The current bindings.
        :return: An Iterable of Bindings for each combination of values.
        """
        ordered_operands = self._optimize_operands_order_(sources)
        return cartesian_product_while_passing_the_bindings_around(
            ordered_operands, sources, parent=self
        )

    def _optimize_operands_order_(
        self, sources: Bindings
    ) -> Tuple[SymbolicExpression, ...]:
        """
        Should be overridden by derived classes if they can optimize the operands order.
        """
        return self._product_operands_


@dataclass(eq=False, repr=False)
class MultiArityExpressionThatPerformsACartesianProduct(
    MultiArityExpression, PerformsCartesianProduct, ABC
):
    """
    An abstract superclass of expressions that have multiple operands and performs a cartesian product on them.
    """

    @property
    def _product_operands_(self) -> Tuple[SymbolicExpression, ...]:
        return self._operation_children_
