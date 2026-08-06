"""
Set operations and cartesian-product execution for the Entity Query Language.

This module includes multi-arity union and abstract helpers to evaluate expressions via
nested cartesian products.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import Iterable, Optional, Tuple, Iterator

from krrood.entity_query_language.core.base_expressions import (
    MultiArityExpression,
    TruthValuedExpression,
    Bindings,
    OperationResult,
    SymbolicExpression,
)
from krrood.entity_query_language.utils import (
    cartesian_product_while_passing_the_bindings_around,
)


@dataclass(eq=False, repr=False)
class EvaluatesChildrenInSequence(MultiArityExpression, ABC):
    """
    An expression that yields the results of each of its children in turn.

    A result it yields is its own, and a result's truth is read from the binding of the
    expression that produced it, so each child result's truth is recorded under this
    expression's identifier before the result is passed on. A subclass that selects a
    value overwrites that binding with the value.
    """

    def _evaluate__(
        self,
        sources: OperationResult,
    ) -> Iterable[OperationResult]:
        yield from (
            self._build_operation_result_with_truth_(
                child_result.is_true, child_result.bindings, child_result
            )
            for child_result in itertools.chain(
                *(var._evaluate_(sources) for var in self._operation_children_)
            )
        )

    def add_child(self, child: SymbolicExpression) -> None:
        """
        Adds a child operand to this expression.

        :param child: The child operand to add.
        """
        self._operation_children_ = self._operation_children_ + (child,)
        child._parent_ = self


@dataclass(eq=False, repr=False)
class Union(TruthValuedExpression, EvaluatesChildrenInSequence):
    """
    A symbolic union operation that can be used to evaluate multiple symbolic
    expressions in a sequence.

    Keeps the truth its base records, so its binding is always the truth of the child
    result it yields and never a value a caller selects.
    """


@dataclass(eq=False, repr=False)
class PerformsCartesianProduct(SymbolicExpression, ABC):
    """
    A symbolic operation that evaluates its children in nested sequence, passing
    bindings from one to the next such that each binding has a value from each child
    expression.

    It represents a cartesian product of all child expressions.
    """

    @property
    @abstractmethod
    def _product_operands_(self) -> Tuple[SymbolicExpression, ...]:
        """
        :return: The operands of the cartesian product operation.
        """
        ...

    def _evaluate_product_(
        self, sources: Optional[OperationResult]
    ) -> Iterator[OperationResult]:
        """
        Evaluate the symbolic expressions by generating combinations of values from
        their evaluation generators.

        :param sources: The current OperationResult carrying bindings, or None.
        :return: An Iterable of Bindings for each combination of values.
        """
        ordered_operands = self._optimize_operands_order_(sources)
        return cartesian_product_while_passing_the_bindings_around(
            ordered_operands, sources
        )

    def _optimize_operands_order_(
        self, sources: Optional[OperationResult]
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
    An abstract superclass of expressions that have multiple operands and performs a
    cartesian product on them.
    """

    @property
    def _product_operands_(self) -> Tuple[SymbolicExpression, ...]:
        return self._operation_children_
