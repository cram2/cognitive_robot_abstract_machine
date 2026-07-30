"""
Arithmetic operators for the Entity Query Language.

These nodes let a query contain arithmetic such as ``revenue - expenses`` or ``amount * 12``. Unlike
:class:`~krrood.entity_query_language.operators.comparator.Comparator`, which yields a boolean, an
arithmetic node yields a *value* and is itself a
:class:`~krrood.entity_query_language.core.mapped_variable.CanBehaveLikeAVariable`, so its result can be
compared, aggregated, or used as an operand of further arithmetic.

Each node carries a :class:`~krrood.entity_query_language.operators.math_operations.MathOperator`, which
owns the Python callable that computes the operation over the resolved operand values.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Any, Iterable, Tuple

from krrood.entity_query_language.core.base_expressions import (
    BinaryExpression,
    OperationResult,
    Selectable,
    SymbolicExpression,
    UnaryExpression,
)
from krrood.entity_query_language.core.mapped_variable import CanBehaveLikeAVariable
from krrood.entity_query_language.operators.math_operations import MathOperator
from krrood.entity_query_language.operators.set_operations import (
    PerformsCartesianProduct,
)
from krrood.entity_query_language.utils import T


@dataclass(eq=False, repr=False)
class ArithmeticOperation(
    BinaryExpression, PerformsCartesianProduct, CanBehaveLikeAVariable[T]
):
    """
    A symbolic binary arithmetic operation (for example ``revenue - expenses``).
    """

    left: Selectable
    """
    The left operand.
    """

    right: Selectable
    """
    The right operand.
    """

    math_operator: MathOperator
    """
    The operator applied to the operands.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self._var_ = self

    @property
    def _product_operands_(self) -> Tuple[SymbolicExpression, ...]:
        return self.left, self.right

    @property
    def _name_(self) -> str:
        return f"({self.left._name_} {self.math_operator.symbol} {self.right._name_})"

    def _evaluate__(self, sources: OperationResult) -> Iterable[OperationResult]:
        """
        Apply the operator to every combination of the operands' values.
        """
        yield from (
            self._operation_result_(child_result)
            for child_result in self._evaluate_product_(sources)
        )

    def _operation_result_(self, child_result: OperationResult) -> OperationResult:
        """
        Compute the operation for one product result and bind its value.

        :param child_result: The result carrying the values of both operands.
        :return: The result with this operation's value bound under its own id.
        """
        operands = (
            self.left._process_result_(child_result),
            self.right._process_result_(child_result),
        )
        value = self.math_operator.function(*operands)
        return self._build_operation_result_and_update_truth_value_(
            child_result.bindings | {self._id_: value}, child_result
        )


@dataclass(eq=False, repr=False)
class UnaryArithmeticOperation(UnaryExpression, CanBehaveLikeAVariable[T]):
    """
    A symbolic unary arithmetic operation (for example negation, ``-amount``).
    """

    _child_: CanBehaveLikeAVariable[T]
    """
    The operand the operator is applied to.
    """

    math_operator: MathOperator
    """
    The operator applied to the operand.
    """

    def __post_init__(self) -> None:
        self._var_ = self
        super().__post_init__()

    @property
    def _name_(self) -> str:
        return f"({self.math_operator.symbol}{self._child_._name_})"

    def _evaluate__(self, sources: OperationResult) -> Iterable[OperationResult]:
        """
        Apply the operator to every value produced by the child expression.
        """
        yield from (
            self._build_operation_result_and_update_truth_value_(
                child_result.bindings
                | {self._id_: self._operation_value_(child_result)},
                child_result,
            )
            for child_result in self._child_._evaluate_(sources)
        )

    def _operation_value_(self, child_result: OperationResult) -> Any:
        """
        :param child_result: The result carrying the child's value.
        :return: The operator applied to the child's value.
        """
        value = self._child_._process_result_(child_result)
        return self.math_operator.function(value)
