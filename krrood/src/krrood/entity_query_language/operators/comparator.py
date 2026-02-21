"""
Comparators for the Entity Query Language.

This module provides binary comparison operators that evaluate relationships between symbolic expressions,
including equality, ordering, and membership-style checks.
"""

from __future__ import annotations

import operator
from copy import copy
from dataclasses import dataclass

from typing_extensions import (
    Callable,
    Any,
    ClassVar,
    Dict,
    Tuple,
    Iterable,
    TYPE_CHECKING,
)

from ..query.quantifiers import The
from ..core.base_expressions import (
    Bindings,
    OperationResult,
    SymbolicExpression,
    BinaryExpression,
    Selectable,
)
from .set_operations import PerformsCartesianProduct
from ..utils import is_iterable, make_set


@dataclass(eq=False, repr=False)
class Comparator(BinaryExpression, PerformsCartesianProduct):
    """
    A symbolic equality check that can be used to compare symbolic variables using a provided comparison operation.
    """

    left: Selectable
    right: Selectable
    operation: Callable[[Any, Any], bool]
    operation_name_map: ClassVar[Dict[Any, str]] = {
        operator.eq: "==",
        operator.ne: "!=",
        operator.lt: "<",
        operator.le: "<=",
        operator.gt: ">",
        operator.ge: ">=",
    }

    @property
    def _product_operands_(self) -> Tuple[SymbolicExpression, ...]:
        return self.left, self.right

    @property
    def _name_(self):
        if self.operation in self.operation_name_map:
            return self.operation_name_map[self.operation]
        return self.operation.__name__

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:
        """
        Compares the left and right symbolic variables using the "operation".
        """

        yield from (
            self.get_operation_result(result)
            for result in self._evaluate_product_(sources)
        )

    def get_operation_result(self, child_result: OperationResult) -> OperationResult:
        """
        Evaluate the comparator operation and return the result.

        :param child_result: The current result that has the values for the first and second operands.
        :return: The result of the operation.
        """
        left_value, right_value = (
            child_result[self.left._binding_id_],
            child_result[self.right._binding_id_],
        )
        if (
            self.operation in [operator.eq, operator.ne]
            and is_iterable(left_value)
            and is_iterable(right_value)
        ):
            left_value = make_set(left_value)
            right_value = make_set(right_value)
        res = self.operation(left_value, right_value)
        self._is_false_ = not res
        bindings = copy(child_result.bindings)
        bindings[self._binding_id_] = res
        return OperationResult(bindings, self._is_false_, self, child_result)

    def _optimize_operands_order_(
        self, sources: Bindings
    ) -> Tuple[SymbolicExpression, SymbolicExpression]:
        left_has_the = any(isinstance(desc, The) for desc in self.left._descendants_)
        right_has_the = any(isinstance(desc, The) for desc in self.right._descendants_)
        if left_has_the and not right_has_the:
            return self.left, self.right
        elif not left_has_the and right_has_the:
            return self.right, self.left
        if sources and any(
            v._binding_id_ in sources for v in self.right._unique_variables_
        ):
            return self.right, self.left
        else:
            return self.left, self.right


def not_contains(container, item) -> bool:
    """
    The inverted contains operation.

    :param container: The container.
    :param item: The item to test if contained in the container.
    :return:
    """
    return not operator.contains(container, item)
