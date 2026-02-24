"""
This module defines aggregator functions for the Entity Query Language.

It contains classes for counting, summing, averaging, and finding extreme values in query results.
"""

from __future__ import annotations

import numbers
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import (
    Optional,
    Iterator,
    Iterable,
    Callable,
    Any,
    Collection,
    Dict,
    TYPE_CHECKING,
)

from ..core.base_expressions import (
    UnaryExpression,
    Bindings,
    OperationResult,
    SymbolicExpression,
    Selectable,
)
from ..failures import NestedAggregationError, InvalidChildType
from ..utils import T
from ..core.variable import Variable
from ..core.mapped_variable import CanBehaveLikeAVariable


if TYPE_CHECKING:
    from ..query.query import Entity


IntOrFloat = int | float
"""
A type representing a number, which can be either an integer or a float.
"""


@dataclass(eq=False, repr=False)
class Aggregator(UnaryExpression, CanBehaveLikeAVariable[T], ABC):
    """
    Base class for aggregators. Aggregators are unary selectable expressions that take a single expression
     as a child.
    They aggregate the results of the child expression and evaluate to either a single value or a set of aggregated
     values for each group when `grouped_by()` is used.
    """

    _default_value_: Optional[T] = field(kw_only=True, default=None)
    """
    The default value to be returned if the child results are empty.
    """
    _distinct_: bool = field(kw_only=True, default=False)
    """
    Whether to consider only distinct values from the child results when applying the aggregation function.
    """

    def __post_init__(self):
        if isinstance(self._child_, Aggregator):
            raise NestedAggregationError(self)
        super().__post_init__()
        self._var_ = self

    def evaluate(self) -> Iterator[T]:
        """
        Wrap the aggregator in an entity and evaluate it (i.e., make a query with this aggregator as the selected
        expression and evaluate it.).

        :return: An iterator over the aggregator results.
        """
        from ..query.query import Entity

        return Entity(_selected_variables_=(self,)).evaluate()

    def grouped_by(self, *variables: Variable) -> Entity[T]:
        """
        Group the results by the given variables.
        """
        from ..query.query import Entity

        return Entity(_selected_variables_=(self,)).grouped_by(*variables)

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:
        yield from (
            OperationResult(
                sources
                | self._apply_aggregation_function_and_get_bindings_(child_result),
                False,
                self,
            )
            for child_result in self._child_._evaluate_(sources, parent=self)
        )

    @abstractmethod
    def _apply_aggregation_function_and_get_bindings_(
        self, child_result: OperationResult
    ) -> Bindings:
        """
        Apply the aggregation function to the results of the child.

        :param child_result: The result of the child.
        :return: Bindings containing the aggregated result.
        """
        ...


@dataclass(eq=False, repr=False)
class Count(Aggregator[T]):
    """
    Count the number of child results.
    """

    _child_: Optional[SymbolicExpression] = None
    """
    The child expression to be counted. If not given, the count of all results (by group if `grouped_by()` is specified)
     is returned.
    """

    def _apply_aggregation_function_and_get_bindings_(
        self, child_result: OperationResult
    ) -> Bindings:
        if self._distinct_:
            return {self._binding_id_: len(set(child_result.value))}
        else:
            return {self._binding_id_: len(child_result.value)}


@dataclass(eq=False, repr=False)
class EntityAggregator(Aggregator[T], ABC):
    """
    Entity aggregators are aggregators where the child (the entity to be aggregated) is a selectable expression. Also,
     If given, make use of the key function to extract the value to be aggregated from the child result.
    """

    _child_: Selectable[T]
    """
    The child entity to be aggregated.
    """
    _key_function_: Optional[Callable[[Any], Any]] = field(kw_only=True, default=None)
    """
    An optional function that extracts the value to be used in the aggregation.
    """

    def __post_init__(self):
        if not isinstance(self._child_, Selectable):
            raise InvalidChildType(type(self._child_), [Selectable])
        self._var_ = self
        super().__post_init__()

    def get_aggregation_result_from_child_result(self, result: OperationResult) -> Any:
        """
        :param result: The current operation result from the child.
        :return: The aggregated result or the default value if the child result is empty.
        """
        if not result.has_value or len(result.value) == 0:
            return self._default_value_
        results = result.value
        if self._distinct_:
            results = set(results)
        return self.aggregation_function(results)

    @abstractmethod
    def aggregation_function(self, result: Collection) -> Any:
        """
        :param result: The child result to be aggregated.
        :return: The aggregated result.
        """
        ...


@dataclass(eq=False, repr=False)
class Sum(EntityAggregator[numbers.Number]):
    """
    Calculate the sum of the child results.
    """

    def _apply_aggregation_function_and_get_bindings_(
        self, child_result: OperationResult
    ) -> Dict[uuid.UUID, Optional[IntOrFloat]]:
        return {
            self._binding_id_: self.get_aggregation_result_from_child_result(
                child_result
            )
        }

    def aggregation_function(self, result: Collection[IntOrFloat]) -> IntOrFloat:
        return sum(result)


@dataclass(eq=False, repr=False)
class Average(Sum):
    """
    Calculate the average of the child results.
    """

    def aggregation_function(self, result: Collection[IntOrFloat]) -> IntOrFloat:
        sum_value = super().aggregation_function(result)
        return sum_value / len(result)


@dataclass(eq=False, repr=False)
class Extreme(EntityAggregator[T], ABC):
    """
    Find and return the extreme value among the child results. If given, make use of the key function to extract
    the value to be compared.
    """

    def _apply_aggregation_function_and_get_bindings_(
        self, child_result: OperationResult
    ) -> Bindings:
        extreme_val = self.get_aggregation_result_from_child_result(child_result)
        bindings = child_result.bindings.copy()
        bindings[self._binding_id_] = extreme_val
        return bindings


@dataclass(eq=False, repr=False)
class Max(Extreme[T]):
    """
    Find and return the maximum value among the child results. If given, make use of the key function to extract
     the value to be compared.
    """

    def aggregation_function(self, values: Iterable) -> Any:
        return max(values, key=self._key_function_)


@dataclass(eq=False, repr=False)
class Min(Extreme[T]):
    """
    Find and return the minimum value among the child results. If given, make use of the key function to extract
     the value to be compared.
    """

    def aggregation_function(self, values: Iterable) -> Any:
        return min(values, key=self._key_function_)
