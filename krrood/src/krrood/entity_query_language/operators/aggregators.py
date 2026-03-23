"""
This module defines aggregator functions for the Entity Query Language.

It contains classes for counting, summing, averaging, and finding extreme values in query results.
"""

from __future__ import annotations

import numbers
import statistics
import uuid
from abc import ABC, abstractmethod
from collections import Counter
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

from krrood.entity_query_language.core.base_expressions import (
    UnaryExpression,
    Bindings,
    OperationResult,
    Selectable,
)
from krrood.entity_query_language.exceptions import (
    NestedAggregationError,
    InvalidChildType,
)
from krrood.entity_query_language.operators.set_operations import Union
from krrood.entity_query_language.utils import T
from krrood.entity_query_language.core.mapped_variable import CanBehaveLikeAVariable

if TYPE_CHECKING:
    from krrood.entity_query_language.query.query import Entity
    from krrood.entity_query_language.query.operations import GroupedBy


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
        from krrood.entity_query_language.query.query import Entity

        return Entity(_selected_variables_=(self,)).evaluate()

    def grouped_by(self, *variables: Selectable) -> Entity:
        """
        Group the results by the given variables.
        """
        from krrood.entity_query_language.query.query import Entity

        return Entity(_selected_variables_=(self,)).grouped_by(*variables)

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterator[OperationResult]:
        yield from (
            OperationResult(
                sources | aggregation_result,
                False,
                self,
            )
            for child_result in self._child_._evaluate_(sources, parent=self)
            for aggregation_result in self._apply_aggregation_function_and_get_bindings_(
                child_result
            )
        )

    @abstractmethod
    def _apply_aggregation_function_and_get_bindings_(
        self, child_result: OperationResult
    ) -> Iterator[Bindings]:
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

    def _apply_aggregation_function_and_get_bindings_(
        self, child_result: OperationResult
    ) -> Iterator[Bindings]:
        if self._distinct_:
            yield {self._id_: len(set(child_result.value))}
        else:
            yield {self._id_: len(child_result.value)}


@dataclass(eq=False, repr=False)
class CountAll(Count[T]):
    """
    Count all results per group.
    """

    _child_: Optional[GroupedBy] = field(init=False, default=None)
    """
    The child expression to be counted which is the GroupedBy Operation, this will count of all results per group.
    It is set later during the query build process.
    """


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
    _default_value_: Optional[T] = field(kw_only=True, default=None)
    """
    The default value to be returned if the child results are empty.
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

    def get_aggregation_result_from_child_result(
        self, result: OperationResult
    ) -> Iterator:
        """
        :param result: The current operation result from the child.
        :return: The aggregated result or the default value if the child result is empty.
        """
        if not result.has_value or len(result.value) == 0:
            yield self._default_value_
            return
        results = result.value
        if self._distinct_:
            results = set(results)
        yield from self.aggregation_function(results)

    @abstractmethod
    def aggregation_function(self, result: Collection) -> Iterator:
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
    ) -> Iterator[Dict[uuid.UUID, Optional[IntOrFloat]]]:
        for aggregation_result in self.get_aggregation_result_from_child_result(
            child_result
        ):
            yield {self._id_: aggregation_result}

    def aggregation_function(
        self, result: Collection[IntOrFloat]
    ) -> Iterator[IntOrFloat]:
        yield sum(result)


@dataclass(eq=False, repr=False)
class Average(Sum):
    """
    Calculate the average of the child results.
    """

    def aggregation_function(
        self, result: Collection[IntOrFloat]
    ) -> Iterator[IntOrFloat]:
        for sum_value in super().aggregation_function(result):
            yield sum_value / len(result)


@dataclass(eq=False, repr=False)
class Extreme(EntityAggregator[T], ABC):
    """
    Find and return the extreme value among the child results. If given, make use of the key function to extract
    the value to be compared.
    """

    def _apply_aggregation_function_and_get_bindings_(
        self, child_result: OperationResult
    ) -> Iterator[Bindings]:
        for extreme_val in self.get_aggregation_result_from_child_result(child_result):
            bindings = child_result.bindings.copy()
            bindings[self._id_] = extreme_val
            yield bindings


@dataclass(eq=False, repr=False)
class Max(Extreme[T]):
    """
    Find and return the maximum value among the child results. If given, make use of the key function to extract
     the value to be compared.
    """

    def aggregation_function(self, values: Iterable) -> Iterator[T]:
        yield max(values, key=self._key_function_)


@dataclass(eq=False, repr=False)
class Min(Extreme[T]):
    """
    Find and return the minimum value among the child results. If given, make use of the key function to extract
     the value to be compared.
    """

    def aggregation_function(self, values: Iterable) -> Iterator[T]:
        yield min(values, key=self._key_function_)


@dataclass(eq=False, repr=False)
class MultiMode(Extreme[T]):
    """
    Find and return all the equivalent mode values among the child results. Similar to `statistics.multimode`, see
     its documentation for more details: https://docs.python.org/3/library/statistics.html#statistics.multimode.
    """

    def aggregation_function(self, values: Iterable) -> Iterator[T]:
        counter = Counter(values)
        max_count = max(counter.values())
        yield from (k for k, v in counter.items() if v == max_count)


@dataclass(eq=False, repr=False)
class Mode(Extreme[T]):
    """
    Find and return the mode value among the child results. Same as {py:class}`MultiMode`, but only returns the
    first mode value found.
    """

    def aggregation_function(self, values: Iterable) -> Iterator[T]:
        yield statistics.mode(values)
