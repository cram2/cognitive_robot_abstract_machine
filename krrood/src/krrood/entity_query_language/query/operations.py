"""
Query operations for the Entity Query Language.

This module implements core query operations such as filtering (Where/Having), grouping, and ordering over
symbolic expressions and their aggregated results.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property, lru_cache

from typing_extensions import Tuple, Any, Iterator, Iterable, Optional, Callable, Dict

from ..operators.aggregators import Aggregator, Count
from ..core.base_expressions import (
    DerivedExpression,
    SymbolicExpression,
    UnaryExpression,
    Bindings,
    OperationResult,
    BinaryExpression,
    Filter,
    Selectable,
)
from ..failures import UnsupportedAggregationOfAGroupedByVariable
from ..operators.set_operations import MultiArityExpressionThatPerformsACartesianProduct
from ..utils import ensure_hashable, is_iterable
from ..core.mapped_variable import MappedVariable

GroupKey = Tuple[Any, ...]
"""
A tuple representing values of variables that are used in the grouped_by clause.
"""


@dataclass(eq=False, repr=False)
class Where(Filter, UnaryExpression):
    """
    A symbolic expression that represents the `where()` statement of `QueryObjectDescriptor`. It is used to filter
    ungrouped data. Is constructed through the `Where()` method of the `QueryObjectDescriptor`.
    """

    @property
    def condition(self) -> SymbolicExpression:
        return self._child_

    def _evaluate__(self, sources: Bindings) -> Iterator[OperationResult]:
        yield from (
            result
            for result in self._child_._evaluate_(sources, parent=self)
            if result.is_true
        )


@dataclass(eq=False, repr=False)
class Having(Filter, BinaryExpression):
    """
    A symbolic having expression that can be used to filter the grouped results of a query.
    Is constructed through the `QueryObjectDescriptor` using the `having()` method.
    """

    left: GroupedBy
    """
    The grouped by expression that is used to group the results of the query. This is a child of the Having expression.
    As the results need to be grouped before filtering.
    """
    right: SymbolicExpression
    """
    The condition expression that is used to filter the grouped results of the query.
    """

    @property
    def condition(self) -> SymbolicExpression:
        return self.right

    @property
    def grouped_by(self) -> GroupedBy:
        return self.left

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:
        yield from (
            OperationResult(
                grouping_result.bindings | annotated_result.bindings,
                self._is_false_,
                self,
            )
            for grouping_result in self.grouped_by._evaluate_(sources, parent=self)
            for annotated_result in self.condition._evaluate_(
                grouping_result.bindings, parent=self
            )
            if annotated_result.is_true
        )


@dataclass(eq=False, repr=False)
class OrderedBy(BinaryExpression, DerivedExpression):
    """
    Represents an ordered by clause in a query. This orders the results of query according to the values of the
    specified variable.
    """

    right: Selectable
    """
    The variable to order by.
    """
    descending: bool = False
    """
    Whether to order the results in descending order.
    """
    key: Optional[Callable] = None
    """
    A function to extract the key from the variable value.
    """

    @property
    def _original_expression_(self) -> SymbolicExpression:
        """
        The original expression that this expression was derived from.
        """
        return self.left

    @property
    def variable(self) -> Selectable:
        """
        The variable to order by.
        """
        return self.right

    def _evaluate__(self, sources: Bindings) -> Iterator[OperationResult]:
        results = list(self.left._evaluate_(sources, parent=self))
        yield from sorted(
            results,
            key=self.apply_key,
            reverse=self.descending,
        )

    def apply_key(self, result: OperationResult) -> Any:
        """
        Apply the key function to the variable to extract the reference value to order the results by.
        """
        var = self.variable
        var_id = var._binding_id_
        if var_id not in result.all_bindings:
            variable_value = next(var._evaluate_(result.all_bindings, self)).value
        else:
            variable_value = result.all_bindings[var_id]
        if self.key:
            return self.key(variable_value)
        else:
            return variable_value

    @property
    def _name_(self) -> str:
        return f"OrderedBy({self.variable._name_})"


GroupBindings = Dict[GroupKey, OperationResult]
"""
A dictionary for grouped bindings which maps a group key to its corresponding bindings.
"""


@dataclass(eq=False, repr=False)
class GroupedBy(MultiArityExpressionThatPerformsACartesianProduct):
    """
    This operation groups the results of a query by specific variables. This is useful for aggregating results
    separately for each group.
    """

    aggregators: Tuple[Aggregator, ...] = field(default_factory=tuple)
    """
    The aggregators to apply to the grouped results.
    """
    variables_to_group_by: Tuple[Selectable, ...] = ()
    """
    The variables to group the results by their values.
    """

    def _evaluate__(self, sources: Bindings = None) -> Iterator[OperationResult]:
        """
        Generate results grouped by the specified variables in the grouped_by clause.

        :param sources: The current bindings.
        :return: An iterator of OperationResult objects, each representing a group of child results.
        """

        if any(
            not isinstance(var, Count)
            for var in self.aggregators_of_grouped_by_variables
        ):
            raise UnsupportedAggregationOfAGroupedByVariable(self)

        groups, group_key_count = self.get_groups_and_group_key_count(sources)

        for agg in self.aggregators_of_grouped_by_variables:
            for group_key, group in groups.items():
                group[agg._binding_id_] = group_key_count[group_key]

        yield from groups.values()

    def get_groups_and_group_key_count(
        self, sources: Bindings
    ) -> Tuple[GroupBindings, Dict[GroupKey, int]]:
        """
        Create a dictionary of groups and a dictionary of group keys to their corresponding counts starting from the
        initial bindings, then applying the constraints in the where expression then grouping by the variables in the
        grouped_by clause.

        :param sources: The initial bindings.
        :return: A tuple containing the dictionary of groups and the dictionary of group keys to their corresponding counts.
        """

        groups = defaultdict(lambda: OperationResult({}, False, self))
        group_key_count = defaultdict(lambda: 0)

        for res in self._evaluate_product_(sources):

            group_key = tuple(
                ensure_hashable(res[var._binding_id_])
                for var in self.variables_to_group_by
            )

            if self.count_occurrences_of_each_group_key:
                group_key_count[group_key] += 1

            self.update_group_from_bindings(groups[group_key], res.bindings)

        if len(groups) == 0:
            # if there are no groups, add one empty group with an empty list for each aggregated variable.
            for var in [
                var._child_ for var in self.aggregators if var._child_ is not None
            ]:
                groups[()][var._binding_id_] = []

        return groups, group_key_count

    def update_group_from_bindings(self, group: OperationResult, results: Bindings):
        """
        Updates the group with the given results.

        :param group: The group to be updated.
        :param results: The results to be added to the group.
        """
        for id_, val in results.items():
            if id_ in self.ids_of_variables_to_group_by:
                group[id_] = val
            elif self.is_already_grouped(id_):
                group[id_] = val if is_iterable(val) else [val]
            else:
                if id_ not in group:
                    group[id_] = []
                group[id_].append(val)

    @lru_cache
    def is_already_grouped(self, var_id: uuid.UUID) -> bool:
        expression = self._get_expression_by_id_(var_id)
        return (
            len(self.variables_to_group_by) == 1
            and isinstance(expression, MappedVariable)
            and expression._child_._binding_id_ in self.ids_of_variables_to_group_by
        )

    @cached_property
    def count_occurrences_of_each_group_key(self) -> bool:
        """
        :return: True if there are any aggregators of type Count in the selected variables of the query descriptor that
         are counting values of variables that are in the grouped_by clause, False otherwise.
        """
        return len(self.aggregators_of_grouped_by_variables) > 0

    @cached_property
    def aggregators_of_grouped_by_variables(self):
        """
        :return: A list of the aggregators in the selected variables of the query descriptor that are aggregating over
         expressions having variables that are in the grouped_by clause.
        """
        return [
            var
            for var in self.aggregators
            if (var._child_ is None)
            or (var._child_._binding_id_ in self.ids_of_variables_to_group_by)
        ]

    @cached_property
    def ids_of_variables_to_group_by(self) -> Tuple[uuid.UUID, ...]:
        """
        :return: A tuple of the binding IDs of the variables to group by.
        """
        return tuple(var._binding_id_ for var in self.variables_to_group_by)

    @property
    def _name_(self) -> str:
        return f"{self.__class__.__name__}({', '.join([var._name_ for var in self.variables_to_group_by])})"
