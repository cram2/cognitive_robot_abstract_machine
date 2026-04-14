"""
Query builders for the Entity Query Language.

This module defines builder classes that collect metadata and produce symbolic expressions for filtering,
 grouping, ordering, and quantifying query results.
"""

from __future__ import annotations

import itertools
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property, lru_cache

from ordered_set import OrderedSet
from typing_extensions import Tuple, List, Type, Optional, Callable, TYPE_CHECKING

from krrood.entity_query_language.core.base_expressions import (
    SymbolicExpression,
    Selectable,
)
from krrood.entity_query_language.operators.core_logical_operators import (
    chained_logic,
    AND,
)
from krrood.entity_query_language.exceptions import (
    NoConditionsProvided,
    LiteralConditionError,
    AggregatorInWhereConditionsError,
    NonAggregatorInHavingConditionsError,
    NonAggregatedSelectedVariablesError,
)
from krrood.entity_query_language.query.quantifiers import (
    ResultQuantificationConstraint,
    ResultQuantifier,
    An,
)
from krrood.entity_query_language.query.operations import (
    Where,
    Having,
    OrderedBy,
    GroupedBy,
)
from krrood.entity_query_language.operators.aggregators import Aggregator, CountAll
from krrood.entity_query_language.core.variable import (
    Literal,
)
from krrood.entity_query_language.core.mapped_variable import MappedVariable
from krrood.utils import memoize

if TYPE_CHECKING:
    from krrood.entity_query_language.factories import ConditionType
    from krrood.entity_query_language.query.query import Query


@dataclass
class ExpressionBuilder(ABC):
    """
    Base class for builder classes of symbolic expressions. This class collects meta-data about expressions to finally
    build the expression.
    """

    query: Query
    """
    The query that the expression is being built for.
    """
    _built_expression: Optional[SymbolicExpression] = field(init=False, default=None)
    """
    The expression that is built from the metadata.
    """

    @property
    def expression(self) -> SymbolicExpression:
        """
        :return: The expression that is built from the metadata.
        """
        if self._built_expression is not None:
            return self._built_expression
        self._built_expression = self.build()
        return self._built_expression

    def build(self) -> SymbolicExpression:
        """
        :return: The expression that is built from the metadata.
        """

    def __hash__(self) -> int:
        return hash((self.__class__, self.query))


@dataclass(eq=False)
class FilterBuilder(ExpressionBuilder, ABC):
    """
    Metadata for Filter subclasses.
    """

    conditions: Tuple[ConditionType, ...]
    """
    The conditions that must be satisfied.
    """

    def __post_init__(self):
        self.assert_correct_conditions()

    def assert_correct_conditions(self):
        """
        :raises NoConditionsProvided: If no conditions are provided.
        :raises LiteralConditionError: If any of the conditions is a literal expression.
        """
        # If there are no conditions raise error.
        if len(self.conditions) == 0:
            raise NoConditionsProvided(self.query)

        # If there's a constant condition raise error.
        literal_expressions = [
            exp for exp in self.conditions if not isinstance(exp, SymbolicExpression)
        ]
        if literal_expressions:
            raise LiteralConditionError(self.query, literal_expressions)

    @cached_property
    def aggregators_and_non_aggregators_in_conditions(
        self,
    ) -> Tuple[Tuple[Aggregator, ...], Tuple[Selectable, ...]]:
        """
        :return: A tuple containing the aggregators and non-aggregators in the conditions.
        """
        from krrood.entity_query_language.query.query import Query

        aggregators, non_aggregators = [], []

        def walk(expr: SymbolicExpression):

            if isinstance(expr, Aggregator):
                aggregators.append(expr)
                # No need to traverse inside aggregators
                return
            elif isinstance(expr, Selectable) and not isinstance(
                expr, (Literal, ResultQuantifier, Query)
            ):
                non_aggregators.append(expr)

            if isinstance(expr, (Literal, ResultQuantifier, Query)):
                # Subqueries are a boundary, we don't need to traverse inside them.
                return

            for child in expr._children_:
                walk(child)

        for condition in self.conditions:
            walk(condition)

        return tuple(aggregators), tuple(non_aggregators)

    @cached_property
    def conditions_expression(self) -> SymbolicExpression:
        """
        :return: The expression representing the conditions of the Filter.
        """
        return chained_logic(AND, *self.conditions)


@dataclass(eq=False)
class WhereBuilder(FilterBuilder):
    """
    Metadata for the `Where` Filter.
    """

    def assert_correct_conditions(self):
        """
        Assert that the where conditions are correct.

        :raises AggregatorInWhereConditionsError: If the where conditions contain any aggregators.
        """
        super().assert_correct_conditions()
        aggregators, non_aggregators = (
            self.aggregators_and_non_aggregators_in_conditions
        )
        if aggregators:
            raise AggregatorInWhereConditionsError(aggregators, query=self.query)

    def build(self) -> Where:
        return Where(self.conditions_expression)


@dataclass(eq=False)
class HavingBuilder(FilterBuilder):
    """
    Metadata for the `Having` Filter.
    """

    grouped_by: GroupedBy = field(kw_only=True, default=None)
    """
    The GroupedBy expression associated with the having Filter, as the having conditions are applied on
     the aggregations of grouped results.
    """

    def build(self) -> Having:
        aggregators, non_aggregators = (
            self.aggregators_and_non_aggregators_in_conditions
        )
        if any(
            var._id_ not in self.grouped_by.ids_of_variables_to_group_by
            for var in non_aggregators
        ):
            raise NonAggregatorInHavingConditionsError(
                non_aggregators, query=self.query
            )
        return Having(self.grouped_by, self.conditions_expression)


@dataclass(eq=False)
class GroupedByBuilder(ExpressionBuilder):
    """
    Metadata for the GroupedBy operation.
    """

    variables_to_group_by: Tuple[Selectable, ...] = ()
    """
    The variables to group the results by their values.
    """

    def __post_init__(self):
        self.assert_correct_selected_variables()

    def build(self) -> GroupedBy:
        aggregators, non_aggregators = self.aggregators_and_non_aggregators
        where = self.query._where_expression_
        children = OrderedSet()
        if where:
            children.add(where)
        children.update(non_aggregators)
        children.update(self.variables_to_group_by)
        return GroupedBy(
            _operation_children_=tuple(children),
            aggregators=tuple(aggregators),
            variables_to_group_by=tuple(self.variables_to_group_by),
        )

    @memoize
    def assert_correct_selected_variables(self):
        """
        Assert that the selected variables are correct.

        :raises UsageError: If the selected variables are not valid.
        """
        aggregators, non_aggregated_variables = (
            self.query._aggregators_and_non_aggregators_in_selection_
        )
        if aggregators and not all(
            self.variable_is_in_or_derived_from_a_grouped_by_variable(v)
            for v in non_aggregated_variables
        ):
            raise NonAggregatedSelectedVariablesError(
                self,
                non_aggregated_variables,
                aggregators,
                query=self.query,
            )

    @memoize
    def variable_is_in_or_derived_from_a_grouped_by_variable(
        self, variable: SymbolicExpression
    ) -> bool:
        """
        Check if the variable is in or derived from a grouped by variable.

        :param variable: The variable to check.
        """
        if variable._id_ in self.ids_of_variables_to_group_by:
            return True
        elif variable._id_ in self.ids_of_aggregated_variables:
            return False
        elif isinstance(variable, MappedVariable) and any(
            self.variable_is_in_or_derived_from_a_grouped_by_variable(d)
            for d in variable._descendants_
        ):
            return True
        else:
            return False

    @cached_property
    def ids_of_aggregated_variables(self) -> Tuple[uuid.UUID, ...]:
        """
        :return: A tuple of ids of aggregated variables.
        """
        return tuple(
            v._child_._id_
            for v in self.aggregators_in_selected_variables
            if v._child_ is not None
        )

    @cached_property
    def ids_of_variables_to_group_by(self) -> Tuple[uuid.UUID, ...]:
        """
        :return: A tuple of the binding IDs of the variables to group by.
        """
        return tuple(var._id_ for var in self.variables_to_group_by)

    @cached_property
    def aggregators_and_non_aggregators(
        self,
    ) -> Tuple[List[Aggregator], List[Selectable]]:
        """
        :return: A tuple of lists of aggregator and non-aggregator variables used in the query.
        """
        all_aggregators = OrderedSet()
        all_non_aggregators = OrderedSet()

        aggregators_in_selection, non_aggregators_in_selection = (
            self.query._aggregators_and_non_aggregators_in_selection_
        )

        # Extend aggregators
        all_aggregators.update(aggregators_in_selection)

        # Extend non-aggregators
        all_non_aggregators.update(non_aggregators_in_selection)

        if self.query._having_builder_:
            having_aggregators, having_non_aggregators = (
                self.query._having_builder_.aggregators_and_non_aggregators_in_conditions
            )
            all_aggregators.update(having_aggregators)
            all_non_aggregators.update(having_non_aggregators)

        if self.query._ordered_by_builder_:
            ordered_by_variable = self.query._ordered_by_builder_.variable
            if isinstance(ordered_by_variable, Aggregator):
                all_aggregators.add(ordered_by_variable)
            else:
                all_non_aggregators.add(ordered_by_variable)

        all_non_aggregators.update(
            [
                aggregator._child_
                for aggregator in all_aggregators
                if not isinstance(aggregator, CountAll)
            ]
        )

        return list(all_aggregators), list(all_non_aggregators)

    @cached_property
    def aggregators_and_non_aggregators_in_ordered_by(
        self,
    ) -> Tuple[List[Aggregator], List[Selectable]]:
        if not self.query._ordered_by_builder_:
            return [], []

        variable = self.query._ordered_by_builder_.variable

        if isinstance(variable, Aggregator):
            return [variable], (
                [variable._child_] if variable._child_ is not None else []
            )

        return [], [variable]

    @cached_property
    def aggregators_in_selected_variables(self) -> Tuple[Aggregator, ...]:
        """
        :return: A tuple of aggregators in the selected variables of the query.
        """
        return tuple(
            var
            for var in self.query._selected_variables_
            if isinstance(var, Aggregator)
        )


@dataclass(eq=False)
class QuantifierBuilder(ExpressionBuilder):
    """
    Builds a result quantifier (An/The) of the specified type with the given child and quantification constraint.
    """

    type: Type[ResultQuantifier] = An
    """
    The type of the quantifier to be built.
    """
    quantification_constraint: Optional[ResultQuantificationConstraint] = None
    """
    The quantification constraint that must be satisfied by the result quantifier if present.
    """
    child: Optional[Selectable] = None
    """
    The child expression of the quantifier.
    """

    def build(self) -> ResultQuantifier:
        """
        Builds a result quantifier of the specified type with the given child and quantification constraint.
        """
        if self.type is An:
            return self.type(
                self.child,
                _quantification_constraint_=self.quantification_constraint,
            )
        else:
            return self.type(self.child)


@dataclass(eq=False)
class OrderedByBuilder(ExpressionBuilder):
    variable: Selectable
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
    data_source: Optional[SymbolicExpression] = None
    """
    The data source that generates the results to be ordered.
    """

    def build(self) -> OrderedBy:
        return OrderedBy(self.data_source, self.variable, self.descending, self.key)
