"""
Core symbolic expression system used to build and evaluate entity queries.

This module defines the symbolic types (variables, sources, logical and
comparison operators) and the evaluation mechanics.
"""

from __future__ import annotations

from abc import ABC
from copy import copy
from dataclasses import dataclass, field
from functools import cached_property, wraps

from typing_extensions import (
    Iterable,
    Any,
    Optional,
    Type,
    Dict,
    Union as TypingUnion,
    TYPE_CHECKING,
    List,
    Tuple,
    Callable,
    Self,
    Iterator,
)

from ..core.domain_mapping import CanBehaveLikeAVariable
from .query_builders import (
    WhereBuilder,
    HavingBuilder,
    GroupedByBuilder,
    QuantifierBuilder,
    OrderedByBuilder,
)
from .query_descriptor_operations import Where, Having, GroupedBy
from .result_quantifiers import (
    ResultQuantificationConstraint,
    ResultQuantifier,
    An,
)
from ..core.base_expressions import (
    Bindings,
    OperationResult,
    SymbolicExpression,
    UnaryExpression,
    Selectable,
)
from ..cache_data import (
    SeenSet,
)
from ..failures import (
    UnsupportedNegation,
    TryingToModifyAnAlreadyBuiltQuery,
)
from ..operators.aggregators import Aggregator
from ..operators.set_operations import MultiArityExpressionThatPerformsACartesianProduct
from ..utils import (
    T,
)

if TYPE_CHECKING:
    from ..factories import ConditionType

ResultMapping = Callable[[Iterator[OperationResult]], Iterator[OperationResult]]
"""
A function that maps the results of a query object descriptor to a new set of results.
"""


@dataclass(eq=False, repr=False)
class Query(MultiArityExpressionThatPerformsACartesianProduct, ABC):
    """
    Describes the queried object(s), could be a query over a single variable or a set of variables,
    also describes the condition(s)/properties of the queried object(s).
    """

    _selected_variables_: Tuple[Selectable, ...] = field(
        default_factory=tuple, kw_only=True
    )
    """
    The variables that are selected by the query object descriptor.
    """
    _distinct_on: Tuple[Selectable, ...] = field(default_factory=tuple, init=False)
    """
    Parameters for distinct results of the query object descriptor.
    """
    _results_mapping: List[ResultMapping] = field(init=False, default_factory=list)
    """
    Mapping functions that map the results of the query object descriptor to a new set of results.
    """
    _seen_results: Optional[SeenSet] = field(init=False, default=None)
    """
    A set of seen results, used when distinct is called in the query object descriptor.
    """
    _where_builder_: Optional[WhereBuilder] = field(init=False, default=None)
    """
    The builder for the `Where` expression of the query object descriptor.
    """
    _grouped_by_builder_: Optional[GroupedByBuilder] = field(init=False, default=None)
    """
    The builder for the `GroupedBy` expression of the query object descriptor.
    """
    _having_builder: Optional[HavingBuilder] = field(init=False, default=None)
    """
    The builder for the `Having` expression of the query object descriptor.
    """
    _ordered_by_builder_: Optional[OrderedByBuilder] = field(default=None, init=False)
    """
    The builder for the `OrderedBy` expression if present.
    """
    _quantifier_builder_: Optional[QuantifierBuilder] = field(default=None, init=False)
    """
    The builder for the `ResultQuantifier` expression of the query object descriptor. The default quantifier is `An`
     which yields all results.
    """
    _built_: bool = field(default=False, init=False)
    """
    Whether the query object descriptor has built the query (wired the query operations) or not. If built already, it
    cannot be modified further and an error will be raised if a user tries to modify the query object descriptor.
    """

    def __post_init__(self):
        for var in self._selected_variables_:
            if isinstance(var, Query):
                var.build()
        self._operation_children_ = tuple(self._selected_variables_)
        super().__post_init__()
        self._quantifier_builder_ = QuantifierBuilder(self)

    @staticmethod
    def modifies_query_structure(method):
        """
        A decorator to mark methods that modify the structure of the query. If the query is already
        built, an error will be raised when trying to call any of these methods.
        """

        @wraps(method)
        def wrapper(self, *args, **kwargs):
            if self._built_:
                raise TryingToModifyAnAlreadyBuiltQuery(self)
            return method(self, *args, **kwargs)

        return wrapper

    def evaluate(self, limit: Optional[int] = None) -> Iterator:
        """
        Wrap the query object descriptor in a ResultQuantifier expression and evaluate it,
         returning an iterator over the results.
        """
        self.build()
        return self._expression_.evaluate(limit=limit)

    @modifies_query_structure
    def where(self, *conditions: ConditionType) -> Self:
        """
        Set the conditions that describe the query object. The conditions are chained using AND.

        :param conditions: The conditions that describe the query object.
        :return: This query object descriptor.
        """
        if self._where_builder_ is None:
            self._where_builder_ = WhereBuilder(
                conditions=conditions, query_descriptor=self
            )
        else:
            self._where_builder_.conditions += conditions
        return self

    @modifies_query_structure
    def having(self, *conditions: ConditionType) -> Self:
        """
        Set the conditions that describe the query object. The conditions are chained using AND.

        :param conditions: The conditions that describe the query object.
        :return: This query object descriptor.
        """
        if self._having_builder is None:
            self._having_builder = HavingBuilder(
                conditions=conditions, query_descriptor=self
            )
        else:
            self._having_builder.conditions += conditions
        return self

    def ordered_by(
        self,
        variable: TypingUnion[Selectable[T], Any],
        descending: bool = False,
        key: Optional[Callable] = None,
    ) -> Self:
        """
        Order the results by the given variable, using the given key function in descending or ascending order.

        :param variable: The variable to order by.
        :param descending: Whether to order the results in descending order.
        :param key: A function to extract the key from the variable value.
        """
        self._ordered_by_builder_ = OrderedByBuilder(
            self, variable, descending=descending, key=key
        )
        return self

    def distinct(
        self,
        *on: TypingUnion[Selectable, Any],
    ) -> TypingUnion[Self, T]:
        """
        Apply distinctness constraint to the query object descriptor results.

        :param on: The variables to be used for distinctness.
        :return: This query object descriptor.
        """
        self._distinct_on = on if on else self._selected_variables_
        self._seen_results = SeenSet(keys=self._distinct_on_ids_)
        self._results_mapping.append(self._get_distinct_results_)
        return self

    @modifies_query_structure
    def grouped_by(
        self, *variables_to_group_by: TypingUnion[Selectable, Any]
    ) -> TypingUnion[Self, T]:
        """
        Specify the variables to group the results by.

        :param variables_to_group_by: The variables to group the results by.
        :return: This query object descriptor.
        """
        self._grouped_by_builder_ = GroupedByBuilder(self, variables_to_group_by)
        return self

    def _quantify_(
        self,
        quantifier_type: Type[ResultQuantifier] = An,
        quantification_constraint: Optional[ResultQuantificationConstraint] = None,
    ) -> Self:
        """
        Specify the quantifier type and constraint for the query results, also build the query.

        :param quantifier_type: The type of the quantifier to be used.
        :param quantification_constraint: The constraint to apply to the quantifier.
        :return: This query object descriptor.
        """
        self._quantifier_builder_ = QuantifierBuilder(
            self, quantifier_type, quantification_constraint
        )
        return self

    def __enter__(self):
        """
        Make sure the query is built before entering the context manager for rule trees.
        """
        self.build()
        super().__enter__()

    def build(self) -> Self:
        """
        Build the query object descriptor by wiring the nodes together in the correct order of evaluation.

        :return: This query object descriptor.
        """
        if self._built_:
            return self

        self._built_ = True

        if self._group_ and self._grouped_by_builder_ is None:
            self._grouped_by_builder_ = GroupedByBuilder(self)

        children = []
        if self._having_builder is not None:
            self._having_builder.grouped_by = self._grouped_by_builder_.expression
            children.append(self._having_builder.expression)
        elif self._grouped_by_builder_ is not None:
            children.append(self._grouped_by_builder_.expression)
        elif self._where_builder_ is not None:
            children.append(self._where_builder_.expression)

        children.extend(self._selected_variables_)

        self.update_children(*children)

        if self._ordered_by_builder_ is not None:
            self._ordered_by_builder_.data_source = self._expression_
            self._expression_ = self._ordered_by_builder_.expression

        self._quantifier_builder_.child = self._expression_
        self._expression_ = self._quantifier_builder_.expression

        return self

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:
        """
        Evaluate the query descriptor by constraining values, updating conclusions,
        and selecting variables.
        """

        yield from (
            self._get_operation_result_(result)
            for result in self._apply_results_mapping_(
                self._evaluate_product_(sources),
            )
        )

        if self._seen_results is not None:
            self._seen_results.clear()

    def _get_operation_result_(self, child_result: OperationResult) -> OperationResult:
        """
        :param child_result: The child result to construct the operation result from.
        :return: The operation result.
        """
        return OperationResult(
            {
                v._binding_id_: child_result[v._binding_id_]
                for v in self._selected_variables_
            },
            self._is_false_,
            self,
            child_result,
        )

    @property
    def _where_expression_(self) -> Optional[Where]:
        """
        The built `Where` expression.
        """
        return self._where_builder_.expression if self._where_builder_ else None

    @property
    def _having_expression_(self) -> Optional[Having]:
        """
        The built `Having` expression.
        """
        return self._having_builder.expression if self._having_builder else None

    @property
    def _grouped_by_expression_(self) -> Optional[GroupedBy]:
        """
        The built `GroupedDataSource` expression.
        """
        return (
            self._grouped_by_builder_.expression if self._grouped_by_builder_ else None
        )

    @property
    def _quantifier_expression_(self) -> Optional[ResultQuantifier]:
        return (
            self._quantifier_builder_.expression if self._quantifier_builder_ else None
        )

    @cached_property
    def _distinct_on_ids_(self) -> Tuple[int, ...]:
        """
        Get the IDs of variables used for distinctness.
        """
        return tuple(k._binding_id_ for k in self._distinct_on)

    def _get_distinct_results_(
        self, results_gen: Iterator[OperationResult]
    ) -> Iterator[OperationResult]:
        """
        Apply distinctness constraint to the query object descriptor results.

        :param results_gen: Generator of results.
        :return: Generator of distinct results.
        """
        for result in results_gen:
            bindings = copy(result.bindings)
            self._update_res_with_distinct_on_variables_(bindings)
            if self._seen_results.check(bindings):
                continue
            self._seen_results.add(bindings)
            yield result

    def _update_res_with_distinct_on_variables_(self, res: Dict[int, Any]):
        """
        Update the result dictionary with values from distinct-on variables if not already present.

        :param res: The result dictionary to update.
        """
        for i, id_ in enumerate(self._distinct_on_ids_):
            if id_ in res:
                continue
            var_value = self._distinct_on[i]._evaluate_(copy(res), parent=self)
            res[id_] = next(var_value).value

    @cached_property
    def _group_(self) -> bool:
        """
        :return: Whether the results should be grouped or not. Is true when an aggregator is selected.
        """
        return (
            len(self._aggregated_and_non_aggregated_variables_in_selection_[0]) > 0
        ) or (self._grouped_by_builder_ is not None)

    @cached_property
    def _aggregated_and_non_aggregated_variables_in_selection_(
        self,
    ) -> Tuple[List[Selectable], List[Selectable]]:
        """
        :return: The aggregated and non-aggregated variables from the selected variables.
        """
        aggregated_variables = []
        non_aggregated_variables = []
        for variable in self._selected_variables_:
            if isinstance(variable, Aggregator):
                aggregated_variables.append(variable)
            else:
                non_aggregated_variables.append(variable)
        return aggregated_variables, non_aggregated_variables

    def _apply_results_mapping_(
        self, results: Iterator[OperationResult]
    ) -> Iterable[OperationResult]:
        """
        Process and transform an iterable of results based on predefined mappings and ordering.

        This method applies a sequence of result transformations defined in the instance,
        using a series of mappings to modify the results.

        :param results: An iterable containing dictionaries that represent the initial result set to be transformed.
        :return: An iterable containing dictionaries that represent the transformed data.
        """
        for result_mapping in self._results_mapping:
            results = result_mapping(results)
        return results

    @UnaryExpression._parent_.setter
    def _parent_(self, parent: SymbolicExpression):
        """
        Make sure to set the parent of the built expression of the query instead of the query object descriptor itself.
        """
        self.build()
        if self._expression_ is not self:
            self._expression_._parent_ = parent
        else:
            UnaryExpression._parent_.__set__(self, parent)

    def _invert_(self):
        raise UnsupportedNegation(self.__class__)

    @property
    def _name_(self) -> str:
        return f"({', '.join(var._name_ for var in self._selected_variables_)})"


@dataclass(eq=False, repr=False)
class SetOf(Query):
    """
    A query over a set of variables.
    """

    def __getitem__(
        self, selected_variable: TypingUnion[CanBehaveLikeAVariable[T], T]
    ) -> TypingUnion[T, SetOfSelectable[T]]:
        """
        Select one of the set of variables, this is useful when you have another query that uses this set of and
        wants to select a specific variable out of the set of variables.

        :param selected_variable: The selected variable from the set of variables.
        """
        return SetOfSelectable(self, selected_variable)


@dataclass(eq=False, repr=False)
class SetOfSelectable(UnaryExpression, CanBehaveLikeAVariable[T]):
    """
    A selected variable from the SetOf operation selected variables.
    """

    _child_: SetOf
    """
    The SetOf operation from which `_selected_var_` was selected.
    """
    _selected_var_: CanBehaveLikeAVariable[T]
    """
    The selected variable in the SetOf.
    """

    def __post_init__(self):
        super().__post_init__()
        self._var_ = self

    @property
    def _set_of_(self) -> SetOf:
        """
        The SetOf operation from which `_selected_var_` was selected.
        """
        return self._child_

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterator[OperationResult]:
        for v in self._set_of_._evaluate_(sources, self):
            yield OperationResult(
                {**v.bindings, self._binding_id_: v[self._selected_var_._binding_id_]},
                False,
                self,
            )

    @property
    def _name_(self) -> str:
        return f"{self._set_of_._name_}[{self._selected_var_._name_}]"


@dataclass(eq=False, repr=False)
class Entity(Query, CanBehaveLikeAVariable[T]):
    """
    A query over a single variable.
    """

    def __post_init__(self):
        self._var_ = self
        super().__post_init__()

    def _get_operation_result_(self, child_result: OperationResult) -> OperationResult:
        """
        Update the result bindings with this operation's bindings.
        """
        operation_result = super()._get_operation_result_(child_result)
        operation_result.bindings = {
            self._binding_id_: operation_result[self.selected_variable._binding_id_]
        }
        return operation_result

    @property
    def selected_variable(self):
        return self._selected_variables_[0] if self._selected_variables_ else None
