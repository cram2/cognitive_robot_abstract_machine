"""
Query abstractions for the Entity Query Language.

This module implements query composition and evaluation, including selection, filtering, grouping, ordering,
distinct handling, and quantification over symbolic expressions.
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

from krrood.entity_query_language.core.mapped_variable import CanBehaveLikeAVariable
from krrood.entity_query_language.query.builders import (
    WhereBuilder,
    HavingBuilder,
    GroupedByBuilder,
    QuantifierBuilder,
    OrderedByBuilder,
)
from krrood.entity_query_language.query.operations import (
    Where,
    Having,
    GroupedBy,
    OrderedBy,
)
from krrood.entity_query_language.query.quantifiers import (
    ResultQuantificationConstraint,
    ResultQuantifier,
    An,
)
from krrood.entity_query_language.core.base_expressions import (
    Bindings,
    OperationResult,
    SymbolicExpression,
    UnaryExpression,
    Selectable,
    UnificationDict,
)
from krrood.entity_query_language.cache_data import (
    SeenSet,
)
from krrood.entity_query_language.core.variable import (
    InstantiatedVariable,
    Variable,
    ExternallySetVariable,
)
from krrood.entity_query_language.enums import DomainSource
from krrood.entity_query_language.exceptions import (
    UnsupportedNegation,
    TryingToModifyAnAlreadyBuiltQuery,
    NonPositiveLimitValue,
)
from krrood.entity_query_language.operators.aggregators import Aggregator, CountAll
from krrood.entity_query_language.operators.set_operations import (
    MultiArityExpressionThatPerformsACartesianProduct,
)
from krrood.entity_query_language.utils import (
    T,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.factories import ConditionType

ResultMapping = Callable[[Iterator[OperationResult]], Iterator[OperationResult]]
"""
A function that maps the results of a query to a new set of results.
"""


@dataclass(eq=False, repr=False)
class Query(
    MultiArityExpressionThatPerformsACartesianProduct, CanBehaveLikeAVariable[T], ABC
):
    """
    Describes the queried object(s), could be a query over a single variable or a set of variables,
    also describes the condition(s)/properties of the queried object(s).
    """

    _selected_variables_: Tuple[Selectable, ...] = field(
        default_factory=tuple, kw_only=True
    )
    """
    The variables that are selected by the query.
    """
    _distinct_on: Tuple[Selectable, ...] = field(default_factory=tuple, init=False)
    """
    Parameters for distinct results of the query.
    """
    _results_mapping: List[ResultMapping] = field(init=False, default_factory=list)
    """
    Mapping functions that map the results of the query to a new set of results.
    """
    _seen_results: Optional[SeenSet] = field(init=False, default=None)
    """
    A set of seen results, used when distinct is called in the query.
    """
    _where_builder_: Optional[WhereBuilder] = field(init=False, default=None)
    """
    The builder for the `Where` expression of the query.
    """
    _grouped_by_builder_: Optional[GroupedByBuilder] = field(init=False, default=None)
    """
    The builder for the `GroupedBy` expression of the query.
    """
    _having_builder_: Optional[HavingBuilder] = field(init=False, default=None)
    """
    The builder for the `Having` expression of the query.
    """
    _ordered_by_builder_: Optional[OrderedByBuilder] = field(default=None, init=False)
    """
    The builder for the `OrderedBy` expression if present.
    """
    _quantifier_builder_: Optional[QuantifierBuilder] = field(default=None, init=False)
    """
    The builder for the `ResultQuantifier` expression of the query. The default quantifier is `An`
     which yields all results.
    """
    _built_: bool = field(default=False, init=False)
    """
    Whether the query has built the query (wired the query operations) or not. If built already, it
    cannot be modified further and an error will be raised if a user tries to modify the query.
    """
    _update_ordered_by_: bool = field(default=True, init=False)
    """
    Whether the query has updated the ordered by expression or not. If updated already, it
    cannot be modified further and an error will be raised if a user tries to modify the query.
    """
    _update_quantifier_: bool = field(default=True, init=False)
    """
    Whether the query has updated the quantifier expression or not. If updated already, it
    cannot be modified further and an error will be raised if a user tries to modify the query.
    """

    def __post_init__(self):
        self._operation_children_ = tuple(self._selected_variables_)
        MultiArityExpressionThatPerformsACartesianProduct.__post_init__(self)

        self._var_ = self
        Selectable.__post_init__(self)

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

    def evaluate(self) -> Iterator:
        """
        Wrap the query in a ResultQuantifier expression and evaluate it,
         returning an iterator over the results.
        """
        self.build()
        if self._expression_ is not self:
            return self._expression_.evaluate()
        else:
            return MultiArityExpressionThatPerformsACartesianProduct.evaluate(self)

    @modifies_query_structure
    def where(self, *conditions: ConditionType) -> Self:
        """
        Set the conditions that describe the query object. The conditions are chained using AND.

        :param conditions: The conditions that describe the query object.
        :return: This query.
        """
        if self._where_builder_ is None:
            self._where_builder_ = WhereBuilder(conditions=conditions, query=self)
        else:
            self._where_builder_.conditions += conditions
        return self

    @modifies_query_structure
    def having(self, *conditions: ConditionType) -> Self:
        """
        Set the conditions that describe the query object. The conditions are chained using AND.

        :param conditions: The conditions that describe the query object.
        :return: This query.
        """
        if self._having_builder_ is None:
            self._having_builder_ = HavingBuilder(conditions=conditions, query=self)
        else:
            self._having_builder_.conditions += conditions
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
        self._update_ordered_by_ = True
        return self

    def distinct(
        self,
        *on: TypingUnion[Selectable, Any],
    ) -> TypingUnion[Self, T]:
        """
        Apply distinctness constraint to the query results.

        :param on: The variables to be used for distinctness.
        :return: This query.
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
        :return: This query.
        """
        self._grouped_by_builder_ = GroupedByBuilder(self, variables_to_group_by)
        return self

    def limit(self, n: int) -> Self:
        """
        Limit the number of results to n.

        :param n: The maximum number of results to return.
        :return: This query.
        """
        self._limit_ = n
        if not isinstance(self._limit_, int) or self._limit_ <= 0:
            raise NonPositiveLimitValue(self._limit_)
        if self._built_:
            self._expression_._limit_ = self._limit_
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
        :return: This query.
        """
        self._quantifier_builder_ = QuantifierBuilder(
            self, quantifier_type, quantification_constraint
        )
        self._update_quantifier_ = True
        return self

    def __enter__(self):
        """
        Make sure the query is built before entering the context manager for rule trees.
        """
        self.build()
        expression = self._conditions_root_
        SymbolicExpression._symbolic_expression_stack_.append(expression)
        return expression

    def build(self) -> Self:
        """
        Build the query by wiring the nodes together in the correct order of evaluation.

        :return: This query.
        """
        if self._built_:
            # TODO: This is a temporary fix, a coming PR will clean it up.
            self._update_ordered_by_expression_()
            self._update_quantifier_expression_()
            return self

        self._built_ = True

        if self._group_ and self._grouped_by_builder_ is None:
            self._grouped_by_builder_ = GroupedByBuilder(self)

        children = []
        if self._having_builder_ is not None:
            self._having_builder_.grouped_by = self._grouped_by_builder_.expression
            children.append(self._having_builder_.expression)
        elif self._grouped_by_builder_ is not None:
            children.append(self._grouped_by_builder_.expression)
        elif self._where_builder_ is not None:
            children.append(self._where_builder_.expression)

        self._if_count_all_is_used_update_its_child_to_be_the_grouped_by_expression_()

        children.extend(self._selected_variables_)

        self.update_children(*children)

        self._update_ordered_by_expression_()

        self._update_quantifier_expression_()

        return self

    def _if_count_all_is_used_update_its_child_to_be_the_grouped_by_expression_(
        self,
    ) -> None:
        """
        Update the child of the `CountAll` aggregator to be the `GroupedBy` expression if it exists.
        """
        if self._grouped_by_builder_ is None:
            return
        count_all = next(
            (
                aggregator
                for aggregator in self._grouped_by_builder_.aggregators_and_non_aggregators[
                    0
                ]
                if isinstance(aggregator, CountAll)
            ),
            None,
        )
        if count_all is None:
            return
        count_all._replace_child_(
            count_all._child_, self._grouped_by_builder_.expression
        )

    # TODO: This is a temporary fix, a coming PR will clean it up.
    def _update_ordered_by_expression_(self):
        if (self._ordered_by_builder_ is None) or not self._update_ordered_by_:
            return self
        og_child = self._expression_
        if isinstance(self._expression_, OrderedBy):
            og_child = self._expression_._child_
            self._remove_parent_(self._expression_)
        self._update_ordered_by_ = False
        self._ordered_by_builder_.data_source = og_child
        self._expression_ = self._ordered_by_builder_.expression
        return self

    # TODO: This is a temporary fix, a coming PR will clean it up.
    def _update_quantifier_expression_(self):
        if (self._quantifier_builder_ is None) or not self._update_quantifier_:
            return self
        og_child = self._expression_
        if isinstance(self._expression_, ResultQuantifier):
            og_child = self._expression_._child_
            self._remove_parent_(self._expression_)
        self._update_quantifier_ = False
        self._quantifier_builder_.child = og_child
        self._expression_ = self._quantifier_builder_.expression
        self._expression_._limit_ = self._limit_
        return self

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:
        """
        Evaluate the query by constraining values, updating conclusions,
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
            {v._id_: child_result[v._id_] for v in self._selected_variables_},
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
        return self._having_builder_.expression if self._having_builder_ else None

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
        return tuple(k._id_ for k in self._distinct_on)

    def _get_distinct_results_(
        self, results_gen: Iterator[OperationResult]
    ) -> Iterator[OperationResult]:
        """
        Apply distinctness constraint to the query results.

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
        return (len(self._aggregators_and_non_aggregators_in_selection_[0]) > 0) or (
            self._grouped_by_builder_ is not None
        )

    @cached_property
    def _aggregators_and_non_aggregators_in_selection_(
        self,
    ) -> Tuple[List[Aggregator], List[Selectable]]:
        """
        :return: The aggregated and non-aggregated variables from the selected variables.
        """
        aggregated_variables = []
        non_aggregated_variables = []
        for variable in self._selected_variables_:
            if isinstance(variable, Aggregator):
                aggregated_variables.append(variable)
            elif isinstance(variable, InstantiatedVariable):
                non_aggregated_variables.extend(variable._operation_children_)
            elif (
                isinstance(variable, ExternallySetVariable)
                and variable._domain_source_ == DomainSource.DEDUCTION
            ):
                continue
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
        Make sure to set the parent of the built expression of the query instead of the query itself.
        """
        # TODO: A hot fix for now, will be cleaned in a coming PR.
        if not isinstance(parent, (ResultQuantifier, OrderedBy)):
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

    def _get_operation_result_(self, child_result: OperationResult) -> OperationResult:
        """
        Update the result bindings with this operation's bindings.
        """
        operation_result = super()._get_operation_result_(child_result)
        operation_result.bindings = {
            self._id_: UnificationDict(
                {var: operation_result[var._id_] for var in self._selected_variables_}
            )
        }
        return operation_result


@dataclass(eq=False, repr=False)
class Entity(Query[T]):
    """
    A query over a single variable.
    """

    def _get_operation_result_(self, child_result: OperationResult) -> OperationResult:
        """
        Update the result bindings with this operation's bindings.
        """
        operation_result = super()._get_operation_result_(child_result)
        operation_result.bindings = {
            self._id_: operation_result[self.selected_variable._id_]
        }
        return operation_result

    @property
    def selected_variable(self):
        return self._selected_variables_[0] if self._selected_variables_ else None
