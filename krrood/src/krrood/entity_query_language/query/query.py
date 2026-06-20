"""
Query abstractions for the Entity Query Language.

This module implements query composition and evaluation, including selection, filtering, grouping, ordering,
distinct handling, and quantification over symbolic expressions.
"""

from __future__ import annotations

from abc import ABC
from copy import copy
from dataclasses import dataclass, field
from functools import cached_property

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

from krrood.entity_query_language.core.mapped_variable import (
    CanBehaveLikeAVariable,
    MappedVariable,
)
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
    NonPositiveLimitValue,
)
from krrood.entity_query_language.operators.aggregators import Aggregator, CountAll
from krrood.entity_query_language.operators.set_operations import (
    MultiArityExpressionThatPerformsACartesianProduct,
)
from krrood.entity_query_language._monitoring import (
    monitored,
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


@monitored
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
    _dirty_: bool = field(default=True, init=False)
    """
    Whether anything needs (re)building. Any modifier method marks the query dirty; :meth:`build`
    is a no-op while clean and otherwise applies only the parts flagged below. The query is never
    frozen, so it can always be modified, even after being built or embedded as a child.
    """
    _inner_dirty_: bool = field(default=True, init=False)
    """
    Whether the data-source chain (Where / GroupedBy / Having + selected variables) needs rewiring,
    set by :meth:`where`, :meth:`having` and :meth:`grouped_by`.
    """
    _building_: bool = field(default=False, init=False)
    """
    Re-entrancy guard set while :meth:`build` wires the wrapper layers, so that parenting the query
    to its own wrappers does not recursively trigger another build.
    """
    _compiled_inner_head_: Optional[SymbolicExpression] = field(
        default=None, init=False
    )
    """
    The data-source head (Where / GroupedBy / Having) wired by the last build, tracked so it can be
    detached cleanly when the chain is rewired.
    """
    _embedding_snapshot_: Optional[SymbolicExpression] = field(default=None, init=False)
    """
    The immutable snapshot embedded into parent expressions, cached so that multiple embeddings of
    this query (for example as a selected variable and within a condition) share one identity. It is
    discarded whenever the query is modified, so a later embedding reflects the change while
    already-embedded snapshots stay frozen.
    """

    def __post_init__(self):
        self._operation_children_ = tuple(self._selected_variables_)
        MultiArityExpressionThatPerformsACartesianProduct.__post_init__(self)

        self._var_ = self
        Selectable.__post_init__(self)

        self._quantifier_builder_ = QuantifierBuilder(self)

    def _mark_dirty_(self) -> None:
        """
        Flag the compiled expression as stale and drop caches that depend on modifier state, so the
        next :meth:`build` recomputes them. Called by every modifier method.
        """
        self._dirty_ = True
        self._embedding_snapshot_ = None
        self.__dict__.pop("_group_", None)
        self.__dict__.pop("_distinct_on_ids_", None)

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
        self._inner_dirty_ = True
        self._mark_dirty_()
        return self

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
        self._inner_dirty_ = True
        self._mark_dirty_()
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
        self._mark_dirty_()
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
        self._mark_dirty_()
        self._seen_results = SeenSet(keys=self._distinct_on_ids_)
        self._results_mapping.append(self._get_distinct_results_)
        return self

    def grouped_by(
        self, *variables_to_group_by: TypingUnion[Selectable, Any]
    ) -> TypingUnion[Self, T]:
        """
        Specify the variables to group the results by.

        :param variables_to_group_by: The variables to group the results by.
        :return: This query.
        """
        self._grouped_by_builder_ = GroupedByBuilder(self, variables_to_group_by)
        self._inner_dirty_ = True
        self._mark_dirty_()
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
        # The quantifier wrapper keeps its identity across rebuilds, so update it in place when it
        # already exists; otherwise the first build picks the limit up.
        if isinstance(self._expression_, ResultQuantifier):
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
        self._mark_dirty_()
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
        Build (or rebuild) the query by wiring the nodes together in the correct order of
        evaluation, and cache the compiled expression.

        The query is never frozen: any modifier marks it dirty and flags exactly which part changed
        (data-source chain, ordering, quantification), so it can always be modified — even after it
        has been built or embedded as a child of another expression. Building only re-does the
        flagged parts; the ordering and quantification wrappers are applied *around* the current
        compiled expression and keep their identity across rebuilds, so an expression derived from
        the query (e.g. ``query.ordered_by(query.name)``) stays consistent with what it ordered.

        :return: This query.
        """
        if not self._dirty_:
            return self

        self._building_ = True

        if self._inner_dirty_:
            self._rewire_data_source_chain_()
            self._inner_dirty_ = False

        self._apply_wrapping_modifiers_()

        self._dirty_ = False
        self._building_ = False
        return self

    def _rewire_data_source_chain_(self) -> None:
        """
        (Re)wire the data-source chain (Where / GroupedBy / Having) and the selected variables as the
        children of this query node. Safe to call repeatedly: the previous chain head is detached and
        modifier caches discarded so a fresh head reflecting the current conditions is built. The
        ordering/quantification wrappers keep pointing at this query node, so they pick up the change.
        """
        # Detach the head built last time and discard cached metadata on the chain modifiers, which
        # may have been mutated in place (e.g. extra ``where``/``having`` conditions appended).
        if self._compiled_inner_head_ is not None:
            self._compiled_inner_head_._parent_ = None
            self._compiled_inner_head_ = None
        for modifier in (
            self._where_builder_,
            self._grouped_by_builder_,
            self._having_builder_,
        ):
            if modifier is not None:
                modifier.reset()

        children = self._data_source_chain_head_()
        self._compiled_inner_head_ = children[0] if children else None

        self._if_count_all_is_used_update_its_child_to_be_the_grouped_by_expression_()

        children.extend(self._selected_variables_)
        self.update_children(*children)

    def _data_source_chain_head_(self) -> List[SymbolicExpression]:
        """
        Build the head of the data-source chain that feeds the selected variables. At most one of
        Having / GroupedBy / Where heads the chain, with precedence ``Having > GroupedBy > Where``
        since each already incorporates the previous one.

        :return: A list containing the chain head, or empty if the query is unfiltered/ungrouped.
        """
        if self._group_ and self._grouped_by_builder_ is None:
            self._grouped_by_builder_ = GroupedByBuilder(self)

        if self._having_builder_ is not None:
            self._having_builder_.grouped_by = self._grouped_by_builder_.expression
            return [self._having_builder_.expression]
        if self._grouped_by_builder_ is not None:
            return [self._grouped_by_builder_.expression]
        if self._where_builder_ is not None:
            return [self._where_builder_.expression]
        return []

    def _apply_wrapping_modifiers_(self) -> None:
        """
        Apply the ordering and quantification wrappers, innermost first, around the current compiled
        expression. Each modifier is applied only when it needs (re)applying; an already-applied
        wrapper keeps its identity across rebuilds, so a reference captured by an expression derived
        from the query stays valid. ``OrderedBy`` nests inside the ``ResultQuantifier``.
        """
        for modifier in (self._ordered_by_builder_, self._quantifier_builder_):
            if modifier is None or not modifier._needs_apply_:
                continue
            self._expression_ = modifier.rewrap(self._expression_)
            modifier._needs_apply_ = False

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

    def _evaluate__(
        self,
        sources: OperationResult,
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
            child_result.is_false,
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
        """
        :return: The compiled result quantifier at the root of the built expression, if any.
        """
        self.build()
        return (
            self._expression_
            if isinstance(self._expression_, ResultQuantifier)
            else None
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
            var_value = self._distinct_on[i]._evaluate_(OperationResult(copy(res)))
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

        def _update_aggregated_and_non_aggregated_variables(
            variable_: SymbolicExpression,
        ):
            if isinstance(variable_, Aggregator):
                aggregated_variables.append(variable_)
            elif isinstance(variable_, InstantiatedVariable):
                for child in variable_._operation_children_:
                    _update_aggregated_and_non_aggregated_variables(child)
            elif (
                isinstance(variable_, ExternallySetVariable)
                and variable_._domain_source_ == DomainSource.DEDUCTION
            ):
                pass
            else:
                non_aggregated_variables.append(variable_)

        for variable in self._selected_variables_:
            _update_aggregated_and_non_aggregated_variables(variable)
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

    def _as_embeddable_child_(self, parent: SymbolicExpression) -> SymbolicExpression:
        """
        Embed an immutable snapshot of this query when it becomes a value operand of another
        expression, so a later edit to this query cannot mutate the already-embedded copy.

        The live compiled node is used instead in two cases: while this query wires its own wrapper
        layers during :meth:`build` (``_building_``), where the wrappers must wrap the query itself;
        and when the parent is a derived reference (a :class:`MappedVariable` such as ``query.name``),
        which must track the live query so its values resolve against the query's own bindings.

        :param parent: The expression about to take this query as a child.
        :return: A freshly compiled snapshot of the query, or the live node for self-wrapping and
            derived references.
        """
        if self._building_:
            return self._expression_
        self.build()
        if isinstance(parent, MappedVariable):
            return self._expression_
        if not self._is_snapshotable_():
            return self._expression_
        if self._embedding_snapshot_ is None:
            self._embedding_snapshot_ = self._compile_snapshot_()
        return self._embedding_snapshot_

    def _is_snapshotable_(self) -> bool:
        """
        A snapshot replays this query's spec onto a fresh query that shares the original's selected
        variables. That is unsound only for :class:`CountAll
        <krrood.entity_query_language.operators.aggregators.CountAll>`, whose child is rewired to the
        grouping at build time: building the snapshot would mutate the shared aggregator and corrupt
        the original. Such queries share the live expression instead.

        .. note:: This narrow fallback is removed once the compiled node no longer mutates shared
            aggregators.

        :return: Whether the compiled expression can be safely snapshotted.
        """
        return not any(
            isinstance(node, CountAll) for node in self._expression_._descendants_
        )

    def _compile_snapshot_(self) -> SymbolicExpression:
        """
        Compile an independent snapshot of this query by replaying its modifiers onto a fresh query
        built from the same spec, rather than cloning the live compiled graph.

        The snapshot is a separate object graph sharing this query's identifier and selected
        variables, so it evaluates identically and co-references the same variables, while later
        edits that rebuild the original cannot reach it. Building from the spec sidesteps the
        cross-reference hazards of graph cloning, so grouping, distinct, and self-referential
        ordering all compile correctly.

        :return: The compiled expression of the freshly built snapshot query.
        """
        snapshot = type(self)(_selected_variables_=self._selected_variables_)
        snapshot._id_ = self._id_
        snapshot._expression_id_cache_ = {}
        if self._where_builder_ is not None:
            snapshot.where(*self._where_builder_.conditions)
        if self._grouped_by_builder_ is not None:
            snapshot.grouped_by(*self._grouped_by_builder_.variables_to_group_by)
        if self._having_builder_ is not None:
            snapshot.having(*self._having_builder_.conditions)
        if self._ordered_by_builder_ is not None:
            ordering = self._ordered_by_builder_
            snapshot.ordered_by(
                ordering.variable, descending=ordering.descending, key=ordering.key
            )
        if self._distinct_on:
            snapshot.distinct(*self._distinct_on)
        if self._limit_ is not None:
            snapshot.limit(self._limit_)
        quantifier = self._quantifier_builder_
        snapshot._quantify_(quantifier.type, quantifier.quantification_constraint)
        snapshot.build()
        return snapshot._expression_

    @UnaryExpression._parent_.setter
    def _parent_(self, parent: SymbolicExpression):
        """
        Route parenting to the compiled expression of the query (its outer wrapper) rather than to
        the query node itself, building the snapshot first. The ``_building_`` re-entrancy guard
        prevents recursion while the query is wiring its own wrapper layers during :meth:`build`.
        """
        if not self._building_:
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
