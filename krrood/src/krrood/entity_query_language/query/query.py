"""
Query abstractions for the Entity Query Language.

This module implements query composition and evaluation, including
selection, filtering, grouping, ordering, distinct handling, and
quantification over symbolic expressions.
"""

from __future__ import annotations

import uuid
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
    Set,
    Tuple,
    Callable,
    Self,
    Iterator,
)

from krrood.entity_query_language.core.mapped_variable import CanBehaveLikeAVariable
from krrood.entity_query_language.core.expression_structure import chain_root
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
from krrood.entity_query_language.query.result_transformers import (
    Ordering,
    Quantification,
    ResultTransformer,
)
from krrood.entity_query_language.core.base_expressions import (
    OperationResult,
    SymbolicExpression,
    UnaryExpression,
    Selectable,
    UnificationDict,
)
from krrood.entity_query_language.evaluable import Evaluable
from krrood.entity_query_language.cache_data import (
    SeenSet,
)
from krrood.entity_query_language.evaluation_context import get_evaluation_context
from krrood.entity_query_language.core.variable import (
    InstantiatedVariable,
    ExternallySetVariable,
)
from krrood.entity_query_language.enums import DomainSource
from krrood.entity_query_language.exceptions import (
    UnsupportedNegation,
    NonPositiveLimitValue,
)
from krrood.entity_query_language.operators.aggregators import Aggregator
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
    from krrood.entity_query_language.backends import QueryBackend

ResultMapping = Callable[[Iterator[OperationResult]], Iterator[OperationResult]]
"""
A function that maps the results of a query to a new set of results.
"""


@dataclass
class CachedResultStream:
    """
    A lazily-filled, replayable view over a source iterator.

    The source is advanced on demand and each produced item is buffered, so the stream can be
    iterated many times — once per outer row that reaches an uncorrelated subquery — while the
    underlying computation runs at most once. Filling lazily preserves short-circuiting for callers
    that stop early.
    """

    _source: Iterator[OperationResult]
    """
    The underlying result iterator, advanced at most once per produced item.
    """
    _buffer: List[OperationResult] = field(default_factory=list)
    """
    The results produced so far, replayed to every iterator.
    """
    _exhausted: bool = field(default=False)
    """
    Whether the source has been fully consumed.
    """

    def __iter__(self) -> Iterator[OperationResult]:
        # Sentinel distinguishing a genuinely exhausted source from a ``None`` result.
        stream_exhausted = object()
        index = 0
        # Not ``while not self._exhausted``: an already-exhausted stream must still replay its
        # buffer to later iterators, so the loop always runs and the exhaustion guard below only
        # stops the pulling of new items, not the replay of buffered ones.
        while True:
            if index < len(self._buffer):
                yield self._buffer[index]
                index += 1
                continue
            if self._exhausted:
                return
            # Sentinel form of next() rather than try/except StopIteration: this method is a
            # generator, and per PEP 479 a StopIteration raised inside it would surface as a
            # RuntimeError instead of ending iteration.
            next_item = next(self._source, stream_exhausted)
            if next_item is stream_exhausted:
                self._exhausted = True
                return
            self._buffer.append(next_item)
            yield next_item
            index += 1


def modifies_query_structure(modifier):
    """
    Mark a query modifier so the query is flagged dirty once the modifier has updated its builders,
    causing the next :meth:`build` to recompile the product.
    """

    @wraps(modifier)
    def wrapper(self, *args, **kwargs):
        result = modifier(self, *args, **kwargs)
        self._mark_dirty_()
        return result

    return wrapper


@monitored
@dataclass(eq=False, repr=False)
class Query(
    Evaluable,
    MultiArityExpressionThatPerformsACartesianProduct,
    CanBehaveLikeAVariable[T],
    ABC,
):
    """
    Describes the queried object(s), could be a query over a single variable or
    a set of variables, also describes the condition(s)/properties of the
    queried object(s).
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
    Mapping functions that map the results of the query to a new set of
    results.
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
    The ordering specification applied as a pipeline stage, if the query is ordered.
    """

    _quantifier_builder_: Optional[QuantifierBuilder] = field(default=None, init=False)
    """
    The quantification specification applied as a pipeline stage. Defaults to `An`, which accepts all
    results.
    """
    _dirty_: bool = field(default=True, init=False)
    """
    Whether anything needs (re)building. Any modifier method marks the query dirty; :meth:`build`
    is a no-op while clean and otherwise applies only the parts flagged below. The query is never
    frozen, so it can always be modified, even after being built or embedded as a child.
    """
    _building_: bool = field(default=False, init=False)
    """
    Re-entrancy guard set while :meth:`build` wires the compiled product, so that parenting the query
    to its product does not recursively trigger another build.
    """
    _is_compiled_product_: bool = field(default=False, init=False)
    """
    Whether this instance is a compiled product (wired as the cartesian-product node) rather than a
    specification. A specification delegates evaluation to the product it compiles, so that the specification behaves like its
    product while never being the mutated, embedded node itself.
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
        # ``_group_`` and ``_distinct_on_ids_`` are cached_property values stored in ``__dict__``;
        # popping the keys invalidates them so they recompute from the new modifier state on next
        # access. Assigning ``None`` would instead cache ``None`` and keep returning it.
        self.__dict__.pop("_group_", None)
        self.__dict__.pop("_distinct_on_ids_", None)

    def evaluate(self, backend: Optional[QueryBackend] = None) -> Iterator:
        """
        Evaluate the query using the given backend, returning an iterator over
        the results.

        Builds the query eagerly so that ``evaluate`` consistently marks it as built regardless of
        when the returned iterator is consumed; ``build`` is idempotent.

        :param backend: The query backend to evaluate with. Defaults to
            the ``EntityQueryLanguageBackend`` (native python
            evaluation).
        """
        self.build()
        return super().evaluate(backend)

    def _evaluate_natively_(self) -> Iterator:
        """
        Evaluate the query in this python process, returning an iterator over its results (ordered
        and quantified by the result pipeline). This is the engine used by the
        ``EntityQueryLanguageBackend``.
        """
        self.build()
        if not self._is_compiled_product_:
            return self._expression_.evaluate()
        return MultiArityExpressionThatPerformsACartesianProduct.evaluate(self)

    @modifies_query_structure
    def where(self, *conditions: ConditionType) -> Self:
        """
        Set the conditions that describe the query object.

        The conditions are chained using AND.
        :param conditions: The conditions that describe the query
            object.
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
        Set the conditions that describe the query object.

        The conditions are chained using AND.
        :param conditions: The conditions that describe the query
            object.
        :return: This query.
        """
        if self._having_builder_ is None:
            self._having_builder_ = HavingBuilder(conditions=conditions, query=self)
        else:
            self._having_builder_.conditions += conditions
        return self

    @modifies_query_structure
    def ordered_by(
        self,
        variable: TypingUnion[Selectable[T], Any],
        descending: bool = False,
        key: Optional[Callable] = None,
    ) -> Self:
        """
        Order the results by the given variable, using the given key function
        in descending or ascending order.

        :param variable: The variable to order by.
        :param descending: Whether to order the results in descending
            order.
        :param key: A function to extract the key from the variable
            value.
        """
        self._ordered_by_builder_ = OrderedByBuilder(
            self, variable, descending=descending, key=key
        )
        return self

    @modifies_query_structure
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
        self._seen_results = SeenSet(keys=tuple(v._id_ for v in self._distinct_on))
        self._results_mapping.append(self._get_distinct_results_)
        return self

    @modifies_query_structure
    def grouped_by(
        self, *variables_to_group_by: TypingUnion[Selectable, Any]
    ) -> TypingUnion[Self, T]:
        """
        Specify the variables to group the results by.

        :param variables_to_group_by: The variables to group the results
            by.
        :return: This query.
        """
        self._grouped_by_builder_ = GroupedByBuilder(self, variables_to_group_by)
        return self

    @modifies_query_structure
    def limit(self, n: int) -> Self:
        """
        Limit the number of results to n.

        :param n: The maximum number of results to return.
        :return: This query.
        """
        self._limit_ = n
        if not isinstance(self._limit_, int) or self._limit_ <= 0:
            raise NonPositiveLimitValue(self._limit_)
        return self

    @modifies_query_structure
    def _quantify_(
        self,
        quantifier_type: Type[ResultQuantifier] = An,
        quantification_constraint: Optional[ResultQuantificationConstraint] = None,
    ) -> Self:
        """
        Specify the quantifier type and constraint for the query results, also
        build the query.

        :param quantifier_type: The type of the quantifier to be used.
        :param quantification_constraint: The constraint to apply to the
            quantifier.
        :return: This query.
        """
        self._quantifier_builder_ = QuantifierBuilder(
            self, quantifier_type, quantification_constraint
        )
        return self

    def __enter__(self):
        """
        Make sure the query is built before entering the context manager for
        rule trees.
        """
        self.build()
        expression = self._conditions_root_
        SymbolicExpression._symbolic_expression_stack_.append(expression)
        return expression

    def build(self) -> Self:
        """
        Build (or rebuild) the query, caching a freshly compiled product expression.

        This query is a stable specification: it holds the modifiers and a stable identity that derived
        references and rules point at, but it is never itself the evaluated node. Each build compiles
        a fresh product tree from the current specification via :meth:`_compile_` and stores it in
        :attr:`_expression_`; the specification is never frozen, so any modifier marks it dirty and the next
        build produces a new product. Because every build yields a new tree (rather than mutating the
        previous one in place), a product already embedded elsewhere stays frozen against later edits.

        :return: This query.
        """
        if not self._dirty_:
            return self
        self._expression_ = self._compile_()
        self._dirty_ = False
        return self

    def _compile_(self) -> SymbolicExpression:
        """
        Compile a fresh, independent product expression from this specification — the single compile path.

        The product is a separate node graph that shares this query's identifier and selected
        variables, so it evaluates identically and co-references the same variables, while later
        edits that rebuild the specification cannot reach it. It is produced by replaying the specification's modifiers
        onto a fresh instance and wiring that instance as the cartesian-product node, rather than by
        cloning a live graph, so grouping, distinct, and self-referential ordering all compile
        correctly.

        :return: The compiled inner product node (its ordering/quantification wrappers are reachable
            through its :attr:`_expression_`).
        """
        product = type(self)(_selected_variables_=self._selected_variables_)
        product._id_ = self._id_
        product._expression_id_cache_ = {}
        if self._where_builder_ is not None:
            product.where(*self._where_builder_.conditions)
        if self._grouped_by_builder_ is not None:
            product.grouped_by(*self._grouped_by_builder_.variables_to_group_by)
        if self._having_builder_ is not None:
            product.having(*self._having_builder_.conditions)
        if self._ordered_by_builder_ is not None:
            ordering = self._ordered_by_builder_
            product.ordered_by(
                ordering.variable, descending=ordering.descending, key=ordering.key
            )
        if self._distinct_on:
            product.distinct(*self._distinct_on)
        if self._limit_ is not None:
            product.limit(self._limit_)
        quantifier = self._quantifier_builder_
        product._quantify_(quantifier.type, quantifier.quantification_constraint)
        product._wire_in_place_()
        return product

    def _wire_in_place_(self) -> Self:
        """
        Wire this instance as the cartesian-product node: build the data-source chain (Where /
        GroupedBy / Having) and selected variables as its children. Ordering and quantification are
        applied as result-pipeline stages during evaluation, so the product is its own compiled
        expression. Called once on a freshly compiled product instance.

        :return: This instance.
        """
        self._building_ = True
        self._is_compiled_product_ = True
        self._rewire_data_source_chain_()
        self._dirty_ = False
        self._building_ = False
        return self

    def _rewire_data_source_chain_(self) -> None:
        """
        Wire the data-source chain (Where / GroupedBy / Having) and the selected variables as the
        children of this product node.
        """
        head = self._data_source_chain_head_()
        children = (
            self._selected_variables_
            if head is None
            else (head, *self._selected_variables_)
        )
        self.update_children(*children)

    def _data_source_chain_head_(self) -> Optional[SymbolicExpression]:
        """
        Build the head of the data-source chain that feeds the selected variables. At most one of
        Having / GroupedBy / Where heads the chain, with precedence ``Having > GroupedBy > Where``
        since each already incorporates the previous one.

        :return: The chain head, or ``None`` if the query is unfiltered/ungrouped.
        """
        if self._group_ and self._grouped_by_builder_ is None:
            self._grouped_by_builder_ = GroupedByBuilder(self)

        if self._having_builder_ is not None:
            self._having_builder_.grouped_by = self._grouped_by_builder_.expression
            return self._having_builder_.expression
        if self._grouped_by_builder_ is not None:
            return self._grouped_by_builder_.expression
        if self._where_builder_ is not None:
            return self._where_builder_.expression
        return None

    def _evaluate__(
        self,
        sources: OperationResult,
    ) -> Iterable[OperationResult]:
        """
        Evaluate the query by constraining values, updating conclusions,
        and selecting variables.

        A specification delegates to its compiled product so that evaluating the specification as an expression behaves
        exactly like evaluating the product; only the compiled product runs the real evaluation.

        This query is the scope that isolates a nested subquery: evaluated as a subquery (nested
        inside another query's evaluation) it ignores the surrounding bindings and ranges over its own
        domain, and its result is cached so it is computed once and replayed to every outer row.
        Evaluated directly (as the outermost query) it threads the incoming sources unchanged.
        """
        if not self._is_compiled_product_:
            self.build()
            yield from self._expression_._evaluate__(sources)
            return

        evaluation_context = get_evaluation_context()
        if evaluation_context is None or not self._is_nested_subquery_(
            evaluation_context
        ):
            yield from self._produce_results_(sources)
            return

        cached_stream = evaluation_context.subquery_result_cache.get_or_create(
            self._id_,
            lambda: CachedResultStream(self._produce_results_(OperationResult({}))),
        )
        yield from cached_stream

    def _is_nested_subquery_(self, evaluation_context) -> bool:
        """
        :param evaluation_context: The active evaluation context.
        :return: Whether this compiled query is evaluating as a nested subquery rather than as the
            outermost query of the current evaluation. The first compiled query to evaluate claims the
            outermost role; any other is nested.
        """
        return evaluation_context.outermost_query_claim.is_nested(self._id_)

    def _produce_results_(self, sources: OperationResult) -> Iterator[OperationResult]:
        """
        Produce this product's results: the projected, result-mapped rows of its cartesian product.

        :param sources: The current bindings.
        :return: An iterator over the query's result rows.
        """
        results = (
            self._get_operation_result_(result)
            for result in self._apply_results_mapping_(
                self._evaluate_product_(sources),
            )
        )
        for transformer in self._result_transformers_:
            results = transformer.transform(results)
        yield from results

        if self._seen_results is not None:
            self._seen_results.clear()

    @cached_property
    def _result_transformers_(self) -> List[ResultTransformer]:
        """
        :return: The ordered result-pipeline stages applied to this query's produced rows: ordering
            (when configured), then quantification. Inspectable as :attr:`_result_stages_`.
        """
        transformers: List[ResultTransformer] = []
        if self._ordered_by_builder_ is not None:
            ordering = self._ordered_by_builder_
            transformers.append(
                Ordering(
                    variable=ordering.variable,
                    descending=ordering.descending,
                    key=ordering.key,
                )
            )
        quantifier = self._quantifier_builder_
        constraint = (
            quantifier.quantification_constraint
            or quantifier.type._default_constraint_()
        )
        transformers.append(
            Quantification(
                quantifier_type=quantifier.type, constraint=constraint, owner=self
            )
        )
        return transformers

    @property
    def _result_stages_(self) -> List[ResultTransformer]:
        """
        :return: The result-pipeline stages of this query's compiled product (ordering,
            quantification), so an inspector can identify how results are ordered and quantified
            without traversing or evaluating.
        """
        self.build()
        return self._expression_._result_transformers_

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
        Update the result dictionary with values from distinct-on variables if
        not already present.

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

    @property
    def is_constrained_or_grouped(self) -> bool:
        """
        :return: ``True`` when this query carries a ``WHERE``, ``HAVING``, or
            non-empty ``GROUP BY`` clause (i.e. it filters beyond its selection or groups results).
        """
        if self._where_expression_ is not None:
            return True
        if self._having_expression_ is not None:
            return True
        grouped = self._grouped_by_expression_
        return grouped is not None and bool(grouped.variables_to_group_by)

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
            """
            Update the aggregated and non-aggregated variable collections based
            on the given variable.

            :param variable_: The variable to check.
            """
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

    def aggregated_selections(
        self, group_key_root_ids: Set[uuid.UUID]
    ) -> List[SymbolicExpression]:
        """
        :param group_key_root_ids: The chain-root variable ids of the GROUP BY keys (see
            :attr:`~krrood.entity_query_language.query.operations.GroupedBy.group_key_root_ids`).
        :return: The selected expressions aggregated over those keys — i.e. the selections that are
            not themselves group keys.
        """
        return [
            variable
            for variable in self._selected_variables_
            if variable._id_ not in group_key_root_ids
        ]

    def _apply_results_mapping_(
        self, results: Iterator[OperationResult]
    ) -> Iterable[OperationResult]:
        """
        Process and transform an iterable of results based on predefined
        mappings and ordering.

        This method applies a sequence of result transformations defined
        in the instance, using a series of mappings to modify the
        results.

        :param results: An iterable containing dictionaries that
            represent the initial result set to be transformed.
        :return: An iterable containing dictionaries that represent the
            transformed data.
        """
        for result_mapping in self._results_mapping:
            results = result_mapping(results)
        return results

    def _as_embeddable_child_(self, parent: SymbolicExpression) -> SymbolicExpression:
        """
        Embed this query's compiled product when it becomes an operand of another expression.

        The product is captured as it stands at embed time; because each rebuild produces a new
        product rather than mutating the previous one, a later edit to this query cannot change the
        already-embedded copy. The product shares this query's identifier, so a derived reference
        (``query.name``) embedded over it resolves against the query's own results. While the product
        is wiring itself during :meth:`_wire_in_place_` (``_building_``), its own in-place node is
        returned.

        :param parent: The expression about to take this query as a child.
        :return: The compiled product to embed (or the in-place node during self-wiring).
        """
        if self._building_:
            return self._expression_
        self.build()
        return self._expression_

    @property
    def _root_(self) -> SymbolicExpression:
        """
        Resolve the root through the compiled product.

        ``SatisfiedConditionTracker`` treats ``expression._conditions_root_ is expression._root_``
        as "this query has no where/having condition" (both fall back to the same node when no
        ``Filter`` exists). Since :attr:`_conditions_root_` already resolves within the compiled
        product, :attr:`_root_` must too, or that comparison always sees two different objects (the
        specification and its product) even when the product itself has no condition.

        :return: The root of the compiled product's tree.
        """
        self.build()
        if not self._is_compiled_product_:
            return self._expression_._root_
        return SymbolicExpression._root_.fget(self)

    @property
    def _conditions_root_(self) -> Optional[SymbolicExpression]:
        """
        Resolve the conditions root within the compiled product, so rule definition (:meth:`__enter__`)
        and conclusions attach to the node that is actually evaluated.

        :return: The conditions root of the compiled product.
        """
        self.build()
        if not self._is_compiled_product_:
            return self._expression_._conditions_root_
        return SymbolicExpression._conditions_root_.fget(self)

    @property
    def _all_expressions_(self) -> Iterator[SymbolicExpression]:
        """
        Traverse the whole node tree through the compiled product.

        A specification holds only its selected variables and modifiers; the ``where`` / ``having``
        conditions (and the variables they introduce) live in the compiled product. Delegating the
        traversal keeps the specification behaving like its product, so consumers that scan every node — for
        example verbalization's referent disambiguation — also see condition-only referents.
        """
        self.build()
        if not self._is_compiled_product_:
            return self._expression_._all_expressions_
        return SymbolicExpression._all_expressions_.fget(self)

    @property
    def _descendants_(self) -> Iterator[SymbolicExpression]:
        """
        Traverse descendants through the compiled product.

        A selected variable's own ``_all_expressions_``/``_conditions_root_`` reach this query
        only indirectly, via ``self._root_._descendants_`` on the base class. Without this override
        that walk sees only the specification's structural children (its selected variables), never the
        ``where`` / ``having`` conditions, which live in the compiled product — so callers reaching
        this query as a root through a plain child, rather than through the query object itself,
        must get the same delegation as :attr:`_all_expressions_` and :attr:`_conditions_root_`.
        """
        self.build()
        if not self._is_compiled_product_:
            return self._expression_._descendants_
        return SymbolicExpression._descendants_.fget(self)

    @UnaryExpression._parent_.setter
    def _parent_(self, parent: SymbolicExpression):
        """
        Route parenting to the compiled product rather than to the specification node, building it first. The
        ``_building_`` re-entrancy guard prevents recursion while a product instance wires itself
        during :meth:`_wire_in_place_`.
        """
        if not self._building_:
            self.build()
        if not self._is_compiled_product_:
            self._expression_._parent_ = parent
            return
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

    @property
    def selected_aggregator(self) -> "Optional[Aggregator]":
        """
        :return: The :class:`~krrood.entity_query_language.operators.aggregators.Aggregator`
            this entity selects, or ``None`` when its selection is not an aggregator.
        """
        var = self.selected_variable
        return var if isinstance(var, Aggregator) else None

    def aggregated_selections(
        self, group_key_root_ids: Set[uuid.UUID]
    ) -> List[SymbolicExpression]:
        """
        When the selection is an :class:`InstantiatedVariable`, its aggregated
        fields are the child variables not rooted in a group key; otherwise the
        generic per-selection rule applies.

        :param group_key_root_ids: The chain-root variable ids of the
            GROUP BY keys.
        :return: The selected expressions aggregated over those keys.
        """
        selected = self.selected_variable
        if isinstance(selected, InstantiatedVariable):
            return [
                child
                for child in selected._child_vars_.values()
                if chain_root(child)._id_ not in group_key_root_ids
            ]
        return super().aggregated_selections(group_key_root_ids)
