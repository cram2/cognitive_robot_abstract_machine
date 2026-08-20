"""
Evaluation context and observer protocol for the Entity Query Language.

Extracted into its own module so that :mod:`core.base_expressions` can import the
context infrastructure without pulling in the full :mod:`evaluation` module and the
circular dependency chain it carries.
"""

from __future__ import annotations

import uuid
import weakref
from abc import ABC
from contextvars import ContextVar
from dataclasses import dataclass, field

from ordered_set import OrderedSet
from typing_extensions import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.core.base_expressions import (
        OperationResult,
        SymbolicExpression,
    )

_evaluation_context_var: ContextVar[Optional[EvaluationContext]] = ContextVar(
    "_evaluation_context", default=None
)


def get_evaluation_context() -> Optional[EvaluationContext]:
    """
    :return: The current :class:`EvaluationContext`, or ``None`` if called outside an active evaluation.
    """
    return _evaluation_context_var.get()


def set_evaluation_context(
    evaluation_context: Optional[EvaluationContext],
):
    """
    Set or clear the current evaluation context and return the reset token.

    :param evaluation_context: The context to set, or ``None`` to clear.
    :return: A :class:`contextvars.Token` that can be passed to
        :meth:`contextvars.ContextVar.reset` to restore the previous value.
    """
    return _evaluation_context_var.set(evaluation_context)


class EvaluationObserver(ABC):
    """
    Observer for evaluation events in the EQL evaluation pipeline.
    """

    def on_evaluate_enter(
        self, expression: SymbolicExpression, sources: Optional[OperationResult] = None
    ) -> None:
        """
        Called when entering an expression's _evaluate_ method.
        """

    def on_evaluate_exit(self, expression: SymbolicExpression) -> None:
        """
        Called when exiting an expression's _evaluate_ method.
        """

    def on_result_yielded(
        self, expression: SymbolicExpression, result: OperationResult
    ) -> None:
        """
        Called for each OperationResult yielded from _evaluate_.
        """

    def on_conclusions_processed(
        self, expression: SymbolicExpression, result: OperationResult
    ) -> None:
        """
        Called after _evaluate_conclusions_and_update_bindings_ completes.
        """


@dataclass
class ActiveConditionsRoot:
    """
    Tracks which node is the active conditions root for the current evaluation pass.

    A node reused as the condition of more than one ``Filter`` has no single correct
    "root" — the right answer depends on which evaluation is currently running, not on
    the node's construction history. The first node to set this during a pass wins;
    nested evaluations within the same pass never reassign it.
    """

    _root_id: Optional[uuid.UUID] = field(default=None, init=False)
    """
    Identifier of the active root, or ``None`` before one has been set this pass.
    """

    has_condition: bool = field(default=False, init=False)
    """
    Whether the active root came from a genuine ``Filter``, rather than the Filter-less
    fallback to the evaluation's own starting expression.
    """

    def set_active_root_if_not_set(
        self, root: SymbolicExpression, has_condition: bool
    ) -> None:
        """
        Set *root* as the active conditions root for this pass, unless one is already
        set.

        :param root: The node to set, normally
            ``originating_expression._conditions_root_``.
        :param has_condition: Whether *root* came from a genuine ``Filter``, recorded so
            that it need not be recomputed later in the pass from a node that may itself
            be shared and structurally ambiguous.
        """
        if self._root_id is None:
            self._root_id = root._id_
            self.has_condition = has_condition

    def is_active_root(self, node: SymbolicExpression) -> bool:
        """:return: ``True`` if *node* is the active conditions root for this pass."""
        return self._root_id == node._id_


@dataclass
class TruthValueOperatorChildren:
    """
    Tracks which nodes were evaluated as a direct child of a
    :class:`~krrood.entity_query_language.core.base_expressions.TruthValueOperator` during
    the current evaluation pass.

    Whether a node counts as a condition participant depends on which parent evaluated it
    in this pass, not on the node's construction history: a node reused elsewhere in the
    :class:`~krrood.entity_query_language.core.base_expressions.SymbolicExpression`'s own
    directed acyclic graph of parents keeps a structural parent per position, but only
    the first-ever attachment is its primary ``_parent_``, which may belong to an
    unrelated position. Recording the dynamic, per-pass parent here instead of reading
    the node's primary ``_parent_`` avoids that ambiguity.
    """

    _ids: Set[uuid.UUID] = field(default_factory=set, init=False)
    """
    Identifiers of the nodes evaluated as a direct child of a ``TruthValueOperator`` so
    far this pass.
    """

    def record(self, expression_id: uuid.UUID) -> None:
        """
        Record *expression_id* as evaluated as a direct child of a
        ``TruthValueOperator`` during the current pass.
        """
        self._ids.add(expression_id)

    def __contains__(self, expression_id: uuid.UUID) -> bool:
        return expression_id in self._ids


@dataclass
class EvaluatedExpressionIds:
    """
    Tracks every expression id evaluated so far during the current evaluation pass.

    Used to distinguish evaluated-from-skipped logical operators (for example, short-
    circuited OR/AND branches) when building inference explanations.
    """

    _ids: OrderedSet = field(default_factory=OrderedSet, init=False)
    """
    The expression ids recorded as evaluated so far this pass.
    """

    _snapshot: Optional[Tuple[int, OrderedSet]] = field(default=None, init=False)
    """
    Cached ``(length, snapshot)`` pair.

    The id set is append-only, so its length is a valid version key: a snapshot taken
    while the set has a given length is reused instead of copying the whole set again.
    """

    def record(self, expression_id: uuid.UUID) -> None:
        """
        Record *expression_id* as evaluated during the current pass.
        """
        self._ids.add(expression_id)

    def merge(self, other: OrderedSet) -> None:
        """
        Merge *other* into the recorded ids (for example, ids evaluated by an earlier
        stage of the same result chain).
        """
        self._ids.update(other)

    def __iter__(self) -> Iterator[uuid.UUID]:
        return iter(self._ids)

    def snapshot(self) -> OrderedSet:
        """:return: An immutable snapshot of the ids recorded so far, reusing the cached one
        when the set has not grown since it was last taken."""
        current_length = len(self._ids)
        if self._snapshot is None or self._snapshot[0] != current_length:
            self._snapshot = (current_length, OrderedSet(self._ids))
        return self._snapshot[1]


@dataclass
class OutermostQuery:
    """
    Tracks which compiled query node is the outermost query for the current evaluation
    pass.

    A query node is the scope that isolates a nested subquery from its surrounding
    bindings. The first compiled query node to evaluate during a pass is the outermost
    one; any other compiled query node that evaluates during the same pass is nested
    inside it.
    """

    node: Optional[SymbolicExpression] = field(default=None, init=False)
    """
    The compiled query node holding the outermost role, or ``None`` before any node
    takes it.
    """

    def is_nested(self, query: SymbolicExpression) -> bool:
        """
        Record *query* as the outermost query if none is recorded yet, then report
        whether *query* is nested inside some other, already-recorded outermost query.

        :param query: The compiled query node.
        :return: Whether *query* is a nested subquery (``True``) or the outermost query
            (``False``).
        """
        if self.node is None:
            self.node = query
        return self.node._id_ != query._id_


@dataclass
class SubqueryResultCache:
    """
    Caches an uncorrelated subquery's result stream per query node for one evaluation
    pass.

    So a subquery reached from many outer rows is computed once and its cached stream is
    replayed to each of them, instead of recomputing it on every outer row.
    """

    _streams: Dict[uuid.UUID, Any] = field(default_factory=dict, init=False)
    """
    Cached result stream keyed by the compiled query node's identifier.
    """

    def get_or_create(self, query_id: uuid.UUID, factory: Callable[[], Any]) -> Any:
        """
        Return the cached stream for *query_id*, creating it via *factory* if not
        already cached.

        :param query_id: The compiled query node's identifier.
        :param factory: Called at most once to build the stream if it isn't already
            cached.
        :return: The cached stream for *query_id*.
        """
        if query_id not in self._streams:
            self._streams[query_id] = factory()
        return self._streams[query_id]


@dataclass
class EvaluationContext:
    """
    Carries observer state through the evaluation pipeline.
    """

    observers: List[EvaluationObserver] = field(default_factory=list)
    """
    List of observers to notify of evaluation events.
    """

    subtree_containment_cache: Dict[
        Tuple[uuid.UUID, Type[SymbolicExpression]], bool
    ] = field(default_factory=dict)
    """
    Memoizes, per ``(node id, expression type)``, whether a node's subtree contains a
    descendant of that type — a structural fact constant for the duration of an
    evaluation, so the hot path answers it once instead of re-walking the subtree on
    every step.
    """

    expression_index_cache: Dict[
        uuid.UUID, weakref.WeakValueDictionary[uuid.UUID, SymbolicExpression]
    ] = field(default_factory=dict)
    """
    Memoizes, per tree-root id, an ``id -> node`` index built once per evaluation and
    reused for every lookup instead of re-scanning the tree.

    ..warning:: The index holds nodes only through weak references. A context can be captured past
        its evaluation (for example by an inference explanation); strong references here would pin
        the whole query tree and its variables' domains.
    """

    active_conditions_root: ActiveConditionsRoot = field(
        default_factory=ActiveConditionsRoot
    )
    """
    Tracks which node is the active conditions root for the current evaluation pass.
    """

    truth_value_operator_children: TruthValueOperatorChildren = field(
        default_factory=TruthValueOperatorChildren
    )
    """
    Tracks which nodes were evaluated as a direct child of a ``TruthValueOperator``
    during the current evaluation pass.
    """

    evaluated_expression_ids: EvaluatedExpressionIds = field(
        default_factory=EvaluatedExpressionIds
    )
    """
    Tracks every expression id evaluated so far during the current evaluation pass.
    """

    satisfied_condition_ids: Optional[OrderedSet] = field(default=None)
    """
    The satisfied condition-expression ids for the current evaluation iteration, or
    ``None`` if unset.
    """

    outermost_query: OutermostQuery = field(default_factory=OutermostQuery)
    """
    Tracks which compiled query node is the outermost query for the current evaluation
    pass.
    """

    subquery_result_cache: SubqueryResultCache = field(
        default_factory=SubqueryResultCache
    )
    """
    Caches each nested subquery's result stream for the current evaluation pass.
    """

    def is_child_of_truth_value_operator(self, expression: SymbolicExpression) -> bool:
        """
        :param expression: The symbolic expression to test.
        :return: ``True`` if *expression* was evaluated as a direct child of a
            ``TruthValueOperator`` during the current evaluation pass.
        """
        return expression._id_ in self.truth_value_operator_children

    def on_evaluate_enter(
        self,
        *,
        expression: SymbolicExpression,
        sources: Optional[OperationResult] = None,
    ) -> None:
        """
        Notify all observers that evaluation of *expression* is about to begin.

        :param expression: The expression being entered.
        :param sources: The incoming :class:`OperationResult` carrying bindings, or
            ``None``.
        """
        for observer in self.observers:
            observer.on_evaluate_enter(expression, sources)

    def on_evaluate_exit(self, *, expression: SymbolicExpression) -> None:
        """
        Notify all observers that evaluation of *expression* has finished.

        :param expression: The expression that just finished evaluating.
        """
        for observer in self.observers:
            observer.on_evaluate_exit(expression)

    def on_result_yielded(
        self,
        *,
        expression: SymbolicExpression,
        result: OperationResult,
    ) -> None:
        """
        Notify all observers that *expression* has yielded *result*.

        :param expression: The expression that produced the result.
        :param result: The :class:`OperationResult` that was yielded.
        """
        for observer in self.observers:
            observer.on_result_yielded(expression, result)

    def on_conclusions_processed(
        self,
        *,
        expression: SymbolicExpression,
        result: OperationResult,
    ) -> None:
        """
        Notify all observers that conclusions have been processed for *expression*.

        :param expression: The expression whose conclusions were processed.
        :param result: The :class:`OperationResult` after conclusion processing.
        """
        for observer in self.observers:
            observer.on_conclusions_processed(expression, result)
