"""
This module defines the fundamental symbolic expression classes for the Entity Query
Language.

It contains the base classes for all expressions, results, and binding management used
during query evaluation.
"""

from __future__ import annotations

import itertools
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import UserDict
from contextlib import AbstractContextManager
from copy import copy
from dataclasses import dataclass, field
from functools import cached_property
from uuid import UUID

from ordered_set import OrderedSet
from typing_extensions import (
    Collection,
    Dict,
    Any,
    Optional,
    ClassVar,
    List,
    Iterator,
    Union as TypingUnion,
    Tuple,
    Set,
    Self,
    TYPE_CHECKING,
    Generic,
    Type,
    TypeAlias,
)

from krrood.entity_query_language.evaluation_context import (
    get_evaluation_context,
    set_evaluation_context,
    _evaluation_context_var,
)
from krrood.entity_query_language.exceptions import NoExpressionFoundForGivenID
from krrood.entity_query_language.utils import make_list, T, make_set, is_iterable
from krrood.symbol_graph.symbol_graph import SymbolGraph

if TYPE_CHECKING:
    from krrood.entity_query_language.rules.conclusion import Conclusion, ConclusionType
    from krrood.entity_query_language.core.variable import Variable
    from krrood.entity_query_language.query.query import Query

Bindings: TypeAlias = Dict[uuid.UUID, Any]
"""
A dictionary for expressions' bindings in EQL that maps the expression's unique
identifier to its value.
"""


@dataclass
class RuleTreeContext:
    """
    A ``with``-block anchor together with the parent edge its rule tree reaches it by.

    A shared node has several parents, so which parent a rule-tree edit must happen above
    is only defined relative to the branch that is asking. Recording that edge when the
    block is entered keeps the answer independent of which parent happened to be attached
    first.
    """

    condition: SymbolicExpression
    """
    The condition node the ``with`` block anchors on.
    """

    owning_parent: Optional[SymbolicExpression]
    """
    The parent through which the asking rule tree reaches :attr:`condition`, kept current
    by whoever splices a new node into that edge.
    """


@dataclass(eq=False)
class SymbolicExpression(AbstractContextManager):
    """
    Base class for all symbolic expressions.

    Symbolic expressions form a rooted directed acyclic graph and are evaluated lazily
    to produce bindings for variables, subject to logical constraints.
    """

    _id_: uuid.UUID = field(init=False, repr=False, default_factory=uuid.uuid4)
    """
    Unique identifier of this node.
    """

    _conclusions_: Set[Conclusion] = field(init=False, default_factory=set)
    """
    Set of conclusion expressions attached to this node, these are evaluated when the
    truth value of this node is true during evaluation.
    """

    _symbolic_expression_stack_: ClassVar[List[RuleTreeContext]] = []
    """
    The current stack of symbolic expressions that has been entered using the ``with``
    statement, each paired with the parent edge its rule tree reaches it by.
    """

    _children_: List[SymbolicExpression] = field(
        init=False, repr=False, default_factory=list
    )
    """
    The children expressions of this symbolic expression.
    """

    _parents_: List[SymbolicExpression] = field(
        init=False, repr=False, default_factory=list
    )
    """
    The parents expressions of this symbolic expression.
    """

    _parent__: Optional[SymbolicExpression] = field(
        init=False, repr=False, default=None
    )
    """
    Internal attribute used to track the parent symbolic expression of this expression.
    """

    _expression_: SymbolicExpression = field(init=False, repr=False)
    """
    Useful when this expression is a builder that wires multiple components together to
    create the final expression.

    This defaults to Self.
    """

    _limit_: Optional[int] = field(init=False, repr=False, default=None)
    """
    The maximum number of results to return during evaluation.
    """

    _expression_id_cache_: dict[uuid.UUID, SymbolicExpression] = field(
        init=False, repr=False, default_factory=dict, compare=False
    )
    """
    Cache of expressions by their unique identifier.
    """

    def __post_init__(self):
        self._expression_ = self

    def _node_for_new_position_(self) -> SymbolicExpression:
        """
        :return: The node to wire into a new tree position without corrupting this one.

        For a normal expression this is a fresh clone: it gets a new identity and none of the
        original's parents, children, or conclusions, so wiring it into a new position cannot
        overwrite the original's ``_parent_`` (which its old parent still references) nor leak
        conclusions back into it. Children are not deep-copied — only this node is cloned.

        ..note:: :class:`~krrood.entity_query_language.core.mapped_variable.MappedVariable` nodes are
            shared-identity singletons and override this to return themselves, since sharing them
            across positions is intended and safe.
        """
        clone = self.__class__.__new__(self.__class__)
        clone.__dict__.update(self.__dict__)
        clone._id_ = uuid.uuid4()
        clone._children_ = []
        clone._parents_ = []
        clone._parent__ = None
        clone._expression_ = clone
        clone._conclusions_ = set()
        return clone

    def _get_expression_by_id_(self, id_: uuid.UUID) -> SymbolicExpression:
        """
        Retrieve the expression with the given ID from the collection of all
        expressions.

        :param id_: The unique identifier of the expression to retrieve.
        :return: The expression with the specified ID, or raises
            NoExpressionFoundForGivenID if not found.
        """
        # Per-instance cache stored in _expression_id_cache_ so it is collected with the expression object.
        # A class-level @lru_cache would hold strong refs to `self` indefinitely, keeping
        # query trees (and their domain data) alive well beyond the query's lifetime.
        if id_ in self._expression_id_cache_:
            return self._expression_id_cache_[id_]

        # During an evaluation the tree is immutable, so resolve against the id->node index built
        # once per evaluation instead of re-scanning the whole graph on every (often failing) lookup.
        evaluation_scoped_index = self._evaluation_scoped_expression_index_()
        if evaluation_scoped_index is not None:
            if id_ in evaluation_scoped_index:
                self._expression_id_cache_[id_] = evaluation_scoped_index[id_]
                return self._expression_id_cache_[id_]
            raise NoExpressionFoundForGivenID(self, id_)

        try:
            self._expression_id_cache_[id_] = next(
                expression
                for expression in self._all_expressions_
                if expression._id_ == id_
            )
        except StopIteration:
            raise NoExpressionFoundForGivenID(self, id_)
        return self._expression_id_cache_[id_]

    def tolist(
        self,
        backend=None,
    ) -> list[TypingUnion[T, Dict[TypingUnion[T, SymbolicExpression], T]]]:
        """
        Evaluate and return the results as a list.

        :param backend: Optional query backend; forwarded to :py:meth:`evaluate`.
        """
        return make_list(self.evaluate(backend=backend))

    def first(
        self, backend=None
    ) -> TypingUnion[T, Dict[TypingUnion[T, SymbolicExpression], T]]:
        """
        Evaluate and return the first result of the query object descriptor.

        :param backend: Optional query backend; forwarded to :py:meth:`evaluate`.
        :return: The first result of the query object descriptor.
        :raises StopIteration: If no results are found.
        """
        return next(self.evaluate(backend=backend))

    def evaluate(
        self,
        backend=None,
    ) -> Iterator[TypingUnion[T, Dict[TypingUnion[T, SymbolicExpression], T]]]:
        """
        Evaluate the query and map the results to the correct output data structure.
        This is the exposed evaluation method for users.

        :param backend: Accepted for interface uniformity with ``Query``/``Match``; the
            base symbolic-expression engine always evaluates natively and ignores this
            argument.
        """
        SymbolGraph().remove_dead_instances()
        results = (self._process_result_(res) for res in self._true_results_())
        yield from itertools.islice(results, self._limit_)

    def _true_results_(self) -> Iterator[OperationResult]:
        """
        :return: The raw ``OperationResult`` instances from ``_evaluate_()`` whose truth
            value is true, before :meth:`_process_result_` maps them to output values.
            An expression that is not a :class:`TruthValuedExpression` records a
            selectable value rather than a truth, so its results pass through
            unfiltered regardless of that value's truthiness.
        """
        return (
            result
            for result in self._evaluate_()
            if not isinstance(self, TruthValuedExpression) or result.is_true
        )

    def _replace_child_(
        self, old_child: SymbolicExpression, new_child: SymbolicExpression
    ):
        """
        Replace a child expression with a new child expression.

        :param old_child: The old child expression.
        :param new_child: The new child expression.
        """
        if old_child is new_child:
            return
        _children_ids_ = [v._id_ for v in self._children_]
        child_idx = _children_ids_.index(old_child._id_)
        self._children_[child_idx] = new_child
        new_child._parent_ = self
        old_child._remove_parent_(self)
        self._replace_child_field_(old_child, new_child)

    @abstractmethod
    def _replace_child_field_(
        self, old_child: SymbolicExpression, new_child: SymbolicExpression
    ):
        """
        Replace a child field with a new child expression.

        :param old_child: The old child expression.
        :param new_child: The new child expression.
        """
        pass

    def _remove_parent_(self, parent: SymbolicExpression):
        """
        Remove the parent relationship between this expression and the given parent
        expression.

        When the removed parent is the primary one, another remaining parent is promoted
        so a node that still lives elsewhere in the DAG keeps a valid primary parent.

        :param parent: The parent expression to remove.
        """
        if parent in self._parents_:
            self._parents_.remove(parent)
        if self._id_ in [child._id_ for child in parent._children_]:
            parent._children_.remove(self)
        if parent is self._parent__:
            self._parent__ = self._parents_[-1] if self._parents_ else None

    def _update_children_(
        self, *children: SymbolicExpression
    ) -> Tuple[SymbolicExpression, ...]:
        """
        Update multiple children expressions of this symbolic expression.

        :param children: The new children expressions. Non-``SymbolicExpression`` values
            are wrapped in ``Literal`` instances before being attached.
        :return: A tuple of the updated child expressions corresponding to the provided
            ``children`` arguments.
        """
        from krrood.entity_query_language.core.variable import Literal

        children = [
            v if isinstance(v, SymbolicExpression) else Literal(_value_=v)
            for v in children
        ]
        embedded_children = tuple(v._as_embeddable_child_(self) for v in children)
        for child in embedded_children:
            child._parent_ = self
        return embedded_children

    def _as_embeddable_child_(self, parent: SymbolicExpression) -> SymbolicExpression:
        """
        :param parent: The expression about to take this expression as a child.
        :return: The node that should be stored as ``parent``'s child, defaulting to this
            expression's compiled form. Subclasses whose compiled form is mutable and shared (for
            example :class:`~krrood.entity_query_language.query.query.Query`) override this to embed
            an immutable snapshot instead.
        """
        return self._expression_

    def _ensure_children_ids_are_cached_(self, *children: SymbolicExpression) -> None:
        """
        Ensure that the IDs of the provided children expressions, and all of their own
        descendants, are cached within the current expression.

        :param children: The children expressions to cache IDs for.
        """
        for child in children:
            if child._id_ not in self._expression_id_cache_:
                self._expression_id_cache_[child._id_] = child
                self._ensure_children_ids_are_cached_(*child._children_)

    def _process_result_(self, result: OperationResult) -> Any:
        """
        Map the result to the correct output data structure for user usage.

        It defaults to returning the bindings as a dictionary mapping variable objects
        to their values.

        :param result: The result to be mapped.
        :return: The mapped result.
        """
        if self._id_ in result:
            return result[self._id_]
        return self._unification_of_(result)

    def _unification_of_(self, result: OperationResult) -> UnificationDict:
        """
        :param result: The result to be mapped.
        :return: The value bindings of *result*, keyed by the expression that produced
            each one.

        A truth-valued expression records the truth of an operation rather than a value
        a caller selects, so its binding is not part of the unification.
        """
        bound_expressions = (
            (self._get_expression_by_id_(id_), value)
            for id_, value in result.bindings.items()
        )
        return UnificationDict(
            {
                expression: value
                for expression, value in bound_expressions
                if not isinstance(expression, TruthValuedExpression)
            }
        )

    def _result_is_false_(self, result: OperationResult) -> bool:
        """
        Read the truth of a result this expression produced.

        :param result: A result whose operand is this expression.
        :return: Whether the operation was not satisfied, derived from the binding this
            expression recorded: an empty collection or a falsy value is false. Having
            recorded no binding is no truth claim, and therefore not false.
        """
        if self._id_ not in result.bindings:
            return False
        binding = result.bindings[self._id_]
        return not (len(binding) > 0 if is_iterable(binding) else bool(binding))

    def _build_operation_result_with_truth_(
        self,
        truth: bool,
        bindings: Bindings,
        child_result: Optional[OperationResult] = None,
    ) -> OperationResult:
        """
        Build a result that records *truth* as this expression's own truth value.

        :param truth: The truth value this expression concluded.
        :param bindings: The bindings to record the truth value in. Copied, so that a
            sibling evaluation branch sharing them is unaffected.
        :param child_result: The result this one was derived from, if any.
        :return: The result carrying the recorded truth value.
        """
        bindings = copy(bindings)
        bindings[self._id_] = truth
        return OperationResult(bindings, self, child_result)

    def _evaluate_(
        self,
        sources: Optional[OperationResult] = None,
    ):
        """
        Wrapper for ``SymbolicExpression._evaluate__`` that manages evaluation context
        lifecycle.

        :param sources: The current OperationResult carrying bindings of variables, or
            None.
        :return: An iterator of OperationResult instances.
        """
        evaluation_context = get_evaluation_context()
        owns_an_evaluation_context = evaluation_context is None
        if owns_an_evaluation_context:
            from krrood.entity_query_language.evaluation import (
                create_default_evaluation_context,
            )

            evaluation_context = create_default_evaluation_context()
            context_token = set_evaluation_context(evaluation_context)
            evaluation_context.active_conditions_root.set_active_root_if_not_set(
                self._conditions_root_, has_condition=self._has_condition_
            )
        try:
            evaluation_context.on_evaluate_enter(expression=self, sources=sources)
            # Normalize sources: always work with an OperationResult
            previous_result = sources
            if sources is not None:
                bindings = copy(sources.bindings)
            else:
                bindings = {}
                sources = OperationResult({})  # empty sentinel for _evaluate__()
            if self._id_ in bindings:
                result = OperationResult(bindings, self, previous_result)
                evaluation_context.on_result_yielded(expression=self, result=result)
                yield result
            else:
                for result in map(
                    self._evaluate_conclusions_and_update_bindings_,
                    self._evaluate__(sources),
                ):
                    evaluation_context.on_result_yielded(expression=self, result=result)
                    yield result
        finally:
            evaluation_context.on_evaluate_exit(expression=self)
            if owns_an_evaluation_context:
                _evaluation_context_var.reset(context_token)

    def _evaluate_conclusions_and_update_bindings_(
        self, current_result: OperationResult
    ) -> OperationResult:
        """
        Update the bindings of the results by evaluating the conclusions using the
        received bindings.

        :param current_result: The current result of this expression.
        """
        # Only evaluate the conclusions at the active conditions root of the current evaluation
        # pass (i.e. after all conditions have been evaluated) and when the result truth value is
        # True. "Active" is an evaluation-scoped fact, not a structural one: a node reused as the
        # condition of more than one Filter has no single correct root, so this is resolved by
        # which evaluation is currently running (see ActiveConditionsRoot), not by the node's
        # construction history. When no evaluation context is active (this method is only ever
        # reached from inside _evaluate_'s own thread, but a caller may drive evaluation from a
        # thread that never had one set up — contextvars.ContextVar values do not propagate into a
        # plain threading.Thread), fall back to the structural check: it is the only signal left.
        evaluation_context = get_evaluation_context()
        if evaluation_context is not None:
            is_active_root = evaluation_context.active_conditions_root.is_active_root(
                self
            )
        else:
            is_active_root = self._conditions_root_ is self
        if not is_active_root:
            return current_result
        # Truth is only consulted where it decides something, since reading it can be
        # expensive (a bound predicate evaluates itself). The observers below apply the
        # same rule to their own work.
        if self._conclusions_ and current_result.is_true:
            for conclusion in self._conclusions_:
                current_result.bindings = next(
                    conclusion._evaluate_(current_result)
                ).bindings

        if evaluation_context is not None:
            evaluation_context.on_conclusions_processed(
                expression=self,
                result=current_result,
            )
        return current_result

    def conclusions_of_type(
        self, conclusion_type: Type[ConclusionType]
    ) -> List[ConclusionType]:
        """:return: The conclusions attached to this expression that are instances of *conclusion_type*."""
        return [
            conclusion
            for conclusion in self._conclusions_
            if isinstance(conclusion, conclusion_type)
        ]

    @abstractmethod
    def _evaluate__(
        self,
        sources: OperationResult,
    ) -> Iterator[OperationResult]:
        """
        Evaluate the symbolic expression and set the operands bindings in the result
        according to the evaluation logic of this expression.

        :param sources: The current OperationResult carrying bindings of variables.
        :return: An Iterator of OperationResult instances containing the bindings
            resulting from the evaluation of this expression.
        """
        pass

    def _has_parent_(self, expression: SymbolicExpression) -> bool:
        """
        :param expression: The expression to look for among this expression's parents.
        :return: Whether the given expression is one of this expression's parents.
        """
        return expression._id_ in [parent._id_ for parent in self._parents_]

    @property
    def _parent_(self) -> Optional[SymbolicExpression]:
        """
        :return: The parent symbolic expression of this expression.
        """
        return self._parent__

    @_parent_.setter
    def _parent_(self, value: Optional[SymbolicExpression]):
        """
        Set the parent symbolic expression of this expression.

        :param value: The new parent symbolic expression of this expression.
        """
        if value is self:
            return

        if value is None:
            if self._parent__ is not None:
                self._remove_parent_(self._parent__)
            return

        if not self._has_parent_(value):
            self._parents_.append(value)
            value._ensure_children_ids_are_cached_(self)

        if self._id_ not in [v._id_ for v in value._children_]:
            value._children_.append(self)

        # Keep the first structural parent as the primary one: a node reused as an operand
        # elsewhere in the DAG records the extra parent above but must not have its primary
        # parent hijacked. Structural re-parenting removes the old parent first (see
        # ``_replace_child_``), so the promotion in ``_remove_parent_`` re-establishes the primary.
        if self._parent__ is None:
            self._parent__ = value

    @property
    def _filter_condition_(self) -> Optional[SymbolicExpression]:
        """
        :return: The condition of the first ``Filter`` in this expression's graph, or
            ``None`` when no ``Filter`` gates it.
        """
        return next(
            (
                expression.condition
                for expression in self._all_expressions_
                if isinstance(expression, Filter)
            ),
            None,
        )

    @property
    def _conditions_root_(self) -> SymbolicExpression:
        """
        :return: The condition gating this expression, or the root of its graph when no
            ``Filter`` gates it.
        """
        filter_condition = self._filter_condition_
        if filter_condition is None:
            return self._root_
        return filter_condition

    @property
    def _has_condition_(self) -> bool:
        """
        :return: Whether a ``Filter`` gates this expression, i.e. whether
            :attr:`_conditions_root_` came from one rather than falling back to
            :attr:`_root_`.
        """
        return self._filter_condition_ is not None

    @property
    def _root_(self) -> SymbolicExpression:
        """
        :return: The root of the symbolic expression tree.
        """
        expression = self
        while expression._parent_ is not None:
            expression = expression._parent_
        return expression

    @property
    def _root_query_(self) -> Optional[Query]:
        """
        :return: The root query of the symbolic expression tree, or None if no query found.
        """
        from krrood.entity_query_language.query.query import Query

        for expression in self._all_expressions_:
            if isinstance(expression, Query):
                return expression
        return None

    @property
    @abstractmethod
    def _name_(self) -> str:
        """
        :return: The name of this symbolic expression.
        """
        pass

    @property
    def _all_expressions_(self) -> Iterator[SymbolicExpression]:
        """
        :return: All nodes in the symbolic expression tree.
        """
        yield self._root_
        yield from self._root_._descendants_

    def _evaluation_scoped_expression_index_(
        self,
    ) -> Optional[weakref.WeakValueDictionary[uuid.UUID, SymbolicExpression]]:
        """
        :return: An ``id -> node`` index of the whole tree, built once per evaluation, or ``None``
            when called outside an active evaluation.

        The tree is immutable while it is being evaluated, so the index is memoized on the evaluation
        context (keyed by the tree root) and reused for every id lookup instead of re-scanning. The
        values are held weakly so that a context outliving the evaluation cannot pin the tree or its
        variables' domains; during the evaluation every node is kept alive by the tree itself.
        """
        evaluation_context = get_evaluation_context()
        if evaluation_context is None:
            return None
        cache_key = self._root_._id_
        cache = evaluation_context.expression_index_cache
        if cache_key not in cache:
            cache[cache_key] = weakref.WeakValueDictionary(
                {expression._id_: expression for expression in self._all_expressions_}
            )
        return cache[cache_key]

    @property
    def _descendants_(self) -> Iterator[SymbolicExpression]:
        """
        :return: All descendants of this symbolic expression in children first, then depth-first by subtree order.

        A node reachable by more than one path (the expression graph is a DAG once conditions
        share variables, and rule trees share whole subtrees) is yielded only on its first visit.
        Without this a naive walk revisits shared subtrees once per path, which compounds into an
        exponential traversal on deep rule trees.

        Does not recurse into ``ResultQuantifier`` children so that inner query
        subtrees remain isolated from outer query traversals (that isolation is
        provided by :class:`~krrood.entity_query_language.query.query.Query`'s own
        ``_descendants_`` override).
        """
        yield from self._iter_descendants_(set())

    def _subtree_expressions_with_ids_(
        self, ids: Collection[uuid.UUID]
    ) -> Set[SymbolicExpression]:
        """
        :param ids: The identifiers to look for.
        :return: This expression and its descendants whose identifier is in *ids*.
        """
        return {
            expression
            for expression in itertools.chain([self], self._descendants_)
            if expression._id_ in ids
        }

    def _iter_descendants_(
        self, visited_ids: Set[uuid.UUID]
    ) -> Iterator[SymbolicExpression]:
        """
        Yield descendants once each, tracking already-seen ``_id_`` values across the
        recursion.

        :param visited_ids: The identifiers of nodes already yielded, shared across the
            whole walk.
        """
        fresh_children = [
            child for child in self._children_ if child._id_ not in visited_ids
        ]
        for child in fresh_children:
            visited_ids.add(child._id_)
        yield from fresh_children
        for child in fresh_children:
            yield from child._iter_descendants_(visited_ids)

    def _subtree_contains_(self, expression_type: Type[SymbolicExpression]) -> bool:
        """
        :return: Whether this expression's subtree contains a descendant of the given type.

        Whether a subtree contains a node of some type is a structural fact that is constant for the
        duration of an evaluation, so within an active evaluation the boolean answer is memoized on
        the evaluation context to avoid re-walking the subtree on every evaluation step (this is a
        hot path: every comparator evaluation checks for a ``The`` quantifier). Only the boolean is
        cached, never an expression, so the cache cannot outlive-pin the tree or its domains.
        """
        evaluation_context = get_evaluation_context()
        if evaluation_context is None:
            return self._compute_subtree_contains_(expression_type)
        cache_key = (self._id_, expression_type)
        cache = evaluation_context.subtree_containment_cache
        if cache_key not in cache:
            cache[cache_key] = self._compute_subtree_contains_(expression_type)
        return cache[cache_key]

    def _compute_subtree_contains_(
        self, expression_type: Type[SymbolicExpression]
    ) -> bool:
        """
        :return: Whether this expression's subtree contains a descendant of the given type, computed
            by walking the subtree.
        """
        return any(
            isinstance(descendant, expression_type) for descendant in self._descendants_
        )

    @classmethod
    def _current_parent_in_context_stack_(cls) -> Optional[SymbolicExpression]:
        """
        :return: The current parent symbolic expression in the enclosing context of the ``with`` statement. Used when
        making rule trees.
        """
        innermost_context = cls._innermost_rule_tree_context_()
        if innermost_context is None:
            return None
        return innermost_context.condition

    @classmethod
    def _innermost_rule_tree_context_(cls) -> Optional[RuleTreeContext]:
        """
        :return: The context of the innermost enclosing ``with`` statement, or ``None``
            outside any.
        """
        if not cls._symbolic_expression_stack_:
            return None
        return cls._symbolic_expression_stack_[-1]

    @classmethod
    def _rule_tree_context_anchored_on_(
        cls, condition: SymbolicExpression
    ) -> Optional[RuleTreeContext]:
        """
        :param condition: The condition an edit is about to be made relative to.
        :return: The context of the innermost enclosing ``with`` statement that anchors on
            the given condition, or ``None`` when no enclosing statement does.
        """
        return next(
            (
                context
                for context in reversed(cls._symbolic_expression_stack_)
                if context.condition._id_ == condition._id_
            ),
            None,
        )

    def _rule_tree_context_(self) -> RuleTreeContext:
        """
        :return: This expression paired with the parent edge the enclosing rule tree
            reaches it by.

        The enclosing block's own owning parent is the node a rule-tree edit just created
        to hold this expression, so it is this expression's owning parent too whenever
        this expression is the branch that edit introduced. Otherwise there is no asking
        branch to speak of and the structural parent stands in.
        """
        enclosing_context = self._innermost_rule_tree_context_()
        enclosing_parent = (
            enclosing_context.owning_parent if enclosing_context is not None else None
        )
        if enclosing_parent is None or not self._has_parent_(enclosing_parent):
            return RuleTreeContext(self, self._parent_)
        return RuleTreeContext(self, enclosing_parent)

    @property
    def _unique_variables_(self) -> Set[Variable]:
        """
        :return: Set of unique variables in this symbolic expression.
        """
        return make_set(self._all_variable_instances_)

    @property
    def _all_variable_instances_(self) -> List[Variable]:
        """
        Get the leaf instances of the symbolic expression.

        This is useful for accessing the leaves of the symbolic expression tree.
        """
        from krrood.entity_query_language.core.variable import Variable

        return [c for c in self._children_ if isinstance(c, Variable)]

    @property
    def _leaves_(self) -> Iterator[SymbolicExpression]:
        """
        Get the leaf instances of the symbolic expression.

        This is useful for accessing the leaves of the symbolic expression tree.
        """
        if len(self._children_) == 0:
            yield self
        for child in self._children_:
            yield from child._leaves_

    def _invert_(self):
        """
        Invert the symbolic expression.
        """
        from krrood.entity_query_language.operators.core_logical_operators import Not

        return Not(self)

    def __enter__(self) -> Self:
        """
        Enter a context where this symbolic expression is the current parent symbolic
        expression.

        This updates the current parent symbolic expression, the context stack and
        returns this expression.
        """
        SymbolicExpression._symbolic_expression_stack_.append(
            self._rule_tree_context_()
        )
        return self

    def __exit__(self, *args):
        """
        Exit the context and remove this symbolic expression from the context stack.
        """
        SymbolicExpression._symbolic_expression_stack_.pop()

    def __hash__(self):
        return hash(self._id_)

    def __repr__(self):
        return self._name_


@dataclass(eq=False, repr=False)
class UnaryExpression(SymbolicExpression, ABC):
    """
    A unary expression is a symbolic expression that takes a single argument (i.e., has
    a single child expression).

    The results of the child expression are the inputs to this expression.
    """

    _child_: SymbolicExpression
    """
    The child expression of this symbolic expression.
    """

    def __post_init__(self):
        super().__post_init__()
        self._child_ = self._update_children_(self._child_)[0]

    def _replace_child_field_(
        self, old_child: SymbolicExpression, new_child: SymbolicExpression
    ):
        if self._child_ is old_child:
            self._child_ = new_child

    @property
    def _name_(self) -> str:
        return self.__class__.__name__


@dataclass(eq=False, repr=False)
class MultiArityExpression(SymbolicExpression, ABC):
    """
    A multi-arity expression is a symbolic expression that takes multiple arguments
    (i.e., has multiple child expressions).
    """

    _operation_children_: Tuple[SymbolicExpression, ...] = field(default_factory=tuple)
    """
    The children expressions of this symbolic expression.
    """

    def __post_init__(self):
        super().__post_init__()
        self.update_children(*self._operation_children_)

    def _replace_child_field_(
        self, old_child: SymbolicExpression, new_child: SymbolicExpression
    ):
        old_child_index = self._operation_children_.index(old_child)
        self._operation_children_ = (
            self._operation_children_[:old_child_index]
            + (new_child,)
            + self._operation_children_[old_child_index + 1 :]
        )

    def update_children(self, *children: SymbolicExpression) -> None:
        self._operation_children_ = self._update_children_(*children)

    @property
    def _name_(self) -> str:
        return self.__class__.__name__


@dataclass(eq=False, repr=False)
class BinaryExpression(SymbolicExpression, ABC):
    """
    A base class for binary operators that can be used to combine symbolic expressions.
    """

    left: SymbolicExpression
    """
    The left operand of the binary operator.
    """

    right: SymbolicExpression
    """
    The right operand of the binary operator.
    """

    def __post_init__(self):
        super().__post_init__()
        self.left, self.right = self._update_children_(self.left, self.right)

    def _replace_child_field_(
        self, old_child: SymbolicExpression, new_child: SymbolicExpression
    ):
        if self.left is old_child:
            self.left = new_child
        elif self.right is old_child:
            self.right = new_child


@dataclass(eq=False, repr=False)
class TruthValuedExpression(SymbolicExpression, ABC):
    """
    An expression whose own binding is the truth of an operation rather than a value a
    caller can select.

    Truth and value share one binding per expression, so an expression that concludes a
    truth (a logical operator, a quantifier, a rule-tree selector) has no separate value
    to report, and one that produces a value (a variable, an aggregator, a query) makes
    no truth claim of its own outside a condition.
    """

    def _process_result_(self, result: OperationResult) -> UnificationDict:
        """
        Map the result to the bindings it carries.

        :param result: The result to be mapped.
        :return: Every binding of *result*, since this expression's own binding holds a
            truth value rather than a selectable one.
        """
        return self._unification_of_(result)


@dataclass(eq=False, repr=False)
class TruthValueOperator(SymbolicExpression, ABC):
    """
    An abstract superclass for operators that work with truth values of operations, thus
    requiring its children expressions to update their truth value when yielding
    results.
    """

    def _evaluate_child_as_condition_(
        self, child: SymbolicExpression, sources: Optional[OperationResult]
    ) -> Iterator[OperationResult]:
        """
        Evaluate ``child`` in a truth-value context.

        A child that bound a value is re-emitted as a result of its own, so that this
        evaluation is observed as a fresh one: the observers that record which
        expressions were evaluated and which conditions were satisfied fill those in only
        where they are still unset, and would otherwise carry a nested evaluation's
        record outwards.

        :param child: The child expression to evaluate in a truth-value context.
        :param sources: The current OperationResult carrying bindings, or None.
        :return: An iterator of the child's results.
        """
        evaluation_context = get_evaluation_context()
        if evaluation_context is not None:
            evaluation_context.truth_value_operator_children.record(child._id_)
        for result in child._evaluate_(sources):
            if result.has_value:
                yield result._as_fresh_observation_()
            else:
                yield result


@dataclass(eq=False, repr=False)
class DerivedExpression(SymbolicExpression, ABC):
    """
    A symbolic expression that has its results derived from another symbolic expression,
    and thus it's value is the value of the child expression.

    For example, filter expressions just filter the results of their children but they
    do not produce a new value of their own, thus they do not have a binding that
    belongs to them specifically in the result bindings dictionary.
    """

    @property
    @abstractmethod
    def _original_expression_(self) -> SymbolicExpression:
        """
        The original expression from which this expression is derived.
        """
        ...

    def _process_result_(self, result: OperationResult) -> Any:
        return self._original_expression_._process_result_(result)


@dataclass(eq=False, repr=False)
class Filter(DerivedExpression, TruthValueOperator, ABC):
    """
    Data source that evaluates the truth value for each data point according to a
    condition expression and filters out the data points that do not satisfy the
    condition.
    """

    @property
    def _original_expression_(self) -> SymbolicExpression:
        return self.condition

    @property
    @abstractmethod
    def condition(self) -> SymbolicExpression:
        """
        The conditions expression that generates the valid bindings that satisfy the
        constraints.
        """
        ...

    @property
    def _name_(self):
        return self.__class__.__name__


@dataclass
class OperationResult:
    """
    A data structure that carries information about the result of an operation in EQL.
    """

    bindings: Bindings
    """
    The bindings resulting from the operation, mapping variable IDs to their values.
    """

    operand: Optional[SymbolicExpression] = None
    """
    The operand that produced the result.
    """

    previous_operation_result: Optional[OperationResult] = None
    """
    The result of the operation that was evaluated before this one.
    """

    satisfied_condition_ids: Optional[OrderedSet[UUID]] = None
    """
    A set of UUIDs of condition expressions in the condition tree that were satisfied (truth value = True)
    during this evaluation. Populated at the conditions root after all conditions have been evaluated.
    Only set when the overall condition result is True.
    """

    evaluated_expression_ids: Optional[OrderedSet[UUID]] = None
    """
    A snapshot of the cumulative set of expression IDs evaluated so far during this
    evaluation pass, populated by the
    :class:`~krrood.entity_query_language.evaluation.EvaluationTracker` observer.

    Unlike
    ``satisfied_condition_ids``, this is populated on every yielded result, not only at the conditions root.
    """

    _is_false_: Optional[bool] = field(default=None, init=False, repr=False)
    """
    The truth read from the operand, once it has been read. See :attr:`is_false`.
    """

    @property
    def all_bindings(self) -> Bindings:
        """
        :return: All the bindings from all the evaluated operations until this one, including this one.
        Traverses the full previous_operation_result chain (linear traversal with cycle detection).
        """
        combined: Bindings = {}
        seen: set = set()

        def collect(node: Optional[OperationResult]) -> None:
            if node is None or id(node) in seen:
                return
            seen.add(id(node))
            collect(node.previous_operation_result)
            combined.update(node.bindings)  # shallower nodes (closer to self) win

        collect(self)
        return combined

    @property
    def has_value(self) -> bool:
        return self.operand is not None and self.operand._id_ in self.bindings

    def _as_fresh_observation_(self) -> OperationResult:
        """
        :return: An equivalent result that no observer has recorded anything on yet.

        The same operand and bindings, so the truth already read from them carries over
        rather than being read again, while the records the observers fill in start
        empty — this evaluation is a new one to them.
        """
        fresh_observation = OperationResult(
            self.bindings, self.operand, self.previous_operation_result
        )
        fresh_observation._is_false_ = self._is_false_
        return fresh_observation

    @property
    def is_false(self) -> bool:
        """
        Whether the operation was not satisfied, as read by the operand that produced
        this result.

        Read from the operand once and kept, since a bound value's truth can be
        arbitrarily expensive to obtain — a predicate reads as its own truth, and
        evaluating one runs whatever that predicate does — while truth is read many
        times over a single result as it flows through the operators that filter on it.

        ..note:: The operand answers this rather than the result reading its binding
            itself, because what a binding says about truth depends on what the
            expression records there: a truth for an operator, a value whose truthiness
            is its truth in a condition for a variable, and a selection saying nothing
            at all for a query.
        """
        if self._is_false_ is None:
            self._is_false_ = (
                self.operand is not None and self.operand._result_is_false_(self)
            )
        return self._is_false_

    @property
    def is_true(self) -> bool:
        return not self.is_false

    @property
    def value(self) -> Any:
        """
        The value of the operation result, retrieved from the bindings using the
        operand's ID.

        :raises: KeyError if the operand is not found in the bindings.
        """
        if self.operand is None:
            raise ValueError("Cannot get value: operand is None")
        return self.operand._process_result_(self)

    def __contains__(self, item):
        return item in self.bindings

    def __getitem__(self, item):
        return self.bindings[item]

    def __setitem__(self, key, value):
        self.bindings[key] = value

    def update(self, other: Bindings | OperationResult):
        if isinstance(other, OperationResult):
            self.bindings.update(other.bindings)
        else:
            self.bindings.update(other)
        return self

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return (
            self.bindings == other.bindings
            and self.is_true == other.is_true
            and self.operand == other.operand
            and self.previous_operation_result == other.previous_operation_result
        )


class UnificationDict(UserDict):
    """
    A dictionary which maps all expressions that are on a single variable to the
    original variable id.
    """

    def __getitem__(self, key: Selectable[T]) -> T:
        key = self._id_expression_map_[key._id_]
        return super().__getitem__(key)

    @cached_property
    def _id_expression_map_(self) -> Dict[uuid.UUID, Selectable[T]]:
        return {key._id_: key for key in self.data.keys()}


@dataclass(eq=False, repr=False)
class Selectable(SymbolicExpression, Generic[T], ABC):
    _var_: Selectable[T] = field(init=False, default=None)
    """
    A variable that is used if the child class to this class want to provide a variable
    to be tracked other than itself, this is specially useful for child classes that
    holds a variable instead of being a variable and want to delegate the variable
    behaviour to the variable it has instead.

    For example, this is the case for queries and derived references that operate on a
    single selected variable.
    """

    _type_: Type[T] = field(init=False, default=None)
    """
    The type of the selectable.
    """

    def __post_init__(self):
        super().__post_init__()
        if self._type_ is None:
            self._type_ = self._type__

    def _build_operation_result_(
        self,
        bindings: Bindings,
        child_result: Optional[OperationResult] = None,
    ) -> OperationResult:
        """
        Build an OperationResult instance for this binding.

        :param bindings: The bindings of the result, already carrying this expression's
            own value.
        :param child_result: The result of the child operation, if any.
        :return: The OperationResult instance.
        """
        return OperationResult(bindings, self, child_result)

    @cached_property
    def _type__(self):
        return (
            self._var_._type_
            if self._var_ is not None and self._var_ is not self
            else None
        )

    def _process_result_(self, result: OperationResult) -> T:
        """
        Map the result to the correct output data structure for user usage.

        :param result: The result to be mapped.
        :return: The mapped result.
        """
        return result[self._id_]

    @cached_property
    def _name_(self):
        if self._type_:
            return self._type_.__name__
        return self.__class__.__name__
