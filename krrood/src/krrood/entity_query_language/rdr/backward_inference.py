"""
Backward inference for EQL-RDR rule trees.

Given a target conclusion value (e.g. ``Species.molusc``), traverse the rule-tree
(The rules form a tree, but if you look at the level of the used expressions, it is
a directed acyclic graph)
backwards to enumerate every rule path that could produce it. Each path accumulates
*guard conditions* from the ``Refinement``/``Alternative``/``Next`` selectors and wraps
the result as a :class:`SufficientConditionSet`. The full answer is the disjunction of all
such sets (a DNF formula).

This is backward chaining — goal-directed reasoning that works backwards through the
rule tree, the inverse of forward evaluation / classification.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property

from typing_extensions import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)

from krrood.entity_query_language.factories import not_
from krrood.entity_query_language.operators.core_logical_operators import Not
from krrood.entity_query_language.rules.conclusion import Add
from krrood.entity_query_language.rules.conclusion_selector import (
    Alternative,
    ConclusionSelector,
    Next,
    Refinement,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.core.base_expressions import SymbolicExpression
    from krrood.entity_query_language.core.variable import Variable


# %%
# Data structures


@dataclass(frozen=True)
class GuardCondition:
    """A condition that must be satisfied for a rule to be applied.

    Each guard is one leaf-level predicate extracted from the rule tree's
    conclusion selectors.  ``negated=True`` means the rule applies only when the
    condition is False (i.e. the expression must evaluate to False).

    ``expression`` is always a leaf-level EQL node (e.g. a ``Comparator``),
    never a ``ConclusionSelector`` — ``_leaf_guards`` decomposes selectors
    before they reach ``GuardCondition``.
    """

    original_expression: SymbolicExpression
    """The leaf-level EQL predicate to evaluate (e.g. a ``Comparator``)."""
    negated: bool = False
    """When ``True`` the guard is satisfied only if :attr:`expression` is False.

    Polarity is carried here rather than applied to :attr:`expression`, because
    negating an expression reparents it and :attr:`expression` belongs to the
    live rule tree.
    """

    def holds_for(
            self,
            shared_variable: Variable,
            case: Any,
    ) -> bool:
        """Evaluate this guard against *case* bound to *shared_variable*.

        Respects :attr:`negated`: a negated guard must evaluate to ``False`` for
        the result to be ``True``.

        :param shared_variable: The EQL variable the conditions range over.
        :param case: The concrete case object to evaluate against.
        :return: ``True`` if the guard is satisfied.
        """
        shared_variable._update_domain_([case])
        # A leaf predicate yields its own bound boolean per case; a Not() has no
        # id-keyed payload of its own, so it yields the full binding row when it holds
        # and nothing when it does not. Both read correctly through bool().
        truth = any(bool(result) for result in self.original_expression.evaluate())
        return not truth if self.negated else truth

    @cached_property
    def as_expression(self) -> SymbolicExpression:
        """
        Produce the EQL expression where the negation is applied if it is a negated guard.

        A negated guard is wrapped with ``not_`` so the produced condition expression is satisfied
        when the guard's expression is False.
        """
        return not_(self.original_expression) if self.negated else self.original_expression


@dataclass(frozen=True)
class SufficientConditionSet:
    """One rule path's complete conditions to conclude a specific conclusion value.

    The conditions are stored as :class:`GuardCondition` tuples. Use
    :meth:`evaluate_against` to check them against a concrete case without mutating the
    original rule tree.
    """

    conditions: Tuple[GuardCondition, ...]
    """
    The conditions that all must hold to conclude a specific conclusion value.
    """

    def evaluate_against(
            self,
            shared_variable: Variable,
            case: Any,
    ) -> bool:
        """Evaluate every condition against *case* bound to *shared_variable*.

        Delegates per-guard evaluation to :meth:`GuardCondition.holds_for`.
        All conditions must hold for the result to be ``True``.

        :param shared_variable: The EQL variable the conditions range over
            (the rule tree's ``case_variable``).
        :param case: The concrete case object to evaluate against.
        :return: ``True`` if every guard condition is satisfied.
        """
        return all(guard.holds_for(shared_variable, case) for guard in self.conditions)


@dataclass(frozen=True)
class ConclusionSufficientConditionSets:
    """
    The rule tree's complete backward-inference knowledge about one conclusion value. In other words, it is the
    known sets of sufficient conditions any of which if satisfied implies the conclusion value
    """

    conclusion_value: Any
    """The queried conclusion value (e.g. ``Species.molusc``)."""
    sufficient_condition_sets: Tuple[SufficientConditionSet, ...]
    """Every rule path that can produce this conclusion, as sufficient condition sets."""

    def is_satisfiable(self) -> bool:
        """:return: ``True`` when at least one rule path exists for this value."""
        return bool(self.sufficient_condition_sets)


# %%
# Tree traversal


@dataclass
class _RulePath:
    """An internal value object for one discovered rule path during traversal."""

    conditions: Tuple[GuardCondition, ...]
    """Guard conditions accumulated along the path to these add nodes."""
    add_nodes: Tuple[Add, ...]
    """Conclusion nodes at the leaf of this rule path."""


def _leaf_guards(
        expression: SymbolicExpression,
        negated: bool,
) -> List[GuardCondition]:
    """Decompose a ConclusionSelector into leaf-level branch-choice predicates.

    This is NOT tree traversal — it is predicate decomposition.  It answers the
    question: "when this ``ConclusionSelector`` appears as a path guard (i.e. a
    competing sibling branch), what are the minimal leaf conditions that capture
    whether that sibling's branch was taken?"

    The result is always leaf-level ``GuardCondition`` objects (never
    ``ConclusionSelectors``), so guards remain human-readable and semantically
    precise.

    Decomposition rules (each explained in terms of "the sibling's branch was
    taken"):

    * ``Alternative(A, B)`` — the sibling's branch was taken if A OR B passed.
      When negated: NOT(A) AND NOT(B) (De Morgan).
      Both children contribute because Alternative is a simple OR.

    * ``Refinement(A, B)`` — the sibling's refinement branch was taken if A
      passed.  B (the parent default fallback) is a separate rule subtree,
      not a condition on the refinement being taken.  It is ignored.
      When negated: NOT(A).

    * ``Next(...)`` — each child is an independent disjunct at the same depth.
      Propagate the predicate to each child independently.

    * ``Not(ConclusionSelector)`` — push negation inward so that
      NOT(Refinement(A, B)) → NOT(A), and
      NOT(Alternative(A, B)) → NOT(A) AND NOT(B).

    :param expression: The expression to decompose into leaf guards.
    :param negated: Whether the guard polarity is negated.
    :return: The flat list of leaf :class:`GuardCondition` objects.
    """
    if isinstance(expression, Alternative):
        if negated:
            # NOT(A OR B) == NOT(A) AND NOT(B)
            return _leaf_guards(expression.left, True) + _leaf_guards(expression.right, True)
        # A OR (NOT(A) AND B) — decomposed to both sides as leaf conditions
        return _leaf_guards(expression.left, False) + _leaf_guards(expression.right, False)
    if isinstance(expression, Refinement):
        return _leaf_guards(expression.left, negated)
    if isinstance(expression, Next):
        result: List[GuardCondition] = []
        for child in expression._operation_children_:
            result.extend(_leaf_guards(child, negated))
        return result
    if isinstance(expression, Not) and isinstance(expression._child_, ConclusionSelector):
        # Push negation through the selector — refines NOT(Refinement), NOT(Alternative), NOT(Next)
        return _leaf_guards(expression._child_, not negated)
    return [GuardCondition(expression, negated)]


def _collect_rule_paths(
        node: SymbolicExpression,
        guard: List[GuardCondition],
) -> Iterator[_RulePath]:
    """Recursively walk the selector DAG, yielding a path for every leaf rule.

    The *guard* list accumulates path conditions as selectors are descended:
    * ``Alternative(left, right)``: left applies directly; right applies only when
      ``NOT(left)``.
    * ``Refinement(left, right)``: left applies when ``NOT(right)`` (refinement doesn't
      override); right applies when ``left`` (parent applied — positive guard).
    * ``Next``: each child is a separate disjunct (same depth, no cross-guards).

    Guards that are ConclusionSelector nodes are decomposed via
    :func:`_leaf_guards` — a single ``NOT(Alternative(A, B))`` becomes the
    two guards ``NOT(A), NOT(B)``, and ``Refinement(A, B)`` reduces to ``A``.
    This keeps the guard list semantically precise and human-readable.
    """
    if isinstance(node, Refinement):
        yield from _collect_rule_paths(
            node.left,
            guard + _leaf_guards(node.right, negated=True),
        )
        yield from _collect_rule_paths(
            node.right,
            guard + _leaf_guards(node.left, negated=False),
        )
    elif isinstance(node, Alternative):
        yield from _collect_rule_paths(node.left, guard)
        yield from _collect_rule_paths(
            node.right,
            guard + _leaf_guards(node.left, negated=True),
        )
    elif isinstance(node, Next):
        for child in node._operation_children_:
            yield from _collect_rule_paths(child, guard)
    else:
        add_nodes = node.conclusions_of_type(Add)
        if add_nodes:
            yield _RulePath(
                conditions=tuple(guard + [GuardCondition(node, negated=False)]),
                add_nodes=tuple(add_nodes),
            )


# %%
# Indexed cache


def _index_conclusions_by_value(
        conditions_root: SymbolicExpression,
) -> Dict[Any, ConclusionSufficientConditionSets]:
    """One full traversal of the rule tree; buckets every conclusion value once.

    :param conditions_root: The root of the rule tree's condition DAG.
    :return: A dict mapping each conclusion value to its :class:`ConclusionKnowledge`.
    """
    buckets: Dict[Any, List[SufficientConditionSet]] = defaultdict(list)
    for path in _collect_rule_paths(conditions_root, []):
        seen: Set[Any] = set()
        for add_node in path.add_nodes:
            value = add_node.unwrapped_value
            if value not in seen:
                buckets[value].append(SufficientConditionSet(path.conditions))
                seen.add(value)
    return {v: ConclusionSufficientConditionSets(v, tuple(sets)) for v, sets in buckets.items()}


@dataclass
class BackwardInferenceIndex:
    """Lazy cache of the rule tree's backward-inference results.

    On first query after construction (or after :meth:`invalidate`), one full
    traversal builds the entire index for all conclusion values in a single pass.
    Subsequent queries for any value are O(1) dict lookups.
    """

    _cache: Optional[Dict[Any, ConclusionSufficientConditionSets]] = field(default=None, init=False)
    """
    The full index of all conclusion values, or ``None`` if the index is not built.
    """

    def invalidate(self) -> None:
        """:return: None. Marks the cache stale so the next query rebuilds."""
        self._cache = None

    def query(
            self,
            expression: Optional[SymbolicExpression],
            conclusion_value: Any,
    ) -> ConclusionSufficientConditionSets:
        """
        :param expression: Any node belonging to the rule tree, or ``None`` for an empty
            tree. The condition DAG's root is resolved from it via ``_conditions_root_``.
        :param conclusion_value: The target value to search for.
        :return: The backward-inference knowledge for *conclusion_value*.
        """
        if expression is None:
            return ConclusionSufficientConditionSets(conclusion_value, ())
        if self._cache is None:
            self._cache = _index_conclusions_by_value(expression._conditions_root_)
        return self._cache.get(
            conclusion_value,
            ConclusionSufficientConditionSets(conclusion_value, ()),
        )


# %%
# Public API


def get_conclusion_sufficient_conditions_from_a_rule_tree(
        expression: Optional[SymbolicExpression],
        conclusion_value: Any,
) -> ConclusionSufficientConditionSets:
    """Inspect the rule tree for every rule path that produces *conclusion_value*.

    Each discovered path yields one :class:`SufficientConditionSet` containing the
    complete set of conditions (including guards from ``Refinement`` and
    ``Alternative`` selectors) that must be true for the path to be traversed.

    When no path exists, returns a :class:`ConclusionKnowledge` with
    ``is_satisfiable() == False``.

    :param expression: Any node belonging to the rule tree, or ``None`` for an empty
        tree. The condition DAG's root is resolved from it via ``_conditions_root_``.
    :param conclusion_value: The target value to search for.
    :return: The backward-inference knowledge.
    """
    return BackwardInferenceIndex().query(expression, conclusion_value)
