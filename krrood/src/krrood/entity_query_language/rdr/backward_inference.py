"""
Backward inference for EQL-RDR rule trees.

Given a target conclusion value (e.g. ``Species.molusc``), traverse the rule-tree DAG
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

from krrood.entity_query_language.core.base_expressions import OperationResult
from krrood.entity_query_language.rules.conclusion_selector import (
    Alternative,
    ConclusionSelector,
    Next,
    Refinement,
)
from krrood.entity_query_language.rdr.utils import _conclusions_of, _extract_value

if TYPE_CHECKING:
    from krrood.entity_query_language.core.base_expressions import SymbolicExpression
    from krrood.entity_query_language.core.mapped_variable import (
        CanBehaveLikeAVariable,
    )
    from krrood.entity_query_language.rules.conclusion import Add as AddConclusion


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuardCondition:
    """A path guard that must hold for a rule to fire.

    ``negated=True`` means the rule fires only when ``expression`` evaluates to False.
    The expression is the original live EQL node from the rule tree — use
    :meth:`holds_for` to test it against a concrete case.
    """

    expression: "SymbolicExpression"
    negated: bool = False

    def holds_for(
        self,
        shared_variable: "CanBehaveLikeAVariable",
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
        results = list(self.expression.evaluate())
        truth = any(
            r.is_true if isinstance(r, OperationResult) else bool(r) for r in results
        )
        return not truth if self.negated else truth


@dataclass(frozen=True)
class SufficientConditionSet:
    """One rule path's complete conditions for a specific conclusion value to fire.

    The conditions are stored as :class:`GuardCondition` tuples. Use
    :meth:`evaluate_against` to check them against a concrete case without mutating the
    original rule tree.
    """

    conditions: Tuple[GuardCondition, ...]

    def evaluate_against(
        self,
        shared_variable: "CanBehaveLikeAVariable",
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
class ConclusionKnowledge:
    """The rule tree's complete backward-inference knowledge about one conclusion value."""

    conclusion_value: Any
    """The queried conclusion value (e.g. ``Species.molusc``)."""
    sufficient_condition_sets: Tuple[SufficientConditionSet, ...]
    """Every rule path that can produce this conclusion, as sufficient condition sets."""

    def is_satisfiable(self) -> bool:
        """:return: ``True`` when at least one rule path exists for this value."""
        return bool(self.sufficient_condition_sets)


# ---------------------------------------------------------------------------
# Tree traversal
# ---------------------------------------------------------------------------


@dataclass
class _RulePath:
    """An internal value object for one discovered rule path during traversal."""

    conditions: Tuple[GuardCondition, ...]
    """Guard conditions accumulated along the path to these add nodes."""
    add_nodes: Tuple[AddConclusion, ...]
    """Conclusion nodes at the leaf of this rule path."""


def _flatten_guard(
    expr: "SymbolicExpression",
    negated: bool,
) -> List[GuardCondition]:
    """Decompose a ConclusionSelector guard into leaf GuardConditions.

    RDR control-flow operators (Alternative, Refinement) encode branch choice
    semantics. When they appear as guard expressions they should be decomposed
    into their constituent leaf conditions for readability — the guard's
    semantic meaning is just the sibling's immediate condition truth, not the
    entire subtree.

    Semantics:
    * ``NOT(Alternative(A, B))`` → ``NOT(A), NOT(B)``  (De Morgan on OR)
    * ``Refinement(A, B)`` → ``A``  (truth equals left's truth for bool check)
    * ``NOT(Refinement(A, B))`` → ``NOT(A)``
    """
    if isinstance(expr, Alternative) and negated:
        return _flatten_guard(expr.left, True) + _flatten_guard(expr.right, True)
    if isinstance(expr, Refinement):
        return _flatten_guard(expr.left, negated)
    return [GuardCondition(expr, negated)]


def _collect_rule_paths(
    node: "SymbolicExpression",
    guard: List[GuardCondition],
) -> Iterator[_RulePath]:
    """Recursively walk the selector DAG, yielding a path for every leaf rule.

    The *guard* list accumulates path conditions as selectors are descended:
    * ``Alternative(left, right)``: left fires directly; right fires only when
      ``NOT(left)``.
    * ``Refinement(left, right)``: left fires when ``NOT(right)`` (refinement doesn't
      override); right fires when ``left`` (parent fired — positive guard).
    * ``Next``: each child is a separate disjunct (same depth, no cross-guards).

    Guards that are ConclusionSelector nodes are flattened via
    :func:`_flatten_guard` — a single ``NOT(Alternative(A, B))`` becomes the
    two guards ``NOT(A), NOT(B)``, and ``Refinement(A, B)`` reduces to ``A``.
    This keeps the guard list semantically precise and human-readable.
    """
    if isinstance(node, Refinement):
        yield from _collect_rule_paths(
            node.left,
            guard + _flatten_guard(node.right, negated=True),
        )
        yield from _collect_rule_paths(
            node.right,
            guard + _flatten_guard(node.left, negated=False),
        )
    elif isinstance(node, Alternative):
        yield from _collect_rule_paths(node.left, guard)
        yield from _collect_rule_paths(
            node.right,
            guard + _flatten_guard(node.left, negated=True),
        )
    elif isinstance(node, Next):
        for child in node._operation_children_:
            yield from _collect_rule_paths(child, guard)
    else:
        add_nodes = _conclusions_of(node)
        if add_nodes:
            yield _RulePath(
                conditions=tuple(guard + [GuardCondition(node, negated=False)]),
                add_nodes=tuple(add_nodes),
            )


# ---------------------------------------------------------------------------
# Indexed cache
# ---------------------------------------------------------------------------


def _build_full_index(
    conditions_root: "SymbolicExpression",
) -> Dict[Any, ConclusionKnowledge]:
    """One full traversal of the rule tree; buckets every conclusion value once."""
    buckets: Dict[Any, List[SufficientConditionSet]] = defaultdict(list)
    for path in _collect_rule_paths(conditions_root, []):
        seen: Set[Any] = set()
        for add_node in path.add_nodes:
            value = _extract_value(add_node)
            if value not in seen:
                buckets[value].append(SufficientConditionSet(path.conditions))
                seen.add(value)
    return {v: ConclusionKnowledge(v, tuple(sets)) for v, sets in buckets.items()}


@dataclass
class BackwardInferenceIndex:
    """Lazy cache of the rule tree's backward-inference results.

    On first query after construction (or after :meth:`invalidate`), one full
    traversal builds the entire index for all conclusion values in a single pass.
    Subsequent queries for any value are O(1) dict lookups.
    """

    _cache: Optional[Dict[Any, ConclusionKnowledge]] = field(default=None, init=False)

    def invalidate(self) -> None:
        """:return: None. Marks the cache stale so the next query rebuilds."""
        self._cache = None

    def query(
        self,
        conditions_root: Optional["SymbolicExpression"],
        conclusion_value: Any,
    ) -> ConclusionKnowledge:
        """:return: The backward-inference knowledge for *conclusion_value*."""
        if conditions_root is None:
            return ConclusionKnowledge(conclusion_value, ())
        if self._cache is None:
            self._cache = _build_full_index(conditions_root)
        return self._cache.get(
            conclusion_value,
            ConclusionKnowledge(conclusion_value, ()),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


__all__ = [
    "BackwardInferenceIndex",
    "ConclusionKnowledge",
    "SufficientConditionSet",
    "what_do_we_know_about",
]


def what_do_we_know_about(
    conditions_root: Optional["SymbolicExpression"],
    conclusion_value: Any,
) -> ConclusionKnowledge:
    """Inspect the rule tree for every rule path that produces *conclusion_value*.

    Each discovered path yields one :class:`SufficientConditionSet` containing the
    complete set of conditions (including guards from ``Refinement`` and
    ``Alternative`` selectors) that must be true for the path to fire.

    When no path exists, returns a :class:`ConclusionKnowledge` with
    ``is_satisfiable() == False``.

    :param conditions_root: The root of the rule tree's condition DAG,
        or ``None`` for an empty tree.
    :param conclusion_value: The target value to search for.
    :return: The backward-inference knowledge.
    """
    return BackwardInferenceIndex().query(conditions_root, conclusion_value)
