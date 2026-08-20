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

from krrood.entity_query_language.operators.core_logical_operators import Not
from krrood.entity_query_language.rdr.branch_semantics import SelectorBranchSemantics
from krrood.entity_query_language.rdr.guard_condition import GuardCondition
from krrood.entity_query_language.rules.conclusion import Add
from krrood.entity_query_language.rules.conclusion_selector import ConclusionSelector

if TYPE_CHECKING:
    from krrood.entity_query_language.core.base_expressions import SymbolicExpression
    from krrood.entity_query_language.core.variable import Variable


# %%
# Data structures


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
    """Decompose an expression into leaf-level branch-choice predicates.

    This is not tree traversal — it is predicate decomposition. It answers the question:
    "when this expression appears as a path guard (i.e. a competing sibling branch), what
    are the minimal leaf conditions that capture whether that sibling's branch was taken?"

    Each selector's own rule lives on its
    :class:`~krrood.entity_query_language.rdr.branch_semantics.SelectorBranchSemantics`;
    handled here are only the two cases that are not selector-specific — pushing a negation
    through a wrapped selector, and bottoming out on a leaf predicate.

    The result is always leaf-level :class:`GuardCondition` objects, never selectors, so
    guards remain human-readable and directly evaluable.

    :param expression: The expression to decompose into leaf guards.
    :param negated: Whether the guard polarity is negated.
    :return: The flat list of leaf :class:`GuardCondition` objects.
    """
    semantics = SelectorBranchSemantics.most_specific_for(expression)
    if semantics is not None:
        return semantics.sibling_guards(expression, negated, _leaf_guards)
    if isinstance(expression, Not) and isinstance(expression._child_, ConclusionSelector):
        return _leaf_guards(expression._child_, not negated)
    return [GuardCondition(expression, negated)]


def _collect_rule_paths(
        node: SymbolicExpression,
        guard: List[GuardCondition],
) -> Iterator[_RulePath]:
    """Recursively walk the selector DAG, yielding a path for every leaf rule.

    Which children a selector is descended into, and what entering each one contributes to
    the accumulated *guard*, is the selector's own
    :class:`~krrood.entity_query_language.rdr.branch_semantics.SelectorBranchSemantics`.
    A node with no such semantics is a leaf rule: it guards itself positively and
    terminates the path.

    :param node: The rule-tree node to walk.
    :param guard: The guard conditions accumulated on the way to *node*.
    :return: One :class:`_RulePath` per leaf rule reachable from *node*.
    """
    semantics = SelectorBranchSemantics.most_specific_for(node)
    if semantics is None:
        add_nodes = node.conclusions_of_type(Add)
        if add_nodes:
            yield _RulePath(
                conditions=tuple(guard + [GuardCondition(node, negated=False)]),
                add_nodes=tuple(add_nodes),
            )
        return

    for branch in semantics.branches(node, _leaf_guards):
        yield from _collect_rule_paths(branch.child_expression, guard + list(branch.entry_guards))


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
