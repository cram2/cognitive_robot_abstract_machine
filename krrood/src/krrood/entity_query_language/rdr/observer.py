"""
Observer that reads RDR conclusions out of an EQL evaluation.

Classification in the EQL-native RDR is plain EQL evaluation of the rule-tree query.
This module provides the aspect that listens to that evaluation and extracts the
inferred conclusion for the underspecified attribute, without the rule tree (or the
core evaluation methods) knowing anything about RDR.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any, List, Optional
from uuid import UUID

from ordered_set import OrderedSet

from krrood.entity_query_language.core.base_expressions import (
    OperationResult,
    SymbolicExpression,
)
from krrood.entity_query_language.core.mapped_variable import CanBehaveLikeAVariable
from krrood.entity_query_language.core.variable import Literal
from krrood.entity_query_language.enums import EvaluationContextKey
from krrood.entity_query_language.rdr.utils import UNSET
from krrood.entity_query_language.rules.conclusion import Add
from krrood.entity_query_language.evaluation import (
    EvaluationContext,
    EvaluationObserver,
    EvaluationTracker,
    SatisfiedConditionTracker,
    set_evaluation_context,
)


@dataclass
class FiredConclusion:
    """A single conclusion observed during evaluation of the rule tree."""

    value: Any
    """The inferred value bound to the conclusion variable (e.g. ``Species.mammal``)."""
    conditions_root: SymbolicExpression
    """The conditions-root expression at which the conclusion was processed."""
    result: OperationResult
    """The full result, carrying ``bindings`` and ``satisfied_condition_ids``."""
    anchor: Optional[SymbolicExpression] = None
    """
    The condition node of the rule that produced this conclusion (the firing ``Add``'s
    parent). This is the insertion point for a refinement that overrides this conclusion.
    """
    add_node: Optional[Add] = None
    """The ``Add`` conclusion node that fired."""


class ConclusionObserver(EvaluationObserver):
    """Collects the conclusion bound to a target variable during EQL evaluation.

    Hooks :meth:`on_conclusions_processed`, which fires at the conditions root once
    conclusions (``Add`` nodes) have updated the bindings and the result is true. The
    inferred value is whatever the target variable is bound to at that point.
    """

    def __init__(self, conclusion_variable: CanBehaveLikeAVariable) -> None:
        self.conclusion_variable = conclusion_variable
        self.conclusion_id = conclusion_variable._id_
        self.fired: List[FiredConclusion] = []

    def reset(self) -> None:
        """Clear any captured conclusions, ready for a fresh evaluation."""
        self.fired = []

    def on_conclusions_processed(
        self, expression: SymbolicExpression, result: OperationResult
    ) -> None:
        if self.conclusion_id not in result.bindings:
            return
        value = result.bindings[self.conclusion_id]
        add_node = self._find_firing_add(expression, value)
        anchor = add_node._parent_ if add_node is not None else expression
        self.fired.append(
            FiredConclusion(
                value=value,
                conditions_root=expression,
                result=result,
                anchor=anchor,
                add_node=add_node,
            )
        )

    def _find_firing_add(
        self, conditions_root: SymbolicExpression, value: Any
    ) -> Optional[Add]:
        """
        :return: The ``Add`` (among the conclusions that propagated to ``conditions_root``)
            whose value matches the inferred ``value`` — i.e. the conclusion that won.
        """
        for conclusion in conditions_root._conclusions_:
            if not isinstance(conclusion, Add):
                continue
            target = conclusion.right
            target_value = target._value_ if isinstance(target, Literal) else target
            if target_value == value:
                return conclusion
        return None

    @property
    def conclusion(self) -> Any:
        """The single inferred value, or ``UNSET`` if no rule fired.

        Single-class RDR conclusions are mutually exclusive, so all captured
        conclusions for one case carry the same value; we return the last one.
        """
        return self.fired[-1].value if self.fired else UNSET

    @property
    def distinct_conclusions(self) -> List[Any]:
        """The distinct inferred values observed (order-preserving)."""
        seen: List[Any] = []
        for f in self.fired:
            if f.value not in seen:
                seen.append(f.value)
        return seen


def classify_case(
    rule_tree_query: SymbolicExpression,
    case_variable: CanBehaveLikeAVariable,
    conclusion_variable: CanBehaveLikeAVariable,
    case: Any,
) -> ConclusionObserver:
    """
    Evaluate ``rule_tree_query`` for a single ``case`` and return the observer that
    captured the conclusion(s).

    The case is bound by re-targeting ``case_variable``'s domain to ``[case]`` so the
    shared rule-tree DAG is evaluated against exactly this case. A
    :class:`ConclusionObserver` is installed (alongside the default trackers, so
    ``satisfied_condition_ids`` is populated for later insertion-point logic).

    :param rule_tree_query: The root EQL query of the rule tree.
    :param case_variable: The shared variable the rule tree ranges over.
    :param conclusion_variable: The (underspecified) attribute the rules conclude.
    :param case: The single instance to classify.
    :return: The :class:`ConclusionObserver` holding the captured conclusion(s).
    """
    case_variable._update_domain_([case])
    observer = ConclusionObserver(conclusion_variable)
    set_evaluation_context(
        EvaluationContext(
            observers=[observer, EvaluationTracker(), SatisfiedConditionTracker()]
        )
    )
    try:
        list(rule_tree_query.evaluate())
    finally:
        set_evaluation_context(None)
    return observer


@dataclass
class ClassificationTrace:
    """A read-model of one classification, for explaining/visualizing the rule tree.

    Bundles the rule-tree root with the evaluation observers' id-sets so a renderer can
    colour each rule (fired / evaluated / skipped) and anchor an elided view on the rule
    that actually fired — without re-running evaluation or touching the core pipeline.
    """

    rule_tree_root: Optional[SymbolicExpression]
    """The root of the rule tree's condition DAG (``None`` for an empty tree)."""
    satisfied_condition_ids: Optional[OrderedSet[UUID]]
    """Ids of condition nodes whose truth value was True (the fired rules)."""
    evaluated_expression_ids: Optional[OrderedSet[UUID]]
    """Ids of every expression that was evaluated (fired ∪ evaluated-not-fired)."""
    firing_anchor: Optional[SymbolicExpression] = None
    """The condition node of the rule that produced the winning conclusion."""
    conclusion: Any = UNSET
    """The inferred conclusion (``UNSET`` if no rule fired)."""

    @property
    def firing_anchor_id(self) -> Optional[UUID]:
        """:return: The id of :attr:`firing_anchor`, or ``None`` if no rule fired."""
        return self.firing_anchor._id_ if self.firing_anchor is not None else None

    @classmethod
    def from_observer(
        cls,
        observer: ConclusionObserver,
        rule_tree_root: Optional[SymbolicExpression],
        evaluated_ids: Optional[OrderedSet[UUID]],
    ) -> "ClassificationTrace":
        """Build a trace from a finished :class:`ConclusionObserver` and the evaluated set."""
        fired = observer.fired[-1] if observer.fired else None
        return cls(
            rule_tree_root=rule_tree_root,
            satisfied_condition_ids=(
                fired.result.satisfied_condition_ids if fired is not None else None
            ),
            evaluated_expression_ids=evaluated_ids,
            firing_anchor=fired.anchor if fired is not None else None,
            conclusion=observer.conclusion,
        )


def trace_case(
    rule_tree_query: SymbolicExpression,
    case_variable: CanBehaveLikeAVariable,
    conclusion_variable: CanBehaveLikeAVariable,
    case: Any,
    rule_tree_root: Optional[SymbolicExpression],
) -> ClassificationTrace:
    """
    Evaluate ``rule_tree_query`` for one ``case`` and capture a :class:`ClassificationTrace`.

    Like :func:`classify_case` but also retains the cumulative *evaluated* id-set from the
    evaluation context (so even branches the evaluation short-circuited can be coloured
    grey), and packages it together with the satisfied set and firing anchor.

    :param rule_tree_query: The root EQL query of the rule tree.
    :param case_variable: The shared variable the rule tree ranges over.
    :param conclusion_variable: The (underspecified) attribute the rules conclude.
    :param case: The single instance to classify.
    :param rule_tree_root: The conditions-root to render (usually the query's).
    :return: The :class:`ClassificationTrace` for this case.
    """
    case_variable._update_domain_([case])
    observer = ConclusionObserver(conclusion_variable)
    ctx = EvaluationContext(
        observers=[observer, EvaluationTracker(), SatisfiedConditionTracker()]
    )
    set_evaluation_context(ctx)
    try:
        list(rule_tree_query.evaluate())
    finally:
        set_evaluation_context(None)
    evaluated = ctx.data.get(EvaluationContextKey.EVALUATED_IDS_KEY)
    return ClassificationTrace.from_observer(observer, rule_tree_root, evaluated)
