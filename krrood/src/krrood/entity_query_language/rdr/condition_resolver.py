"""
Auto-condition resolution for EQL-RDR using backward inference.

When a rule fires with the wrong conclusion and the expert would normally be asked for
differentiating conditions, a :class:`ConditionResolver` can attempt to derive the
condition automatically from the rule tree's backward-inference knowledge — so the expert
is only consulted when no automatic resolution is possible.

The default built-in strategy, composed as a :class:`ChainConditionResolver`:

* :class:`TargetKnowledgeResolver` — find a condition already known for the target
  conclusion that is True for the new case and False for the corner case.

:class:`CornerCaseKnowledgeResolver` is retained for advanced use but is **not** in the
default chain (see its class docstring for why).

All strategies are gated on ``corner_case is not None`` (refinement branch only).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from typing_extensions import TYPE_CHECKING, Any, List, Optional

from krrood.entity_query_language.factories import not_
from krrood.entity_query_language.rdr.backward_inference import (
    ConclusionKnowledge,
    GuardCondition,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.core.base_expressions import SymbolicExpression
    from krrood.entity_query_language.core.mapped_variable import CanBehaveLikeAVariable


class ResolutionMode(Enum):
    """Controls how an auto-resolved condition is applied when fitting a case.

    See :meth:`~krrood.entity_query_language.rdr.single_class.EQLSingleClassRDR.fit_case`.
    :attr:`SILENT` preserves the original behaviour (no expert prompt); :attr:`HINT`
    shows the suggestion to the expert who may accept or overwrite it.
    """

    SILENT = "silent"
    """Auto-resolved condition is inserted directly without consulting the expert."""
    HINT = "hint"
    """Auto-resolved condition is shown to the expert as a pre-seeded suggestion."""


class ResolutionSource(Enum):
    """Identifies which resolution strategy produced a :class:`ResolvedCondition`."""

    TARGET_KNOWLEDGE = "target_knowledge"
    CORNER_CASE_KNOWLEDGE = "corner_case_knowledge"


@dataclass(frozen=True)
class ResolvedCondition:
    """An automatically derived condition expression and its provenance."""

    expression: "SymbolicExpression"
    """The EQL condition expression to insert as the new rule's condition."""
    source: ResolutionSource
    """The resolution strategy that produced this condition."""


class ConditionResolver(ABC):
    """Strategy for automatically deriving a differentiating condition from the rule tree.

    Implementations receive the full context needed to attempt resolution. Returning
    ``None`` signals that this resolver cannot find a condition; the caller will try
    the next resolver or fall back to the expert.
    """

    @abstractmethod
    def resolve(
        self,
        case: Any,
        case_variable: "CanBehaveLikeAVariable",
        target: Any,
        current: Any,
        corner_case: Any,
        target_knowledge: "ConclusionKnowledge",
        current_knowledge: "ConclusionKnowledge",
    ) -> Optional[ResolvedCondition]:
        """Attempt to auto-derive a differentiating condition.

        :param case: The new case being fit (must be classified as ``target``).
        :param case_variable: The RDR's shared EQL variable.
        :param target: The correct conclusion for ``case``.
        :param current: The wrong conclusion currently returned by the firing rule.
        :param corner_case: The case that triggered the currently-firing rule's creation.
        :param target_knowledge: Backward-inference knowledge for ``target``.
        :param current_knowledge: Backward-inference knowledge for ``current``.
        :return: A :class:`ResolvedCondition`, or ``None`` if this resolver cannot resolve.
        """


def _materialize(guard: "GuardCondition") -> "SymbolicExpression":
    """Apply negation to produce the EQL expression used as a new rule's condition.

    A ``negated=True`` guard fires when its expression is False, so materializing it
    for use as a new rule's condition requires wrapping it with ``not_``.

    :return: ``not_(guard.expression)`` when ``guard.negated`` is ``True``, otherwise
        ``guard.expression``.
    """
    return not_(guard.expression) if guard.negated else guard.expression


class TargetKnowledgeResolver(ConditionResolver):
    """Primary strategy resolver: use backward inference on the target conclusion.

    Searches the sufficient condition sets known for ``target`` and returns the first
    guard that is True for the new case and False for the corner case — guaranteeing
    it discriminates between them.
    """

    def resolve(
        self,
        case: Any,
        case_variable: "CanBehaveLikeAVariable",
        target: Any,
        current: Any,
        corner_case: Any,
        target_knowledge: "ConclusionKnowledge",
        current_knowledge: "ConclusionKnowledge",
    ) -> Optional[ResolvedCondition]:
        """Search ``target_knowledge`` for a guard that discriminates ``case`` from ``corner_case``.

        Returns the first guard that holds for ``case`` and does not hold for ``corner_case``,
        materialized as a :class:`ResolvedCondition` tagged
        :attr:`ResolutionSource.TARGET_KNOWLEDGE`.

        :return: A :class:`ResolvedCondition`, or ``None`` if no discriminating guard is found.
        """
        for sufficient_condition_set in target_knowledge.sufficient_condition_sets:
            for guard in sufficient_condition_set.conditions:
                if guard.holds_for(case_variable, case) and not guard.holds_for(
                    case_variable, corner_case
                ):
                    return ResolvedCondition(
                        _materialize(guard), ResolutionSource.TARGET_KNOWLEDGE
                    )
        return None


class CornerCaseKnowledgeResolver(ConditionResolver):
    """**Experimental** resolver: use backward inference on the wrong (current) conclusion.

    Searches the sufficient condition sets known for ``current`` and finds a guard that
    is True for the corner case. The negation of that guard is then checked: if it is
    True for the new case, the negated condition discriminates correctly.

    .. warning::

        This resolver is intentionally **excluded from the default chain** returned by
        :meth:`ChainConditionResolver.backward_inference_default`. It inspects the
        *current* (wrong) branch — the very guards that caused the misclassification —
        and negates them. This produces shallow, unstable refinements that cause
        oscillation in the convergence loop (e.g. fish/amphibian ping-pong).

        Principled RDR discrimination requires a condition that comes from knowledge
        about the *target* conclusion, not from inverting the wrong rule. Use this
        resolver only in carefully controlled custom chains where you understand its
        limitations.
    """

    def resolve(
        self,
        case: Any,
        case_variable: "CanBehaveLikeAVariable",
        target: Any,
        current: Any,
        corner_case: Any,
        target_knowledge: "ConclusionKnowledge",
        current_knowledge: "ConclusionKnowledge",
    ) -> Optional[ResolvedCondition]:
        """Search ``current_knowledge`` for a guard whose negation discriminates ``case`` from ``corner_case``.

        Finds a guard that holds for ``corner_case`` and whose negation holds for ``case``,
        then returns the negated condition tagged :attr:`ResolutionSource.CORNER_CASE_KNOWLEDGE`.

        :return: A :class:`ResolvedCondition`, or ``None`` if no negated guard discriminates.
        """
        for sufficient_condition_set in current_knowledge.sufficient_condition_sets:
            for guard in sufficient_condition_set.conditions:
                if guard.holds_for(case_variable, corner_case):
                    negated = GuardCondition(guard.expression, not guard.negated)
                    if negated.holds_for(case_variable, case):
                        return ResolvedCondition(
                            _materialize(negated),
                            ResolutionSource.CORNER_CASE_KNOWLEDGE,
                        )
        return None


@dataclass
class ChainConditionResolver(ConditionResolver):
    """A Chain-of-Responsibility that tries each resolver in order, returning the first match."""

    resolvers: List[ConditionResolver] = field(default_factory=list)
    """Ordered list of :class:`ConditionResolver` strategies to try, in priority order."""

    def resolve(
        self,
        case: Any,
        case_variable: "CanBehaveLikeAVariable",
        target: Any,
        current: Any,
        corner_case: Any,
        target_knowledge: "ConclusionKnowledge",
        current_knowledge: "ConclusionKnowledge",
    ) -> Optional[ResolvedCondition]:
        """Try each resolver in :attr:`resolvers` in order, returning the first non-``None`` result.

        :return: The first :class:`ResolvedCondition` produced by a resolver, or ``None``
            if every resolver returns ``None``.
        """
        for resolver in self.resolvers:
            result = resolver.resolve(
                case,
                case_variable,
                target,
                current,
                corner_case,
                target_knowledge,
                current_knowledge,
            )
            if result is not None:
                return result
        return None

    @classmethod
    def backward_inference_default(cls) -> "ChainConditionResolver":
        """Return the standard chain: a single :class:`TargetKnowledgeResolver`.

        :class:`CornerCaseKnowledgeResolver` is intentionally excluded because it
        inspects the *current* (wrong) branch and can produce unstable refinements that
        cause the convergence loop to oscillate. See its class docstring for details.

        :return: A :class:`ChainConditionResolver` with the target-knowledge resolver.
        """
        return cls([TargetKnowledgeResolver()])


__all__ = [
    "ChainConditionResolver",
    "ConditionResolver",
    "CornerCaseKnowledgeResolver",
    "ResolvedCondition",
    "ResolutionMode",
    "ResolutionSource",
    "TargetKnowledgeResolver",
]
