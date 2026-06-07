"""
Auto-condition resolution for EQL-RDR using backward inference.

When a rule fires with the wrong conclusion and the expert would normally be asked for
differentiating conditions, a :class:`ConditionResolver` can attempt to derive the
condition automatically from the rule tree's backward-inference knowledge — so the expert
is only consulted when no automatic resolution is possible.

The two built-in strategies, composed as a :class:`ChainConditionResolver`:

* :class:`TargetKnowledgeResolver` — Phase 1: find a condition already known for the
  target conclusion that is True for the new case and False for the corner case.
* :class:`CornerCaseKnowledgeResolver` — Phase 2: find a condition known for the wrong
  (current) conclusion that is True for the corner case, then negate it; if the negation
  is True for the new case, use it.

Both strategies are gated on ``corner_case is not None`` (refinement branch only) and
are fully silent — no expert prompt is produced when a condition is auto-resolved.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from typing_extensions import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from krrood.entity_query_language.core.base_expressions import SymbolicExpression
    from krrood.entity_query_language.core.mapped_variable import CanBehaveLikeAVariable
    from krrood.entity_query_language.rdr.backward_inference import (
        ConclusionKnowledge,
        GuardCondition,
    )


class ResolutionSource(Enum):
    """Identifies which resolution strategy produced a :class:`ResolvedCondition`."""

    TARGET_KNOWLEDGE = "target_knowledge"
    CORNER_CASE_KNOWLEDGE = "corner_case_knowledge"


@dataclass(frozen=True)
class ResolvedCondition:
    """An automatically derived condition expression and its provenance.

    :param expression: The EQL condition expression to insert as the new rule's condition.
    :param source: Which resolution strategy produced this condition.
    """

    expression: "SymbolicExpression"
    source: ResolutionSource


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
    """Return the EQL expression for *guard*, applying negation when needed.

    A ``negated=True`` guard fires when its expression is False, so materializing it
    for use as a new rule's condition requires wrapping it with ``not_``.
    """
    from krrood.entity_query_language.factories import not_

    return not_(guard.expression) if guard.negated else guard.expression


class TargetKnowledgeResolver(ConditionResolver):
    """Phase-1 resolver: use backward inference on the target conclusion.

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
        for scs in target_knowledge.sufficient_condition_sets:
            for guard in scs.conditions:
                if guard.holds_for(case_variable, case) and not guard.holds_for(
                    case_variable, corner_case
                ):
                    return ResolvedCondition(
                        _materialize(guard), ResolutionSource.TARGET_KNOWLEDGE
                    )
        return None


class CornerCaseKnowledgeResolver(ConditionResolver):
    """Phase-2 resolver: use backward inference on the wrong (current) conclusion.

    Searches the sufficient condition sets known for ``current`` and finds a guard that
    is True for the corner case. The negation of that guard is then checked: if it is
    True for the new case, the negated condition discriminates correctly.
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
        from krrood.entity_query_language.rdr.backward_inference import GuardCondition

        for scs in current_knowledge.sufficient_condition_sets:
            for guard in scs.conditions:
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
    """A Chain-of-Responsibility that tries each resolver in order, returning the first match.

    :param resolvers: Ordered list of :class:`ConditionResolver` strategies to try.
    """

    resolvers: List[ConditionResolver] = field(default_factory=list)

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
        """Return the standard two-phase chain: TargetKnowledge → CornerCaseKnowledge.

        :return: A :class:`ChainConditionResolver` with both built-in resolvers.
        """
        return cls([TargetKnowledgeResolver(), CornerCaseKnowledgeResolver()])


__all__ = [
    "ChainConditionResolver",
    "ConditionResolver",
    "CornerCaseKnowledgeResolver",
    "ResolvedCondition",
    "ResolutionSource",
    "TargetKnowledgeResolver",
]
