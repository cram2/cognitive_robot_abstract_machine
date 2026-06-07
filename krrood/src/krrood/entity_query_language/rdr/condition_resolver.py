"""
Auto-condition resolution for EQL-RDR using backward inference.

When a rule fires with the wrong conclusion and the expert would normally be asked for
differentiating conditions, a :class:`ConditionResolver` can attempt to derive the
condition automatically from the rule tree's backward-inference knowledge — so the expert
is only consulted when no automatic resolution is possible.

The default built-in strategy, composed as a :class:`ChainConditionResolver`:

* :class:`TargetKnowledgeResolver` — find a condition already known for the target
  conclusion that is True for the new case and False for the corner case.
* :class:`CornerCaseKnowledgeResolver` — search non-active paths to the wrong conclusion
  for a positive condition that is True for the new case and False for the corner case.

All strategies are gated on ``corner_case is not None`` (refinement branch only).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from typing_extensions import TYPE_CHECKING, Any, List, Optional

from krrood.entity_query_language.factories import not_
from krrood.entity_query_language.rdr.backward_inference import ConclusionKnowledge

if TYPE_CHECKING:
    from krrood.entity_query_language.core.base_expressions import SymbolicExpression
    from krrood.entity_query_language.core.mapped_variable import CanBehaveLikeAVariable
    from krrood.entity_query_language.rdr.backward_inference import (
        GuardCondition,
        SufficientConditionSet,
    )


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
        case_variable: CanBehaveLikeAVariable,
        target: Any,
        current: Any,
        corner_case: Any,
        target_knowledge: ConclusionKnowledge,
        current_knowledge: ConclusionKnowledge,
        firing_anchor: Optional[SymbolicExpression] = None,
    ) -> Optional[ResolvedCondition]:
        """Attempt to auto-derive a differentiating condition.

        :param case: The new case being fit (must be classified as ``target``).
        :param case_variable: The RDR's shared EQL variable.
        :param target: The correct conclusion for ``case``.
        :param current: The wrong conclusion currently returned by the firing rule.
        :param corner_case: The case that triggered the currently-firing rule's creation.
        :param target_knowledge: Backward-inference knowledge for ``target``.
        :param current_knowledge: Backward-inference knowledge for ``current``.
        :param firing_anchor: The condition expression of the rule that fired; used by
            resolvers to identify the active path without re-evaluating the rule tree.
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
        case_variable: CanBehaveLikeAVariable,
        target: Any,
        current: Any,
        corner_case: Any,
        target_knowledge: ConclusionKnowledge,
        current_knowledge: ConclusionKnowledge,
        firing_anchor: Optional[SymbolicExpression] = None,
    ) -> Optional[ResolvedCondition]:
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
    """Fallback resolver: search non-active paths to the wrong conclusion for a positive condition.

    For each sufficient condition set of the wrong (current) conclusion that is **not** the
    active path (the path that caused the misclassification), searches for a guard that:

    * holds for the new case — so the new exception rule fires for it, and
    * does **not** hold for the corner case — so the original rule is left undisturbed.

    The matching guard is returned without negation, producing a stable positive condition
    grounded in a different characterisation of the wrong conclusion.

    The active path is identified via the ``firing_anchor`` expression using an identity check
    on guard expressions — no re-evaluation of the rule tree is needed.  When ``firing_anchor``
    is ``None``, all paths are considered (safe degradation).
    """

    def _active_path(
        self,
        firing_anchor: Optional[SymbolicExpression],
        current_knowledge: ConclusionKnowledge,
    ) -> Optional[SufficientConditionSet]:
        """Return the :class:`SufficientConditionSet` that contains ``firing_anchor``
        as a positive (non-negated) guard expression.

        Searches every guard in every sufficient condition set using an identity check
        on ``guard.expression``.  The ``not guard.negated`` clause excludes paths where
        the same expression node appears as a negated ancestor guard in a sibling path —
        only the sufficient condition set in which the anchor fires positively is considered active.

        :return: The matching :class:`SufficientConditionSet`, or ``None`` if not found.
        """
        if firing_anchor is None:
            return None
        return next(
            (
                sufficient_condition_set
                for sufficient_condition_set in current_knowledge.sufficient_condition_sets
                if any(
                    guard.expression is firing_anchor and not guard.negated
                    for guard in sufficient_condition_set.conditions
                )
            ),
            None,
        )

    def resolve(
        self,
        case: Any,
        case_variable: CanBehaveLikeAVariable,
        target: Any,
        current: Any,
        corner_case: Any,
        target_knowledge: ConclusionKnowledge,
        current_knowledge: ConclusionKnowledge,
        firing_anchor: Optional[SymbolicExpression] = None,
    ) -> Optional[ResolvedCondition]:
        """Search non-active paths in ``current_knowledge`` for a positive discriminating guard.

        Skips the active path (identified via ``firing_anchor``) and returns the first guard
        from any other path that holds for ``case`` but not for ``corner_case``, tagged
        :attr:`ResolutionSource.CORNER_CASE_KNOWLEDGE`.  The guard is materialized via
        :func:`_materialize`; for the typical non-negated guard this is a no-op, but a
        negated guard in a non-active path will be wrapped with ``not_()`` if encountered.

        :return: A :class:`ResolvedCondition`, or ``None`` if no discriminating guard is found.
        """
        active = self._active_path(firing_anchor, current_knowledge)
        for sufficient_condition_set in current_knowledge.sufficient_condition_sets:
            if sufficient_condition_set is active:
                continue
            for guard in sufficient_condition_set.conditions:
                if guard.holds_for(case_variable, case) and not guard.holds_for(
                    case_variable, corner_case
                ):
                    return ResolvedCondition(
                        _materialize(guard), ResolutionSource.CORNER_CASE_KNOWLEDGE
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
        case_variable: CanBehaveLikeAVariable,
        target: Any,
        current: Any,
        corner_case: Any,
        target_knowledge: ConclusionKnowledge,
        current_knowledge: ConclusionKnowledge,
        firing_anchor: Optional[SymbolicExpression] = None,
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
                firing_anchor,
            )
            if result is not None:
                return result
        return None

    @classmethod
    def backward_inference_default(cls) -> ChainConditionResolver:
        """Return the standard two-resolver chain.

        :class:`TargetKnowledgeResolver` runs first, searching the target conclusion's
        backward-inference paths for a positive discriminating guard.
        :class:`CornerCaseKnowledgeResolver` runs second, searching non-active paths to the
        wrong conclusion for a direct positive condition; it uses the firing anchor to skip the
        active path efficiently without re-evaluating the rule tree.

        :return: A :class:`ChainConditionResolver` with both resolvers in priority order.
        """
        return cls([TargetKnowledgeResolver(), CornerCaseKnowledgeResolver()])


__all__ = [
    "ChainConditionResolver",
    "ConditionResolver",
    "CornerCaseKnowledgeResolver",
    "ResolvedCondition",
    "ResolutionMode",
    "ResolutionSource",
    "TargetKnowledgeResolver",
]
