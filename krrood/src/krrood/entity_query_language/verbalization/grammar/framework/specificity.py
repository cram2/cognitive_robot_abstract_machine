from __future__ import annotations

from abc import ABC

from typing_extensions import Any, List, Optional, Type

from krrood.entity_query_language.verbalization.exceptions import AmbiguousRuleError
from krrood.patterns.specificity_ranking import (
    concrete_subclasses,
    mro_depth,
    sole_maximum,
)


class SpecificityRule(ABC):
    """
    A guarded alternative selected by specificity: the shared base of the small rule
    registries (restriction folding, restriction-subject resolution, navigation forms).

    An alternative is a **subclass** that implements an ``applies(...)`` guard (its
    signature is the subfamily's concern) and a payload method; alternatives
    *self-register* as the family's concrete subclasses and are ranked by class specificity —
    a more-derived alternative (one that subclasses another and refines its guard) outranks the
    alternative it refines. Alternatives that are not in a subclass relationship must have
    mutually exclusive guards, so at most one applies (there is no other ordering between them).

    This mirrors ``select`` for :class:`PhraseRule`: precedence comes from the class hierarchy
    (genuine subsumption) or from disjoint guards, never from a hand-assigned number.

    Reference: production-rule selection; the systemic-functional "most delicate system wins"
    principle (:cite:t:`halliday2014functional`).
    """

    @classmethod
    def alternatives(cls) -> List[Type[SpecificityRule]]:
        """:return: The concrete alternative subclasses of this family (transitive; abstract
        family bases are excluded).

        This is the family-facing view over :func:`concrete_subclasses`: bound to one base, it is the
        candidate set :meth:`most_applicable` ranks — here the three ``RankingForm`` templates.

        >>> from krrood.entity_query_language.verbalization.grammar.query.ranking import RankingForm
        >>> sorted(rule.__name__ for rule in RankingForm.alternatives())
        ['AttributeRankedByForm', 'AttributeSuperlativeForm', 'LeadingRankForm']
        """
        return concrete_subclasses(cls)

    @classmethod
    def most_applicable(cls, *args: Any) -> Optional[Type[SpecificityRule]]:
        """
        *args* are forwarded verbatim to each alternative's ``applies`` classmethod, so
        the subfamily fixes that signature (e.g. ``(item, subject)``).

        :return: The most-specific alternative whose ``applies(*args)`` holds, or ``None``.

        >>> from krrood.entity_query_language.verbalization.grammar.query.ranking import (
        ...     RankingForm, RankingRequest)
        >>> from krrood.entity_query_language.verbalization.grammar.query.planner import (
        ...     RankingPlan, SortDirection, RankingKeyRelation)
        >>> plan = RankingPlan(limit_number=3, direction=SortDirection.DESCENDING,
        ...     relation=RankingKeyRelation.ATTRIBUTE, order_key=None)
        >>> RankingForm.most_applicable(RankingRequest(plan=plan)).__name__
        'AttributeRankedByForm'
        """
        applicable = [alt for alt in cls.alternatives() if alt.applies(*args)]
        return sole_maximum(
            applicable,
            key=mro_depth,
            collision_error=lambda tied: AmbiguousRuleError(
                subject=args, candidates=tied
            ),
        )
