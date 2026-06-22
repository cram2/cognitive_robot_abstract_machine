from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

from typing_extensions import List, Optional

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.expression_structure import walk_chain
from krrood.entity_query_language.verbalization import morphology
from krrood.entity_query_language.verbalization.fragments.base import (
    Fragment,
    PhraseFragment,
    RoleFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import Number
from krrood.entity_query_language.verbalization.grammar.framework.specificity import (
    SpecificityRule,
)
from krrood.entity_query_language.verbalization.grammar.query.planner import (
    RankingDirection,
    RankingKeyRelation,
    RankingPlan,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Articles,
    Prepositions,
    RankingWords,
)


@dataclass(frozen=True)
class RankingRequest:
    """A query's ranking together with the selection type label — the input a ranking form reads."""

    plan: RankingPlan
    """The ``limit`` (+ ordering) decomposition."""


@dataclass(frozen=True)
class RankingSurface:
    """The placed pieces of a ranking phrase, for the selection noun phrase to carry.

    The selection noun is built around these: ``"the"`` + *pre_head* + head (in *number*) +
    *modifiers* — e.g. ``"the"`` + ``"top three"`` + ``"Employees"`` + ``"by salary"``.
    """

    pre_head: Optional[Fragment]
    """The qualifier between the determiner and the head (*"first two"* / *"top three"* /
    *"highest"*), or ``None`` (the attribute-superlative form carries it as a modifier instead)."""

    number: Number
    """The head's grammatical number — ``SINGULAR`` for *n = 1*, ``PLURAL`` for *n > 1*."""

    modifiers: List[Fragment]
    """Post-nominal modifiers — *"with the highest salary"* / *"by salary"* — or empty."""


def _quality(direction: RankingDirection, n: int) -> RankingWords:
    """:return: The leading quality word for a (direction, n): *first* (no order), *highest*/*top*
    (descending, n=1/n>1), *lowest*/*bottom* (ascending, n=1/n>1)."""
    if direction is RankingDirection.DESCENDING:
        return RankingWords.HIGHEST if n == 1 else RankingWords.TOP
    if direction is RankingDirection.ASCENDING:
        return RankingWords.LOWEST if n == 1 else RankingWords.BOTTOM
    return RankingWords.FIRST


def _cardinal(n: int) -> Fragment:
    """:return: The cardinal-word fragment for *n* (``3`` → *"three"*)."""
    return WordFragment(text=morphology.cardinal(n))


def _key_attribute(order_key: SymbolicExpression) -> Fragment:
    """:return: The order key's terminal attribute as a bare attribute word (*"salary"*, not the
    verbose *"the salary of the Employee"*)."""
    chain, _ = walk_chain(order_key)
    attribute = chain[-1]
    return RoleFragment.for_attribute(
        attribute._owner_class_, attribute._attribute_name_
    )


class RankingForm(SpecificityRule):
    """
    One surface template for a query's ``limit`` ranking phrase — recognise the (direction, count,
    key-relation) situation and produce the :class:`RankingSurface` the selection noun carries.

    The registry is *total*: :class:`LeadingRankForm` is the unguarded base every specific form
    refines, so :meth:`~SpecificityRule.most_applicable` always returns a form. Adding a template is
    a new subclass; nothing else changes (open/closed). Mirrors ``grammar/conditions/forms.py``.
    """

    @classmethod
    @abstractmethod
    def applies(cls, request: RankingRequest) -> bool:
        """:return: ``True`` when this form renders *request*."""

    @classmethod
    @abstractmethod
    def render(cls, request: RankingRequest) -> RankingSurface:
        """:return: *request* rendered into the selection's ranking pieces."""


class LeadingRankForm(RankingForm):
    """The default: the quality (+ count) leads the noun, with no key named — *"the first two
    Robots"*, *"the highest Robot"*, *"the top three Robots"*. Covers no-ordering, ordering by the
    selection itself, and the unrelated-key fallback (key suppressed)."""

    @classmethod
    def applies(cls, request: RankingRequest) -> bool:
        return True

    @classmethod
    def render(cls, request: RankingRequest) -> RankingSurface:
        n = request.plan.n
        quality = _quality(request.plan.direction, n).as_fragment()
        pre_head = quality if n == 1 else PhraseFragment(parts=[quality, _cardinal(n)])
        return RankingSurface(pre_head=pre_head, number=Number.of(n > 1), modifiers=[])


class AttributeSuperlativeForm(LeadingRankForm):
    """Ordering by an attribute of the selection, *n = 1* → the superlative attaches to the key:
    *"the Employee with the highest salary"* / *"… with the lowest salary"*."""

    @classmethod
    def applies(cls, request: RankingRequest) -> bool:
        plan = request.plan
        return plan.relation is RankingKeyRelation.ATTRIBUTE and plan.n == 1

    @classmethod
    def render(cls, request: RankingRequest) -> RankingSurface:
        superlative = (
            RankingWords.LOWEST
            if request.plan.direction is RankingDirection.ASCENDING
            else RankingWords.HIGHEST
        )
        modifier = PhraseFragment(
            parts=[
                Prepositions.WITH.as_fragment(),
                Articles.THE.as_fragment(),
                superlative.as_fragment(),
                _key_attribute(request.plan.order_key),
            ]
        )
        return RankingSurface(
            pre_head=None, number=Number.SINGULAR, modifiers=[modifier]
        )


class AttributeRankedByForm(LeadingRankForm):
    """Ordering by an attribute of the selection, *n > 1* → *"the top three Employees by salary"* /
    *"the bottom three Employees by salary"*."""

    @classmethod
    def applies(cls, request: RankingRequest) -> bool:
        plan = request.plan
        return plan.relation is RankingKeyRelation.ATTRIBUTE and plan.n > 1

    @classmethod
    def render(cls, request: RankingRequest) -> RankingSurface:
        plan = request.plan
        quality = (
            RankingWords.BOTTOM
            if plan.direction is RankingDirection.ASCENDING
            else RankingWords.TOP
        )
        pre_head = PhraseFragment(parts=[quality.as_fragment(), _cardinal(plan.n)])
        modifier = PhraseFragment(
            parts=[RankingWords.BY.as_fragment(), _key_attribute(plan.order_key)]
        )
        return RankingSurface(
            pre_head=pre_head, number=Number.PLURAL, modifiers=[modifier]
        )


def ranking_surface(request: RankingRequest) -> RankingSurface:
    """
    Render a query's ranking into the selection's pieces — the single entry the query assembler
    uses. The form is chosen by the registry (most-specific applicable; the leading fallback
    guarantees a match), so the caller never inspects the ranking's shape.

    :param request: The ranking and selection-type label.
    :return: The ranking pieces for the selection noun phrase.
    """
    return RankingForm.most_applicable(request).render(request)
