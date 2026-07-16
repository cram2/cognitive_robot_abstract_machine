from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

from typing_extensions import List, Optional

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.expression_structure import walk_chain
from krrood.entity_query_language.verbalization import morphology
from krrood.entity_query_language.verbalization.fragments.base import (
    VerbalizationFragment,
    PhraseFragment,
    RoleFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    GrammaticalNumber,
)
from krrood.entity_query_language.verbalization.grammar.framework.specificity import (
    SpecificityRule,
)
from krrood.entity_query_language.verbalization.grammar.query.planner import (
    SortDirection,
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
    """
    A query's ranking together with the selection type label — the input a ranking form
    reads.
    """

    plan: RankingPlan
    """
    The ``limit`` (+ ordering) decomposition.
    """


@dataclass(frozen=True)
class RankingSurface:
    """
    The placed pieces of a ranking phrase, for the selection noun phrase to carry.

    The selection noun is built around these: ``"the"`` + *pre_head* + head (in *number*) +
    *modifiers* — e.g. ``"the"`` + ``"top three"`` + ``"Employees"`` + ``"by salary"``.
    """

    pre_head: Optional[VerbalizationFragment]
    """
    The qualifier between the determiner and the head (*"first two"* / *"top three"* /
    *"highest"*), or ``None`` (the attribute-superlative form carries it as a modifier
    instead).
    """

    number: GrammaticalNumber
    """The head's grammatical number — ``SINGULAR`` for *n = 1*, ``PLURAL`` for *n > 1*."""

    modifiers: List[VerbalizationFragment]
    """
    Post-nominal modifiers — *"with the highest salary"* / *"by salary"* — or empty.
    """


def _quality(direction: SortDirection, n: int) -> RankingWords:
    """:return: The leading quality word for a (direction, n): *first* (no order), *highest*/*top*
    (descending, n=1/n>1), *lowest*/*bottom* (ascending, n=1/n>1).

    >>> _quality(SortDirection.DESCENDING, 1).name
    'HIGHEST'
    >>> _quality(SortDirection.DESCENDING, 3).name
    'TOP'
    """
    if direction is SortDirection.DESCENDING:
        return RankingWords.HIGHEST if n == 1 else RankingWords.TOP
    if direction is SortDirection.ASCENDING:
        return RankingWords.LOWEST if n == 1 else RankingWords.BOTTOM
    return RankingWords.FIRST


def _cardinal(n: int) -> VerbalizationFragment:
    """:return: The cardinal-word fragment for *n* (``3`` → *"three"*).

    >>> _cardinal(3).text
    'three'
    """
    return WordFragment(text=morphology.cardinal(n))


def _key_attribute(order_key: SymbolicExpression) -> VerbalizationFragment:
    """:return: The order key's terminal attribute as a bare attribute word (*"salary"*, not the
    verbose *"the salary of the Employee"*).

    >>> _key_attribute(variable(Employee, []).salary).text
    'salary'
    """
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
    a new subclass; nothing else changes (open/closed). Mirrors ``grammar/conditions/placement.py``.

    >>> employee = variable(Employee, [])
    >>> verbalize_expression(entity(employee).ordered_by(employee.salary, descending=True).limit(3))
    'Find the top three Employees by salary'
    """

    @classmethod
    @abstractmethod
    def applies(cls, request: RankingRequest) -> bool:
        """:return: ``True`` when this form renders *request*.

        This is the per-form guard that decides whether the ranking takes this form's surface; the
        winning form for the class example is what makes it read *"the top three Employees by
        salary"*.
        """

    @classmethod
    @abstractmethod
    def render(cls, request: RankingRequest) -> RankingSurface:
        """:return: *request* rendered into the selection's ranking pieces.

        It produces the ranking pieces the selection noun carries — here the *"top three"* qualifier
        and the *"by salary"* modifier that surround the head of the class example.
        """


class LeadingRankForm(RankingForm):
    """The default: the quality (+ count) leads the noun, with no key named — *"the first two
    Robots"*, *"the highest Robot"*, *"the top three Robots"*. Covers no-ordering, ordering by the
    selection itself, and the unrelated-key fallback (key suppressed).

    >>> robot = variable(Robot, [])
    >>> verbalize_expression(entity(robot).limit(2))
    'Find the first two Robots'
    """

    @classmethod
    def applies(cls, request: RankingRequest) -> bool:
        """:return: Always ``True`` — the unguarded base every specific form refines.

        Being the always-matching base, this is the decision that routes any ranking with no more
        specific form to the leading-quality surface — ordering by the selection itself names no key,
        so a descending ``limit(3)`` on the selection reads the bare *"the top three Robots"*.
        """
        return True

    @classmethod
    def render(cls, request: RankingRequest) -> RankingSurface:
        """:return: The quality (+ count) leading the noun, with no key named.

        It emits the leading qualifier *"first two"* placed before the head, with no trailing key
        modifier — so the class example's ranking reads *"the first two Robots"*.
        """
        n = request.plan.limit_number
        quality = _quality(request.plan.direction, n).as_fragment()
        pre_head = quality if n == 1 else PhraseFragment(parts=[quality, _cardinal(n)])
        return RankingSurface(
            pre_head=pre_head, number=GrammaticalNumber.of(n > 1), modifiers=[]
        )


class AttributeSuperlativeForm(LeadingRankForm):
    """Ordering by an attribute of the selection, *n = 1* → the superlative attaches to the key:
    *"the Employee with the highest salary"* / *"… with the lowest salary"*.

    >>> employee = variable(Employee, [])
    >>> verbalize_expression(entity(employee).ordered_by(employee.salary, descending=True).limit(1))
    'Find the Employee with the highest salary'
    """

    @classmethod
    def applies(cls, request: RankingRequest) -> bool:
        """:return: ``True`` for an attribute key with *n = 1*.

        This is the decision that routes a single result ordered by an attribute to the superlative
        surface — so the class example's output is *"the Employee with the highest salary"* rather
        than a leading-quality *"the highest Employee"*.
        """
        plan = request.plan
        return plan.relation is RankingKeyRelation.ATTRIBUTE and plan.limit_number == 1

    @classmethod
    def render(cls, request: RankingRequest) -> RankingSurface:
        """:return: The superlative attached to the key — *"with the highest/lowest <attribute>"*.

        It emits no leading qualifier; instead it attaches a post-nominal modifier such as *"with the
        lowest salary"* (ascending) or the class example's *"with the highest salary"* (descending) to
        the head, which is why the result trails the key rather than fronting a quality word.
        """
        superlative = (
            RankingWords.LOWEST
            if request.plan.direction is SortDirection.ASCENDING
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
            pre_head=None, number=GrammaticalNumber.SINGULAR, modifiers=[modifier]
        )


class AttributeRankedByForm(LeadingRankForm):
    """Ordering by an attribute of the selection, *n > 1* → *"the top three Employees by salary"* /
    *"the bottom three Employees by salary"*.

    >>> employee = variable(Employee, [])
    >>> verbalize_expression(entity(employee).ordered_by(employee.salary, descending=True).limit(3))
    'Find the top three Employees by salary'
    """

    @classmethod
    def applies(cls, request: RankingRequest) -> bool:
        """:return: ``True`` for an attribute key with *n > 1*.

        This is the decision that routes several results ordered by an attribute to the *"by
        <attribute>"* surface — so the class example's ranking reads *"the top three Employees by
        salary"* rather than dropping the key as the leading base form would.
        """
        plan = request.plan
        return plan.relation is RankingKeyRelation.ATTRIBUTE and plan.limit_number > 1

    @classmethod
    def render(cls, request: RankingRequest) -> RankingSurface:
        """:return: The count leading the noun with the key named — *"top/bottom <n> … by <attribute>"*.

        It emits both a leading qualifier — *"bottom three"* (ascending) or the class example's
        *"top three"* (descending) — and the post-nominal *"by salary"* modifier, so the head is
        framed on both sides.
        """
        plan = request.plan
        quality = (
            RankingWords.BOTTOM
            if plan.direction is SortDirection.ASCENDING
            else RankingWords.TOP
        )
        pre_head = PhraseFragment(
            parts=[quality.as_fragment(), _cardinal(plan.limit_number)]
        )
        modifier = PhraseFragment(
            parts=[RankingWords.BY.as_fragment(), _key_attribute(plan.order_key)]
        )
        return RankingSurface(
            pre_head=pre_head, number=GrammaticalNumber.PLURAL, modifiers=[modifier]
        )


def ranking_surface(request: RankingRequest) -> RankingSurface:
    """
    Render a query's ranking into the selection's pieces — the single entry the query assembler
    uses. The form is chosen by the registry (most-specific applicable; the leading fallback
    guarantees a match), so the caller never inspects the ranking's shape.

    :param request: The ranking and selection-type label.
    :return: The ranking pieces for the selection noun phrase.

    The chosen form drives the selection's surface — *"the top three Employees by salary"* for a
    descending attribute ranking of several:

    >>> employee = variable(Employee, [])
    >>> verbalize_expression(
    ...     the(entity(employee)).ordered_by(employee.salary, descending=True).limit(3))
    'Find the top three Employees by salary'
    """
    return RankingForm.most_applicable(request).render(request)
