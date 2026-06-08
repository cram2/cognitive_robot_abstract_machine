"""
Subject restriction clauses — turning a subject's ``WHERE`` condition into a
post-nominal *"whose <attr> is …"* modifier plus a residual *"such that …"* clause.

The "is this conjunct a groupable subject predicate, and how does it render?" decision
is **declarative**: a small :class:`RestrictionRule` hierarchy, dispatched first-match-wins
by specificity (mirroring the main
:class:`~krrood.entity_query_language.verbalization.rule_engine.RuleEngine`).  A folded
conjunct that **no** rule matches becomes residual — groupability lives in the rule
preconditions, not in branching glue.

:class:`RestrictionClauseBuilder` is the thin orchestrator used by the
query-body assembly in :mod:`~krrood.entity_query_language.verbalization.rules.query`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing_extensions import List, Optional, Type

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import Attribute, MappedVariable
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import ConditionType
from krrood.entity_query_language.operators.aggregators import Aggregator
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.operators.core_logical_operators import (
    AND,
    flatten_operands,
)
from krrood.entity_query_language.verbalization.chain_utils import (
    chain_root,
    walk_chain,
)
from krrood.entity_query_language.verbalization.fragments.base import (
    oxford_and,
    PhraseFragment,
    RoleFragment,
    VerbFragment,
)
from krrood.entity_query_language.verbalization.operator_phrase import (
    comparator_operator,
)
from krrood.entity_query_language.verbalization.microplanning.coordination import (
    build_between,
    fold_range_pairs,
    RangeFold,
)
from krrood.entity_query_language.verbalization.subquery import aggregation_source_root
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Conjunctions,
    Keywords,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.verbalization.context import VerbalizationContext
    from krrood.entity_query_language.verbalization.verbalizer import EQLVerbalizer


# ── Chain helpers ──────────────────────────────────────────────────────────────


def _single_hop_attr(expression, subject_variable):
    """Return the :class:`Attribute` node when *expression* is exactly ``subject_variable.<attr>``, else ``None``."""
    if subject_variable is None or not isinstance(expression, MappedVariable):
        return None
    chain, root = walk_chain(expression)
    if not (isinstance(root, Variable) and root._id_ == subject_variable._id_):
        return None
    if len(chain) != 1 or not isinstance(chain[0], Attribute):
        return None
    return chain[0]


def _references(expression, subject_variable) -> bool:
    """Return ``True`` when *expression* mentions *subject_variable* (so it is not a clean RHS value)."""
    try:
        return any(
            getattr(variable, "_id_", None) == subject_variable._id_
            for variable in expression._unique_variables_
        )
    except AttributeError:
        return chain_root(expression) is subject_variable


# ── Restriction rule hierarchy ──────────────────────────────────────────────────


class RestrictionRule(ABC):
    """
    Declarative rule that recognises a folded conjunct as a groupable subject
    restriction and renders it as a bare post-nominal predicate (no root / pronoun;
    the enclosing *"whose"* supplies the subject).

    A conjunct matched by no rule is residual and stays in a *"such that …"* clause.
    """

    @classmethod
    @abstractmethod
    def applies(cls, item, subject_variable, context: VerbalizationContext) -> bool:
        """Return ``True`` when *item* is a groupable restriction on *subject_variable*."""

    @classmethod
    @abstractmethod
    def render(
        cls,
        item,
        subject_variable,
        context: VerbalizationContext,
        verbalizer: EQLVerbalizer,
    ) -> VerbFragment:
        """Render *item* as a bare predicate fragment (e.g. *"booking_date is between X and Y"*)."""


class RangeRestrictionRule(RestrictionRule):
    """Matches a :class:`RangeFold` on a single-hop subject attribute → *"<attr> is between lo and hi"*."""

    @classmethod
    def applies(cls, item, subject_variable, context: VerbalizationContext) -> bool:
        return (
            isinstance(item, RangeFold)
            and _single_hop_attr(item.chain_expression, subject_variable) is not None
        )

    @classmethod
    def render(
        cls,
        item,
        subject_variable,
        context: VerbalizationContext,
        verbalizer: EQLVerbalizer,
    ) -> VerbFragment:
        attr = _single_hop_attr(item.chain_expression, subject_variable)
        left = RoleFragment.for_attribute(attr._owner_class_, attr._attribute_name_)
        return build_between(
            left,
            verbalizer.build(item.lower_expression, context),
            verbalizer.build(item.upper_expression, context),
            compact=False,
        )


class AttributePredicateRestrictionRule(RestrictionRule):
    """
    Matches a :class:`~krrood.entity_query_language.operators.comparator.Comparator`
    that is a single-hop, non-boolean subject-attribute predicate whose RHS does not
    reference the subject → *"<attr> is greater than 100"* / *"<attr> is equal to <calc>"*.
    """

    @classmethod
    def applies(cls, item, subject_variable, context: VerbalizationContext) -> bool:
        if not isinstance(item, Comparator):
            return False
        attr = _single_hop_attr(item.left, subject_variable)
        if attr is None or attr._type_ is bool:
            return False
        return not _references(item.right, subject_variable)

    @classmethod
    def render(
        cls,
        item,
        subject_variable,
        context: VerbalizationContext,
        verbalizer: EQLVerbalizer,
    ) -> VerbFragment:
        attr = _single_hop_attr(item.left, subject_variable)
        attribute_fragment = RoleFragment.for_attribute(
            attr._owner_class_, attr._attribute_name_
        )
        operator_fragment = comparator_operator(item, context, compact=False)
        return PhraseFragment(
            parts=[
                attribute_fragment,
                operator_fragment,
                verbalizer.build(item.right, context),
            ]
        )


RESTRICTION_RULES: List[Type[RestrictionRule]] = [
    RangeRestrictionRule,
    AttributePredicateRestrictionRule,
]


# ── Restriction subject resolution ──────────────────────────────────────────────


class RestrictionSubjectRule(ABC):
    """
    Declarative rule resolving **which variable** a query's selection restricts, so the
    selection's ``WHERE`` clause can fold into a post-nominal *"whose …"* modifier on it.

    Dispatched first-match-wins by registry order (mirroring :class:`RestrictionRule`).  A
    selection matched by **no** rule has no groupable subject — its ``WHERE`` clause stays a
    full *"such that …"* clause.
    """

    @classmethod
    @abstractmethod
    def applies(
        cls, expression, selected_variable, context: VerbalizationContext
    ) -> bool:
        """Return ``True`` when this rule can name the restriction subject of *expression*."""

    @classmethod
    @abstractmethod
    def subject(cls, expression, selected_variable, context: VerbalizationContext):
        """Return the :class:`~krrood.entity_query_language.core.variable.Variable` the ``WHERE`` restricts."""


class SelectedVariableSubjectRule(RestrictionSubjectRule):
    """The selection itself is a plain :class:`~krrood.entity_query_language.core.variable.Variable` → it is its own subject."""

    @classmethod
    def applies(
        cls, expression, selected_variable, context: VerbalizationContext
    ) -> bool:
        return isinstance(selected_variable, Variable)

    @classmethod
    def subject(cls, expression, selected_variable, context: VerbalizationContext):
        return selected_variable


class AggregationSourceSubjectRule(RestrictionSubjectRule):
    """
    The selection is an aggregation over a single source variable's chain
    (e.g. ``max(t.amount_details.amount)`` or ``max(t, key=…)``).

    The aggregation selection renders its child chain root-last
    (*"the average of the amount of the amount_details of a BankTransaction"*), and
    :func:`~krrood.entity_query_language.verbalization.subquery.aggregation_source_root`
    returns exactly that chain root, so the selection ends with the source variable's noun
    and a *"whose …"* modifier attaches to it grammatically.  The ``WHERE`` therefore
    restricts the aggregated entity.
    """

    @classmethod
    def applies(
        cls, expression, selected_variable, context: VerbalizationContext
    ) -> bool:
        return (
            isinstance(selected_variable, Aggregator)
            and aggregation_source_root(expression) is not None
        )

    @classmethod
    def subject(cls, expression, selected_variable, context: VerbalizationContext):
        return aggregation_source_root(expression)


RESTRICTION_SUBJECT_RULES: List[Type[RestrictionSubjectRule]] = [
    SelectedVariableSubjectRule,
    AggregationSourceSubjectRule,
]


def restriction_subject(expression, selected_variable, context: VerbalizationContext):
    """
    Resolve the variable a selection's ``WHERE`` clause restricts (first match in
    :data:`RESTRICTION_SUBJECT_RULES` wins), or ``None`` when nothing groups.

    :param expression: The :class:`~krrood.entity_query_language.query.query.Entity` being verbalized.
    :param selected_variable: Its selected variable (a ``Variable``, ``Aggregator``, …).
    :param context: Shared verbalization state.
    :return: The subject ``Variable`` for *"whose"* folding, or ``None``.
    :rtype: ~krrood.entity_query_language.core.variable.Variable or None
    """
    for rule in RESTRICTION_SUBJECT_RULES:
        if rule.applies(expression, selected_variable, context):
            return rule.subject(expression, selected_variable, context)
    return None


# ── Builder ─────────────────────────────────────────────────────────────────────


@dataclass
class RestrictionClauseBuilder:
    """
    Partition a subject's ``WHERE`` condition into a *"whose <grouped>"* modifier and a
    residual condition fragment.
    """

    _verbalizer: EQLVerbalizer
    """The parent verbalizer for recursive sub-expression rendering."""

    def build(
        self,
        subject_variable: Optional[Variable],
        condition: SymbolicExpression,
        context: VerbalizationContext,
    ) -> tuple[Optional[VerbFragment], Optional[VerbFragment]]:
        """
        :param subject_variable: The subject the conditions restrict (its single-hop attribute
            predicates become *"whose"* modifiers); may be ``None`` (nothing groups).
        :param condition: The ``WHERE`` condition expression.
        :param context: Shared verbalization state. Residual conditions are rendered through the
            main engine, so push *subject_variable* as the coreference subject **before** calling
            this when pronouns are wanted in the residual.
        :return: ``(whose_modifier | None, residual_condition | None)``.  ``whose_modifier``
            already includes the ``whose`` keyword; ``residual_condition`` does **not** include
            a ``such that`` / ``where`` keyword (the caller adds the appropriate one).
        :rtype: tuple
        """
        items = fold_range_pairs(flatten_operands(condition, AND))
        grouped: List[VerbFragment] = []
        residual: List = []
        for item in items:
            rule = self._match(item, subject_variable, context)
            if rule is None:
                residual.append(item)
            else:
                grouped.append(
                    rule.render(item, subject_variable, context, self._verbalizer)
                )

        whose_fragment = None
        if grouped:
            whose_fragment = PhraseFragment(
                parts=[
                    Keywords.WHOSE.as_fragment(),
                    oxford_and(grouped, Conjunctions.AND.as_fragment()),
                ],
            )
        residual_fragment = (
            self._render_residual(residual, context) if residual else None
        )
        return whose_fragment, residual_fragment

    @staticmethod
    def _match(
        item, subject_variable, context: VerbalizationContext
    ) -> Optional[Type[RestrictionRule]]:
        for rule in RESTRICTION_RULES:
            if rule.applies(item, subject_variable, context):
                return rule
        return None

    def _render_residual(
        self, items: List, context: VerbalizationContext
    ) -> VerbFragment:
        parts: List[VerbFragment] = []
        for item in items:
            if isinstance(item, RangeFold):
                parts.append(
                    build_between(
                        self._verbalizer.build(item.chain_expression, context),
                        self._verbalizer.build(item.lower_expression, context),
                        self._verbalizer.build(item.upper_expression, context),
                        compact=context.compact_predicates,
                    )
                )
            else:
                parts.append(self._verbalizer.build(item, context))
        if len(parts) == 1:
            return parts[0]
        return oxford_and(parts, Conjunctions.AND.as_fragment())
