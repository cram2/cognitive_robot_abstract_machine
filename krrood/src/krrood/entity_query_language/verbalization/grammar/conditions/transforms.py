from __future__ import annotations

import operator
from abc import abstractmethod
from itertools import islice

from typing_extensions import List, Optional, Set

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import Attribute
from krrood.entity_query_language.core.variable import Literal, Variable
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.verbalization.fragments.base import (
    Fragment,
    PhraseFragment,
    RoleFragment,
)
from krrood.entity_query_language.verbalization.grammar.chain.assembler import (
    ChainAssembler,
)
from krrood.entity_query_language.verbalization.grammar.chain.planner import (
    ChainPlanner,
)
from krrood.entity_query_language.verbalization.grammar.conditions.operator_phrase import (
    comparator_operator,
)
from krrood.entity_query_language.verbalization.grammar.conditions.recognition import (
    is_boolean_attribute_chain,
    is_none_literal,
    relational_verb_phrase,
)
from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import (
    RuleContext,
)
from krrood.entity_query_language.verbalization.grammar.framework.specificity import (
    SpecificityRule,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Absence,
    NonExistence,
    PassiveAbsence,
    Quantifiers,
)
from krrood.entity_query_language.verbalization.vocabulary.words import Number


def render_absence(
    comparator: Comparator, context: RuleContext, number: Number = Number.SINGULAR
) -> Fragment:
    """
    Render an ``<chain> == None`` comparison as an absence predicate rather than a value: an owned
    attribute reads *"<owner> has no <attribute>"* (the owner is the chain minus its terminal), and a
    bare variable reads *"<subject> does not exist"* (no attribute to name).

    Both flip the subject and object relative to the *"<attribute> of <owner> is <value>"* frame, so
    they are standalone predicates — never folded into a *"whose"* / *"respectively"* coordination
    (see :class:`~…conditions.forms.WhosePredicateForm`'s guard and the match assembler's None split).
    The single definition shared by the standalone predicate path (:class:`AbsenceTransform`) and the
    subject path (:class:`~…conditions.forms.AbsenceForm`).

    :param comparator: The ``<chain> == None`` comparator.
    :param context: The per-node context (recursion and services).
    :param number: The number the verb agrees with (plural owner → *"have no"* / *"do not exist"*).
    :return: The absence predicate fragment.
    """
    left = comparator.left
    if not isinstance(left, Attribute):
        return PhraseFragment(
            parts=[context.child(left), NonExistence.for_number(number).as_fragment()]
        )
    verb_phrase = relational_verb_phrase(left._attribute_name_)
    if verb_phrase is not None:
        return PhraseFragment(
            parts=[
                context.child(left._child_),
                PassiveAbsence.for_number(number).as_fragment(),
                RoleFragment.for_attribute(
                    left._owner_class_, left._attribute_name_, text=verb_phrase
                ),
                *_relation_target(left),
            ]
        )
    return PhraseFragment(
        parts=[
            context.child(left._child_),
            Absence.for_number(number).as_fragment(),
            RoleFragment.for_attribute(left._owner_class_, left._attribute_name_),
        ]
    )


def _relation_target(attribute: Attribute) -> List[Fragment]:
    """:return: The object of a passive absence — *"any <Type>"* using the attribute's declared
    related type (*"any Robot"*), or the bare *"anything"* when that type is not a nameable class
    (a primitive, a typing generic, or unknown)."""
    related_type = getattr(attribute, "_type_", None)
    if isinstance(related_type, type) and related_type.__module__ != "builtins":
        return [Quantifiers.ANY.as_fragment(), RoleFragment.for_type(related_type)]
    return [Quantifiers.ANYTHING.as_fragment()]


def _boolean_constraint(right: SymbolicExpression) -> Optional[Set[bool]]:
    """:return: The set of boolean values *right* constrains a boolean attribute to — ``{True}`` /
    ``{False}`` for a boolean literal or singleton domain, ``{True, False}`` for an open domain — or
    ``None`` when *right* is not a boolean literal / bounded boolean-domain variable.
    """
    if isinstance(right, Literal) and isinstance(right._value_, bool):
        return {right._value_}
    if isinstance(right, Variable) and getattr(right, "_type_", None) is bool:
        values = list(islice(right._re_enterable_domain_generator_, 3))
        if values and len(values) <= 2 and all(isinstance(v, bool) for v in values):
            return set(values)
    return None


class PredicateTransform(SpecificityRule):
    """
    One way to say a comparator as a standalone predicate — the registry the
    :meth:`~…conditions.assembler.ConditionAssembler.predicate` form dispatches over.

    :class:`GenericComparator` is the unguarded base every special case refines, so
    :meth:`~SpecificityRule.most_applicable` always returns a transform (the more-specific one when it
    applies, else the generic *"<left> <op> <right>"*). Adding a predicate-level transform is a new
    subclass; the ``predicate`` method does not change (open/closed) — mirroring the
    :class:`~…conditions.forms.ConditionForm` registry one level up.
    """

    @classmethod
    @abstractmethod
    def applies(cls, comparator: Comparator, negated: bool) -> bool:
        """
        :param comparator: The comparator being said as a predicate.
        :param negated: Whether an outer negation applies.
        :return: ``True`` when this transform renders *comparator*.
        """

    @classmethod
    @abstractmethod
    def render(
        cls, comparator: Comparator, context: RuleContext, negated: bool
    ) -> Fragment:
        """
        :param comparator: The comparator to render.
        :param context: The per-node context (recursion and services).
        :param negated: Whether an outer negation applies.
        :return: *comparator* rendered in this transform's form.
        """


class GenericComparator(PredicateTransform):
    """The unguarded default: *"<left> <operator> <right>"* (the right side in value position)."""

    @classmethod
    def applies(cls, comparator: Comparator, negated: bool) -> bool:
        return True

    @classmethod
    def render(
        cls, comparator: Comparator, context: RuleContext, negated: bool
    ) -> Fragment:
        return PhraseFragment(
            parts=[
                context.child(comparator.left),
                comparator_operator(comparator, context.services, negated=negated),
                context.child(comparator.right, as_value=True),
            ]
        )


class AbsenceTransform(GenericComparator):
    """A non-negated ``<chain> == None`` comparison → the absence predicate (*"has no …"* / *"does
    not exist"*) instead of a value comparison."""

    @classmethod
    def applies(cls, comparator: Comparator, negated: bool) -> bool:
        return (
            not negated
            and comparator.operation is operator.eq
            and is_none_literal(comparator.right)
        )

    @classmethod
    def render(
        cls, comparator: Comparator, context: RuleContext, negated: bool
    ) -> Fragment:
        return render_absence(comparator, context)


class BooleanPolarityTransform(GenericComparator):
    """A boolean attribute compared to a boolean value → a predicative folding the value into the
    verb's polarity (*"is decaf"* / *"is not decaf"* / *"is either decaf or not"*), never *"is decaf
    is True"*."""

    @classmethod
    def applies(cls, comparator: Comparator, negated: bool) -> bool:
        return (
            comparator.operation in (operator.eq, operator.ne)
            and is_boolean_attribute_chain(comparator.left)
            and _boolean_constraint(comparator.right) is not None
        )

    @classmethod
    def render(
        cls, comparator: Comparator, context: RuleContext, negated: bool
    ) -> Fragment:
        constraint = _boolean_constraint(comparator.right)
        plan = context.microplan.plan_for(comparator.left, ChainPlanner)
        chain = ChainAssembler(context)
        if constraint == {True, False}:
            return chain.boolean_alternative(plan)
        positive = True in constraint
        if comparator.operation is operator.ne:
            positive = not positive
        if negated:
            positive = not positive
        return chain.boolean_predicative(plan, negated=not positive)
