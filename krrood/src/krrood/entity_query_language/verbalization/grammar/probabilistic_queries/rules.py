"""
Grammar rules for the probabilistic query constructs:
:class:`~krrood.entity_query_language.operators.probabilistic_queries.Distribution` and
:class:`~krrood.entity_query_language.operators.probabilistic_queries.Probability`.

Auto-registered by :mod:`~krrood.entity_query_language.verbalization.grammar.framework.registry`
(any module named ``rules.py`` under ``grammar/`` is walked and imported, and every concrete
:class:`~krrood.entity_query_language.verbalization.grammar.framework.phrase_rule.PhraseRule`
subclass found is added to ``RULES``) -- nothing else needs editing to wire these in.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import List

from krrood.entity_query_language.operators.probabilistic_queries import (
    Distribution,
    Probability,
)
from krrood.entity_query_language.verbalization.fragments.base import (
    BlockFragment,
    PhraseFragment,
    RoleFragment,
    VerbalizationFragment,
    oxford_comma,
)
from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import (
    PhraseRule,
    RuleContext,
)
from krrood.entity_query_language.verbalization.grammar.match.assembler import (
    MatchAssembler,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Articles,
    Conjunctions,
    Directive,
    Keywords,
    Prepositions,
)


@dataclass
class DistributionRule(PhraseRule):
    """
    Realise ``distribution_of(match)`` as *"The distribution over <match subject>,
    given that …, where …"*, and ``distribution_of(match, marginalize_for=...)``
    (marginalized to a subset of the match's free variables) as *"The distribution
    over the <variables> of <match subject>, given that …, where …"*.

    Reuses the match's own *"given that"*/*"where"* grammar (see
    :class:`~krrood.entity_query_language.verbalization.grammar.match.assembler.MatchAssembler`),
    but builds its own header rather than delegating to :meth:`MatchAssembler.realize`
    outright, for two reasons: the header takes a definite description
    (:attr:`~....vocabulary.english.Directive.DISTRIBUTION_OVER`) rather than either of
    the match's imperative *"Find"*/*"Generate"* verbs, since a distribution is asked
    for, not rows; and a distribution's underspecified (``...``) fields are its
    free/output variables by default, not values to fill in, so the match's generative
    *"and predict its … value(s)"* clause is dropped entirely rather than reworded --
    "the distribution over a Coin" already means "over every attribute not given", the
    same way an unqualified marginal does, so nothing is lost by omitting them. When
    ``marginalize_for`` narrows that default, they're named directly in the subject
    (*"the battery of a Robot"*) -- not a trailing qualifier like *"restricted to"*,
    which reads like a truncation (a ``where``), not a choice of which variables the
    joint is even over.

    >>> from krrood.entity_query_language.factories import a, distribution_of
    >>> match = a(Robot)(name="R2", battery=...)
    >>> verbalize_expression(distribution_of(match))
    "The distribution over a Robot given that its name is 'R2'"
    >>> verbalize_expression(distribution_of(match, marginalize_for=(match.variable.battery,)))
    "The distribution over the battery of a Robot given that its name is 'R2'"
    """

    construct = Distribution

    def build(self, node: Distribution, context: RuleContext) -> VerbalizationFragment:
        context.services.performative_override = Directive.DISTRIBUTION_OVER
        assembler = MatchAssembler(context)
        plan = assembler.plan(node.match)

        subject = context.child(plan.selection)
        if node.marginalize_for:
            variable_list = oxford_comma(
                [
                    RoleFragment.for_attribute(
                        attribute._owner_class_, attribute._attribute_name_
                    )
                    for attribute in node.marginalize_for
                ],
                Conjunctions.AND.as_fragment(),
            )
            subject = PhraseFragment(
                parts=[
                    Articles.THE.as_fragment(),
                    variable_list,
                    Prepositions.OF.as_fragment(),
                    subject,
                ]
            )
        header = PhraseFragment(
            parts=[Directive.DISTRIBUTION_OVER.as_fragment(), subject]
        )

        items: List[VerbalizationFragment] = []
        given = assembler._given_that_block(plan)
        if given is not None:
            items.append(given)
        where = assembler._where_block(plan)
        if where is not None:
            items.append(where)

        return BlockFragment(header=header, items=items, source=node.match.expression)


@dataclass
class ProbabilityRule(PhraseRule):
    """
    Realise ``probability_of(condition)`` as *"the probability that <condition>"* --
    the condition is recursed through the ordinary comparator/logical-operator grammar
    via ``context.child``, so it reads exactly as it would inside a ``where`` clause.

    >>> from krrood.entity_query_language.factories import probability_of, variable
    >>> robot = variable(Robot, [])
    >>> verbalize_expression(probability_of(robot.battery > 50))
    'the probability that the battery of a Robot is greater than 50'
    """

    construct = Probability

    def build(self, node: Probability, context: RuleContext) -> VerbalizationFragment:
        return PhraseFragment(
            parts=[
                Keywords.THE_PROBABILITY_THAT.as_fragment(),
                context.child(node.condition),
            ]
        )
