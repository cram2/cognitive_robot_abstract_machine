"""
Grammar rules for the probabilistic query constructs:
:class:`~krrood.parametrization.probabilistic_queries.Distribution` and
:class:`~krrood.parametrization.probabilistic_queries.Probability`.

:meth:`~krrood.entity_query_language.verbalization.grammar.framework.registry.PhraseRuleRegistry.loaded_rules`
includes them once this module is loaded, which loading :mod:`krrood.parametrization` does.
"""

from __future__ import annotations

from dataclasses import dataclass


from krrood.parametrization.probabilistic_queries import (
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


@dataclass(kw_only=True)
class DistributionRule(PhraseRule):
    """
    Realise ``distribution_of(match)`` as *"The distribution over <match subject>,
    given that …, where …"*, naming the ``marginalize_for`` variables in the subject.

    Example::

        >>> from krrood.entity_query_language.factories import a
        >>> from krrood.entity_query_language.factories import distribution_of
        >>> match = a(Robot)(name="R2", battery=...)
        >>> verbalize_expression(distribution_of(match))
        "The distribution over a Robot given that its name is 'R2'"
        >>> verbalize_expression(distribution_of(match, marginalize_for=(match.battery,)))
        "The distribution over the battery of a Robot given that its name is 'R2'"
    """

    construct = Distribution

    def build(
        self, *, node: Distribution, context: RuleContext
    ) -> VerbalizationFragment:
        context.services.performative_override = Directive.DISTRIBUTION_OVER
        assembler = MatchAssembler(context=context)
        plan = assembler.plan(node=node.match)

        subject = context.child(node=plan.selection)
        if node.marginalize_for:
            variable_list = oxford_comma(
                [
                    RoleFragment.for_attribute(
                        owner=attribute._owner_class_,
                        attribute_name=attribute._attribute_name_,
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

        items: list[VerbalizationFragment] = []
        given = assembler._given_that_block(plan=plan)
        if given is not None:
            items.append(given)
        where = assembler._where_block(plan=plan)
        if where is not None:
            items.append(where)

        return BlockFragment(
            header=header, items=items, source=node.match._get_expression_()
        )


@dataclass(kw_only=True)
class ProbabilityRule(PhraseRule):
    """
    Realise ``probability_of(condition, evidence=evidence)`` as *"the probability that
    <condition> given that <evidence>"*, reading condition and evidence as they would
    inside a ``where`` clause.

    Example::

        >>> from krrood.entity_query_language.factories import variable
        >>> from krrood.entity_query_language.factories import probability_of
        >>> robot = variable(Robot, [])
        >>> verbalize_expression(probability_of(robot.battery > 50))
        'the probability that the battery of a Robot is greater than 50'
    """

    construct = Probability

    def build(
        self, *, node: Probability, context: RuleContext
    ) -> VerbalizationFragment:
        parts = [
            Keywords.THE_PROBABILITY_THAT.as_fragment(),
            context.child(node=node.condition),
        ]
        if node.evidence is not None:
            parts += [
                Keywords.GIVEN_THAT.as_fragment(),
                context.child(node=node.evidence),
            ]
        return PhraseFragment(parts=parts)
