from __future__ import annotations

from krrood.entity_query_language.operators.aggregators import Aggregator
from krrood.entity_query_language.verbalization.fragments.base import (
    VerbalizationFragment,
    NounPhrase,
)
from krrood.entity_query_language.verbalization.fragments.features import Definiteness
from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import (
    PhraseRule,
    RuleContext,
)
from krrood.entity_query_language.verbalization.vocabulary.english import Aggregations


class AggregatorRule(PhraseRule):
    """*"the <aggregation> <plural child>"* (or *"the <aggregation> of <child>"*).

    >>> verbalize_expression(max(variable(Robot, []).battery))
    'the maximum of the battery of a Robot'
    """

    construct = Aggregator

    def build(self, node: Aggregator, context: RuleContext) -> VerbalizationFragment:
        """:return: the definite noun phrase for *node* — *"the <aggregation> of <child>"* — or the
        bare aggregation word for a childless aggregate.

        >>> verbalize_expression(max(variable(BankTransaction, []).amount_details.amount))
        'the maximum of the amount of the amount_details of a BankTransaction'
        """
        # The aggregation word owns its complement realisation (the "of" and the child's number);
        # the rule only chooses the structure — a childless aggregate is the bare word, otherwise a
        # definite noun phrase around the lexicon-built complement.
        kind = Aggregations.for_aggregator(type(node))
        if not kind.has_child:
            return kind.as_fragment()  # childless aggregate, e.g. "count of all"
        child_fragment = context.child(node._child_, number=kind.child_number)
        # In a fronted *"whose <aggregate> is …"* group filter the aggregate is a possession of the
        # group key, so it drops its determiner (*"sum of salaries"*) and does not register for
        # coreference — the reported column stays the canonical full mention.
        if context.configuration.possessive_aggregate:
            return NounPhrase(
                head=kind.as_fragment(),
                definiteness=Definiteness.BARE,
                modifiers=kind.complement(child_fragment),
                subject_of_modifiers=False,
            )
        # A computed quantity is a referring expression: named in full when first introduced (the
        # reported column), a later mention of the same aggregate (a HAVING / ordering on it) is the
        # general repeat-reduction's job — "the sum of salaries of Employees" → "the sum".
        return NounPhrase(
            head=kind.as_fragment(),
            definiteness=Definiteness.DEFINITE,
            modifiers=kind.complement(child_fragment),
            referent_id=node._id_,
            subject_of_modifiers=False,
        )
