from __future__ import annotations

from typing_extensions import List

from krrood.entity_query_language.query.query import Query
from krrood.entity_query_language.verbalization.fragments.base import (
    NounPhrase,
    PhraseFragment,
    RoleFragment,
    Fragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    Definiteness,
    Number,
)
from krrood.entity_query_language.verbalization.grammar.aggregation.kinds import (
    AGGREGATION_KIND,
)
from krrood.entity_query_language.verbalization.grammar.framework.assembler import Assembler
from krrood.entity_query_language.verbalization.grammar.clauses.assembler import (
    HavingAssembler,
)
from krrood.entity_query_language.verbalization.grammar.conditions.restriction_assembler import (
    RestrictionAssembler,
)
from krrood.entity_query_language.verbalization.grammar.query.planner import (
    QueryPlan,
    QueryPlanner,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    FallbackNouns,
    Keywords,
    Prepositions,
)

class AggregationValueAssembler(Assembler[Query, QueryPlan]):
    """
    Realise an aggregation value-subquery from its query plan — an aggregation used as a *value*
    (*"the maximum amount"*, or *"the sum of amounts among BankTransactions whose …"*).  The
    aggregate noun is composed with an optional *"among <plural source> [whose/such that]
    [having]"* scope.

    Reference: Reiter & Dale (2000) — aggregation; Gatt & Reiter (2009), SimpleNLG — realisation.

    >>> transaction = variable(BankTransaction, [])
    >>> verbalize_expression(an(entity(max(transaction.amount_details.amount))))
    'Find the maximum of the amount of the amount_details of a BankTransaction'
    """

    planner = QueryPlanner

    def realize(self, node: Query, plan: QueryPlan) -> Fragment:
        """
        :param node: The aggregation value-subquery.
        :param plan: The query plan.
        :return: *"the <aggregation> <leaf>"*, optionally scoped *"among <source> …"*.
        """
        aggregation_data = plan.aggregation_data
        if aggregation_data.leaf is None:
            return self.context.child(aggregation_data.aggregator)

        aggregation_kind = AGGREGATION_KIND[type(aggregation_data.aggregator)]
        leaf_fragment = RoleFragment.for_attribute(
            aggregation_data.leaf._owner_class_,
            aggregation_data.leaf._attribute_name_,
            number=aggregation_kind.child_number,
        )

        aggregate = NounPhrase(
            head=aggregation_kind.as_fragment(),
            definiteness=Definiteness.DEFINITE,
            modifiers=[leaf_fragment],
        )

        if not aggregation_data.is_constrained_or_grouped:
            return aggregate
        return self._scope(node, plan, aggregate)

    def _scope(self, node: Query, plan: QueryPlan, aggregate: Fragment) -> Fragment:
        """
        Build the *"among …"* scope of a constrained aggregation — the source population followed
        by its restrictions and any HAVING clause. (Whether chains rooted at the population
        pronominalise is the coreference pass's concern, read from this query's provenance.)

        :param node: The aggregation value-subquery.
        :param plan: The query plan.
        :param aggregate: The already-built aggregate noun phrase.
        :return: *"<aggregate> among <source> [whose …] [such that …] [having …]"*.
        """
        source = plan.aggregation_data.source
        source_fragment = (
            self.context.child(source, number=Number.PLURAL)
            if source is not None
            else FallbackNouns.ENTITY.plural_fragment()
        )
        parts: List[Fragment] = [
            aggregate,
            Prepositions.AMONG.as_fragment(),
            source_fragment,
        ]

        if plan.subject_restriction is not None:
            rendered = RestrictionAssembler(self.context).render(
                plan.subject_restriction, plan.subject
            )
            parts.extend(rendered.superlatives)
            if rendered.whose is not None:
                parts.append(rendered.whose)
            if rendered.residual is not None:
                parts += [Keywords.SUCH_THAT.as_fragment(), rendered.residual]

        having = HavingAssembler(self.context).clause(node)
        if having is not None:
            parts.append(having)

        # No scope marker: the engine stamps this phrase with its aggregation-query node, and the
        # coreference pass reads the focus (the population) for that query from the discourse view.
        return PhraseFragment(parts=parts)
