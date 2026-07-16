from __future__ import annotations

from typing_extensions import List

from krrood.entity_query_language.query.query import Query
from krrood.entity_query_language.verbalization.fragments.base import (
    NounPhrase,
    PhraseFragment,
    RoleFragment,
    VerbalizationFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    Definiteness,
    GrammaticalNumber,
)
from krrood.entity_query_language.verbalization.grammar.framework.assembler import (
    Assembler,
)
from krrood.entity_query_language.verbalization.grammar.clauses.composer import (
    ClauseComposer,
)
from krrood.entity_query_language.verbalization.grammar.query.planner import (
    QueryPlan,
    QueryPlanner,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Aggregations,
    FallbackNouns,
    Keywords,
    Prepositions,
)


class AggregationValueAssembler(Assembler[Query, QueryPlan]):
    """
    Realise an aggregation value-subquery from its query plan — an aggregation used as a
    *value* (*"the maximum amount"*, or *"the sum of amounts among BankTransactions
    whose …"*).  The aggregate noun is composed with an optional *"among <plural source>
    [whose/such that] [having]"* scope.

    >>> verbalize_expression(an(entity(max(variable(BankTransaction, []).amount_details.amount))))
    'Find the maximum of the amount of the amount_details of a BankTransaction'

    Reference: :cite:t:`reiter2000building` — aggregation; :cite:t:`gatt2009simplenlg` — realisation.
    """

    planner = QueryPlanner

    def realize(self, node: Query, plan: QueryPlan) -> VerbalizationFragment:
        """
        The unconstrained aggregate value — *"the <aggregation> <leaf>"*; a constrained
        one adds an *"among <population> …"* scope (see :meth:`_among_population`).

        :param node: The aggregation value-subquery.
        :param plan: The query plan.
        :return: *"the <aggregation> <leaf>"*, optionally scoped *"among <source> …"*.
        """
        aggregation_data = plan.aggregation_data
        if aggregation_data.leaf is None:
            return self.context.child(aggregation_data.aggregator)

        aggregation_kind = Aggregations.for_aggregator(
            type(aggregation_data.aggregator)
        )
        leaf_fragment = RoleFragment.for_attribute(
            aggregation_data.leaf._owner_class_,
            aggregation_data.leaf._attribute_name_,
            number=aggregation_kind.child_number,
        )

        aggregate = NounPhrase(
            head=aggregation_kind.as_fragment(),
            definiteness=Definiteness.DEFINITE,
            modifiers=aggregation_kind.compact_complement(leaf_fragment),
        )

        if not aggregation_data.is_constrained_or_grouped:
            return aggregate
        return self._among_population(node, plan, aggregate)

    def _among_population(
        self, node: Query, plan: QueryPlan, aggregate: VerbalizationFragment
    ) -> VerbalizationFragment:
        """
        Build the *"among <population> …"* scope of a constrained aggregation — the source
        population followed by its restriction and any HAVING clause. The clauses are rendered by
        the shared :class:`ClauseComposer`; this method only places them inline. (Whether chains
        rooted at the population pronominalise is the coreference pass's concern, read from this
        query's provenance.)

        >>> t1 = variable(BankTransaction, domain=None)
        >>> t2 = variable(BankTransaction, domain=None)
        >>> verbalize_expression(the(entity(t1).where(
        ...     t1.amount_details.amount == an(entity(max(t2.amount_details.amount)).where(
        ...         t2.booking_date < datetime.datetime(2024, 1, 1))))))
        'Find the unique BankTransaction such that the amount of its amount_details is equal to the maximum amount among BankTransactions whose booking_date is before January 1, 2024'

        :param node: The aggregation value-subquery.
        :param plan: The query plan.
        :param aggregate: The already-built aggregate noun phrase.
        :return: *"<aggregate> among <source> [whose …] [such that …] [having …]"*.
        """
        source = plan.aggregation_data.source
        source_fragment = (
            self.context.child(source, number=GrammaticalNumber.PLURAL)
            if source is not None
            else FallbackNouns.ENTITY.plural_fragment()
        )
        parts: List[VerbalizationFragment] = [
            aggregate,
            Prepositions.AMONG.as_fragment(),
            source_fragment,
        ]

        composer = ClauseComposer(self.context)
        rendered = composer.restriction(plan)
        if rendered is not None:
            parts.extend(rendered.inline_modifiers)
            # This aggregate value is an inline noun phrase, so the "whose" block flattens inline.
            if rendered.whose is not None:
                parts.append(rendered.whose)
            if rendered.residual is not None:
                parts += [Keywords.SUCH_THAT.as_fragment(), rendered.residual]

        having = composer.having(node)
        if having is not None:
            parts.append(having)

        # No scope marker: the engine stamps this phrase with its aggregation-query node, and the
        # coreference pass reads the focus (the population) for that query from the discourse view.
        return PhraseFragment(parts=parts)
