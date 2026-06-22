from __future__ import annotations

import operator

from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.verbalization.fragments.base import (
    PhraseFragment,
    RoleFragment,
    Fragment,
)
from krrood.entity_query_language.verbalization.grammar.aggregation.kinds import (
    AGGREGATION_KIND,
)
from krrood.entity_query_language.verbalization.grammar.framework.assembler import (
    Assembler,
)
from krrood.entity_query_language.verbalization.grammar.conditions.operator_phrase import (
    comparator_operator,
)
from krrood.entity_query_language.verbalization.grammar.conditions.recognition import (
    single_hop_attribute,
    superlative_aggregation,
)
from krrood.entity_query_language.verbalization.grammar.conditions.transforms import (
    PredicateTransform,
)
from krrood.entity_query_language.verbalization.microplanning.coordination import (
    reduce_conjuncts,
    RangeFold,
    build_between,
)
from typing_extensions import List
from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Articles,
    Copulas,
    Prepositions,
)
from krrood.entity_query_language.verbalization.vocabulary.words import Number


class ConditionAssembler(Assembler[Comparator, None]):
    """
    Render a condition in a requested surface form (predicate / modifier / …) — the single owner
    of every surface form a condition can take.

    A comparator/condition is said differently depending on where it sits: a standalone predicate
    (*"x is greater than 5"*), a post-nominal attribute modifier on a subject (the bare *"<attribute>
    op <value>"* that a *"whose …"* envelope wraps), a range modifier (*"<attribute> is between low and
    high"*), or the inference whose-attribute body (*"<attribute> is <value>"* agreeing in number).

    Reference: Gatt & Reiter (2009), SimpleNLG — surface realisation.
    """

    def realize(self, node: Comparator, plan: None = None) -> Fragment:
        """
        :param node: The condition (comparator) to render.
        :param plan: Unused (this assembler has no plan).
        :return: The default form — a standalone predicate.
        """
        return self.predicate(node)

    def predicate(self, comparator: Comparator, *, negated: bool = False) -> Fragment:
        """
        :param comparator: The comparator to render.
        :param negated: Whether an outer negation applies.
        :return: The standalone predicate — dispatched over the :class:`PredicateTransform` registry,
            so the generic *"<left> <operator> <right>"* is one transform alongside the absence
            (*"has no …"* / *"does not exist"*) and boolean-polarity (*"is [not] <attr>"*) forms; the
            most-specific applicable one wins, and adding a new one is a new subclass.
        """
        transform = PredicateTransform.most_applicable(comparator, negated)
        return transform.render(comparator, self.context, negated)

    def as_statements(self, conditions: List[SymbolicExpression]) -> List[Fragment]:
        """
        Say a list of conditions as standalone statements — the entry a caller uses when the
        conditions stand on their own (an ``AND``'s operands, a ``where`` block), as opposed to
        attaching to a subject noun (:func:`as_subject_restrictions`). The verbalizer decides
        everything inside: it reduces the conjuncts (a complementary lower/upper bound pair on one
        chain becomes one *"… is between …"*; co-indexed comparisons across two prefixes fold into
        one *"… have the same …"*) and says each resulting condition.

        The caller only knows it has conditions and that this says them; it never sees the folding,
        nor chooses among the per-form methods below.

        :param conditions: The conditions to say, in order.
        :return: One standalone-statement fragment per condition (after reduction), in order.
        """
        return [self.context.child(item) for item in reduce_conjuncts(list(conditions))]

    def attribute_modifier(
        self,
        comparator: Comparator,
        subject: Variable,
        number: Number = Number.SINGULAR,
    ) -> Fragment:
        """
        :param comparator: The comparator on *subject*'s single-hop attribute.
        :param subject: The subject variable.
        :param number: The number the attribute noun, operator, and value agree with — singular for
            a query subject; plural for an aggregated inference antecedent (*"whose children are …"*).
        :return: The bare *"<attribute> <operator> <value>"* grouped predicate a *"whose …"* envelope
            wraps. An equality reads as a copula agreeing with *number* (so a plural subject gives
            *"are"*); any other operator keeps its comparator phrase (singular).
        """
        attribute = single_hop_attribute(comparator.left, subject)
        if comparator.operation is operator.eq and number is Number.PLURAL:
            operator_fragment = Copulas.for_number(number)
        else:
            operator_fragment = comparator_operator(
                comparator, self.context.services, compact=False
            )
        return PhraseFragment(
            parts=[
                RoleFragment.for_attribute(
                    attribute._owner_class_, attribute._attribute_name_, number=number
                ),
                operator_fragment,
                self.context.child(comparator.right, number=number, as_value=True),
            ]
        )

    def superlative_modifier(
        self, comparator: Comparator, subject: Variable
    ) -> Fragment:
        """
        :param comparator: The ``subject.<chain> == max/min(over all <Type>.<chain>)`` comparator.
        :param subject: The subject variable.
        :return: The superlative selection modifier *"with the maximum <leaf>"* / *"with the
            minimum <leaf>"*.
        """
        fold = superlative_aggregation(comparator, subject)
        leaf = fold.aggregator._leaf_attribute_
        return PhraseFragment(
            parts=[
                Prepositions.WITH.as_fragment(),
                Articles.THE.as_fragment(),
                AGGREGATION_KIND[type(fold.aggregator)].as_fragment(),
                RoleFragment.for_attribute(leaf._owner_class_, leaf._attribute_name_),
            ]
        )

    def range_modifier(self, range_fold: RangeFold, subject: Variable) -> Fragment:
        """
        :param range_fold: The folded lower/upper bound pair on *subject*'s single-hop attribute.
        :param subject: The subject variable.
        :return: The modifier *"<attribute> is between low and high"*.
        """
        attribute = single_hop_attribute(range_fold.chain_expression, subject)
        left = RoleFragment.for_attribute(
            attribute._owner_class_, attribute._attribute_name_
        )
        return build_between(
            left,
            self.context.child(range_fold.lower_expression),
            self.context.child(range_fold.upper_expression),
            compact=False,
        )

    def attribute_predicate(
        self, attribute_name: str, number: Number, value: Fragment
    ) -> Fragment:
        """
        The bare *"<attribute> <copula> <value>"* predicate (the noun and copula agree with
        *number*), with no source link on the noun — for a field binding whose owner is implicit.
        The caller gathers these under a shared *"whose …, and …"* envelope, exactly as a query
        subject restriction does.

        :param attribute_name: The attribute / field's name.
        :param number: The grammatical number the noun and copula agree with.
        :param value: The value fragment (supplied by the caller; it may itself be number-folded).
        :return: The bare predicate *"<attribute> <copula> <value>"*.
        """
        return PhraseFragment(
            parts=[
                self._attribute_noun(attribute_name, number),
                Copulas.for_number(number),
                value,
            ]
        )

    def _attribute_noun(self, name: str, number: Number) -> Fragment:
        """
        :param name: The attribute's name.
        :param number: The grammatical number to tag for inflection.
        :return: A role-tagged attribute noun (no source link — name only) tagged with *number*.
        """
        return RoleFragment.for_attribute(None, name, number)
