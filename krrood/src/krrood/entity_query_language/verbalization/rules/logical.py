"""
Verbalization rules for logical operators — AND, OR, Not, and negated-variant
special cases.

The rule hierarchy:

* :class:`LogicalRule` — abstract base for any
  :class:`~krrood.entity_query_language.operators.core_logical_operators.LogicalOperator`.
* :class:`AndRule` — Oxford-comma conjunction (*"a, b, and c"*).
* :class:`RangeConjunctionRule` — range-folded *between* conjunction.
* :class:`OrRule` — *"either a, b, or c"*.
* :class:`NotRule` — generic *"not (child)"*.
* :class:`NotComparatorRule` — inline *"is not greater than"*.
* :class:`NotBoolAttrRule` — *"is not <attr>"* for boolean attributes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from krrood.entity_query_language.core.mapped_variable import Attribute, MappedVariable
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.operators.core_logical_operators import (
    AND,
    OR,
    Not,
    LogicalOperator,
    flatten_operands,
)
from krrood.entity_query_language.verbalization.chain_utils import walk_chain
from krrood.entity_query_language.verbalization.fragments.base import (
    join_with,
    oxford_and,
    PhraseFragment,
    VerbFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.factory import phrase, word
from krrood.entity_query_language.verbalization.operator_phrase import comparator_phrase
from krrood.entity_query_language.verbalization.rules.chains import verbalize_chain
from krrood.entity_query_language.verbalization.microplanning.coordination import (
    build_between,
    fold_range_pairs,
    has_pair,
    RangeFold,
)
from krrood.entity_query_language.verbalization.rule_engine import VerbalizationRule
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Conjunctions,
    Logicals,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.verbalization.context import VerbalizationContext
    from krrood.entity_query_language.verbalization.verbalizer import EQLVerbalizer


def _is_bool_attr_chain(expression) -> bool:
    """Return ``True`` when *expression* is a MappedVariable chain ending in a bool-typed Attribute."""
    if not isinstance(expression, MappedVariable):
        return False
    chain, _ = walk_chain(expression)
    return bool(chain) and isinstance(chain[-1], Attribute) and chain[-1]._type_ is bool


class LogicalRule(VerbalizationRule):
    """
    Abstract base rule: catches any
    :class:`~krrood.entity_query_language.operators.core_logical_operators.LogicalOperator`.

    Concrete subclasses (:class:`AndRule`, :class:`OrRule`, :class:`NotRule`)
    handle specific operator types and take priority over this class due to MRO-depth
    sorting in :class:`~krrood.entity_query_language.verbalization.rule_engine.RuleEngine`.
    """

    @classmethod
    def applies(cls, expression, context: VerbalizationContext) -> bool:
        """Return ``True`` for any :class:`~krrood.entity_query_language.operators.core_logical_operators.LogicalOperator`."""
        return isinstance(expression, LogicalOperator)


class AndRule(LogicalRule):
    """
    Verbalizes conjunctions (``AND(a, b, c)``) as *"a, b, and c"* using Oxford-comma style.

    Flattens nested AND chains before joining so that ``AND(AND(a,b),c)``
    produces *"a, b, and c"* rather than *"(a and b) and c"*.
    """

    @classmethod
    def applies(cls, expression, context: VerbalizationContext) -> bool:
        """Return ``True`` for :class:`~krrood.entity_query_language.operators.core_logical_operators.AND` expressions."""
        return isinstance(expression, AND)

    @classmethod
    def transform(
        cls, expression: AND, context: VerbalizationContext, verbalizer: EQLVerbalizer
    ) -> VerbFragment:
        """
        Flatten the AND chain and join with Oxford-comma *"and"*.

        :param expression: Root AND expression.
        :param context: Shared verbalization state.
        :param verbalizer: Parent verbalizer for recursive calls.
        :return: Oxford-comma joined fragment.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        parts = [
            verbalizer.build(conjunct, context)
            for conjunct in flatten_operands(expression, AND)
        ]
        if len(parts) == 1:
            return parts[0]
        return oxford_and(parts, Conjunctions.AND.as_fragment())


class RangeConjunctionRule(AndRule):
    """
    Verbalizes a conjunction that contains a lower-bound / upper-bound pair on the
    same attribute by folding it into a *"… is between lo and hi"* phrase.

    Precondition (declarative): an ``AND`` whose flattened conjuncts contain at least
    one foldable pair (:func:`~krrood.entity_query_language.verbalization.microplanning.coordination.has_pair`).
    Takes priority over :class:`AndRule`; non-range conjunctions fall through.
    The left side of the range is verbalized normally, so it still picks up a
    pronoun (*"its booking_date is between …"*) when the chain root is the subject.
    """

    @classmethod
    def applies(cls, expression, context: VerbalizationContext) -> bool:
        """Return ``True`` for an ``AND`` containing a foldable lo/hi range pair."""
        return isinstance(expression, AND) and has_pair(
            flatten_operands(expression, AND)
        )

    @classmethod
    def transform(
        cls, expression: AND, context: VerbalizationContext, verbalizer: EQLVerbalizer
    ) -> VerbFragment:
        """
        Fold range pairs and join the resulting items Oxford-comma style.

        :param expression: Root AND expression containing a range pair.
        :param context: Shared verbalization state.
        :param verbalizer: Parent verbalizer for recursive calls.
        :return: Conjunction fragment with folded *between* phrase(s).
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        parts: list[VerbFragment] = []
        for item in fold_range_pairs(flatten_operands(expression, AND)):
            if isinstance(item, RangeFold):
                parts.append(
                    build_between(
                        verbalizer.build(item.chain_expression, context),
                        verbalizer.build(item.lower_expression, context),
                        verbalizer.build(item.upper_expression, context),
                        compact=context.compact_predicates,
                    )
                )
            else:
                parts.append(verbalizer.build(item, context))
        if len(parts) == 1:
            return parts[0]
        return oxford_and(parts, Conjunctions.AND.as_fragment())


class OrRule(LogicalRule):
    """
    Verbalizes disjunctions as *"either a, b, or c"* using Oxford-comma style.

    Flattens nested OR chains before joining.
    """

    @classmethod
    def applies(cls, expression, context: VerbalizationContext) -> bool:
        """Return ``True`` for :class:`~krrood.entity_query_language.operators.core_logical_operators.OR` expressions."""
        return isinstance(expression, OR)

    @classmethod
    def transform(
        cls, expression: OR, context: VerbalizationContext, verbalizer: EQLVerbalizer
    ) -> VerbFragment:
        """
        Flatten the OR chain and produce *"either a, b, or c"*.

        :param expression: Root OR expression.
        :param context: Shared verbalization state.
        :param verbalizer: Parent verbalizer for recursive calls.
        :return: Disjunction phrase fragment.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        parts = [
            verbalizer.build(conjunct, context)
            for conjunct in flatten_operands(expression, OR)
        ]
        if len(parts) == 1:
            return parts[0]
        head_with_comma = PhraseFragment(
            parts=[join_with(parts[:-1], word(", ")), word(",")], separator=""
        )
        return phrase(
            Logicals.EITHER.as_fragment(),
            head_with_comma,
            Conjunctions.OR.as_fragment(),
            parts[-1],
        )


class NotRule(LogicalRule):
    """
    Generic negation rule: wraps the child in *"not (<child>)"*.

    :class:`NotComparatorRule` and :class:`NotBoolAttrRule` take priority when
    they match (they are deeper in the MRO hierarchy).
    """

    @classmethod
    def applies(cls, expression, context: VerbalizationContext) -> bool:
        """Return ``True`` for :class:`~krrood.entity_query_language.operators.core_logical_operators.Not` expressions."""
        return isinstance(expression, Not)

    @classmethod
    def transform(
        cls, expression: Not, context: VerbalizationContext, verbalizer: EQLVerbalizer
    ) -> VerbFragment:
        """
        Build *"not (<child>)"*.

        :param expression: Not expression.
        :param context: Shared verbalization state.
        :param verbalizer: Parent verbalizer for recursive calls.
        :return: Negation phrase fragment.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        child_fragment = verbalizer.build(expression._child_, context)
        return phrase(
            Logicals.NOT.as_fragment(),
            PhraseFragment(parts=[word("("), child_fragment, word(")")], separator=""),
        )


class NotComparatorRule(NotRule):
    """
    Negates a Comparator inline: *"a is not greater than b"* instead of *"not (a is greater than b)"*.

    Applies when the Not child is a
    :class:`~krrood.entity_query_language.operators.comparator.Comparator`.
    Uses :meth:`~krrood.entity_query_language.verbalization.vocabulary.english.Operators.from_callable`
    with ``negated=True`` to select the negated operator phrase.
    """

    @classmethod
    def applies(cls, expression, context: VerbalizationContext) -> bool:
        """Return ``True`` when the Not child is a Comparator."""
        return isinstance(expression, Not) and isinstance(
            expression._child_, Comparator
        )

    @classmethod
    def transform(
        cls, expression: Not, context: VerbalizationContext, verbalizer: EQLVerbalizer
    ) -> VerbFragment:
        """
        Build *"<left> <negated_op> <right>"*.

        :param expression: Not-wrapping-Comparator expression.
        :param context: Shared verbalization state.
        :param verbalizer: Parent verbalizer for recursive calls.
        :return: Negated comparator phrase.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        return comparator_phrase(expression._child_, context, verbalizer, negated=True)


class NotBoolAttrRule(NotRule):
    """
    Negates a boolean attribute chain: *"<nav> is not <attr>"*.

    Applies when the Not child is a
    :class:`~krrood.entity_query_language.core.mapped_variable.MappedVariable`
    chain whose terminal node is a ``bool``-typed
    :class:`~krrood.entity_query_language.core.mapped_variable.Attribute`.
    Renders via :func:`~krrood.entity_query_language.verbalization.rules.chains.verbalize_chain`
    with ``negated=True``.
    """

    @classmethod
    def applies(cls, expression, context: VerbalizationContext) -> bool:
        """Return ``True`` when the Not child is a bool-typed Attribute chain."""
        return isinstance(expression, Not) and _is_bool_attr_chain(expression._child_)

    @classmethod
    def transform(
        cls, expression: Not, context: VerbalizationContext, verbalizer: EQLVerbalizer
    ) -> VerbFragment:
        """
        Render *"<nav> is not <attr>"* for the negated boolean attribute chain.

        :param expression: Not-wrapping-bool-Attribute expression.
        :param context: Shared verbalization state.
        :param verbalizer: Parent verbalizer for recursive calls.
        :return: Predicative *"is not <attr>"* fragment.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        return verbalize_chain(expression._child_, context, verbalizer, negated=True)
