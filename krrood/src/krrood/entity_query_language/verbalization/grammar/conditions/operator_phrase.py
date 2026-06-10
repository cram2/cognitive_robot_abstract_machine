"""
Comparator **operator** selection — the single source of truth for the operator phrase of
a :class:`~krrood.entity_query_language.operators.comparator.Comparator` (e.g. *"is greater
than"*, *"is not equal to"*, *"is before"*).  The full *"<left> <operator> <right>"* phrase
is assembled by
:meth:`~krrood.entity_query_language.verbalization.grammar.conditions.verbalizer.ConditionVerbalizer.predicate`.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING
from typing_extensions import Optional

from krrood.entity_query_language.verbalization.chain_utils import is_temporal
from krrood.entity_query_language.verbalization.fragments.base import (
    RoleFragment,
    VerbFragment,
)
from krrood.entity_query_language.verbalization.subquery import is_calculation_value
from krrood.entity_query_language.verbalization.vocabulary.english import Operators

if TYPE_CHECKING:
    from krrood.entity_query_language.operators.comparator import Comparator
    from krrood.entity_query_language.verbalization.context import VerbalizationContext


def comparator_operator(
    comparator: Comparator,
    context: VerbalizationContext,
    *,
    negated: bool = False,
    compact: Optional[bool] = None,
) -> VerbFragment:
    """
    Select the operator fragment for *comparator* (e.g. *"is greater than"*,
    *"is not equal to"*, *"is before"*).

    Handles three orthogonal concerns declaratively:

    * **Calculation equality** — ``==`` / ``!=`` against an aggregation reads
      *"is equal to"* / *"is not equal to"* rather than the bare *"is"*.
    * **Temporality** — datetime operands select the temporal phrase variant.
    * **Negation / compactness** — flags forwarded to the vocabulary.

    :param comparator: The comparator expression.
    :param context: Shared verbalization state.
    :param negated: Outer negation (from a wrapping ``Not``).
    :param compact: Copula-less variant (HAVING clauses).  Defaults to
        ``context.compact_predicates`` when ``None``.
    :return: The operator fragment.
    :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
    """
    if compact is None:
        compact = context.compact_predicates
    op = comparator.operation

    is_calc = op in (operator.eq, operator.ne) and (
        is_calculation_value(comparator.left) or is_calculation_value(comparator.right)
    )
    if is_calc:
        calc_negated = (op is operator.ne) ^ negated
        return Operators.CALC_EQ.select(
            negated=calc_negated, compact=compact
        ).as_fragment()

    temporal = is_temporal(comparator.left) or is_temporal(comparator.right)
    try:
        return (
            Operators.from_callable(op)
            .select(negated=negated, compact=compact, temporal=temporal)
            .as_fragment()
        )
    except KeyError:
        name = comparator._name_
        return RoleFragment.for_operator(f"not {name}" if negated else name)
