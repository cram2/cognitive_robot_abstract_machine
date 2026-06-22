from __future__ import annotations

import operator
from typing_extensions import TYPE_CHECKING, Optional

from krrood.entity_query_language.core.expression_structure import is_temporal
from krrood.entity_query_language.verbalization.fragments.base import (
    RoleFragment,
    Fragment,
)
from krrood.entity_query_language.verbalization.subquery import is_calculation_value
from krrood.entity_query_language.verbalization.vocabulary.english import (
    CoindexedOperators,
    Logicals,
    Operators,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.operators.comparator import Comparator
    from krrood.entity_query_language.verbalization.context import MicroplanningServices


#: Map a foldable comparison operator to its plural co-indexed phrase (*"are equal to"*, …).
_COINDEXED_OPERATOR_MAP = {
    operator.eq: CoindexedOperators.EQ,
    operator.gt: CoindexedOperators.GT,
    operator.lt: CoindexedOperators.LT,
    operator.ge: CoindexedOperators.GE,
    operator.le: CoindexedOperators.LE,
}


def coindexed_operator(operation) -> Fragment:
    """
    :param operation: A foldable comparison operator (see ``COINDEXED_OPERATORS``).
    :return: The plural copular operator fragment for the faithful co-indexed form — *"are equal
        to"* for ``eq``, *"are greater than"* for ``gt``, etc.
    """
    return _COINDEXED_OPERATOR_MAP[operation].as_fragment()


def comparator_operator(
    comparator: Comparator,
    services: MicroplanningServices,
    *,
    negated: bool = False,
    compact: Optional[bool] = None,
) -> Fragment:
    """
    Select the operator fragment for *comparator* (e.g. *"is greater than"*,
    *"is not equal to"*, *"is before"*).

    Handles three orthogonal concerns declaratively:

    * **Calculation equality** — ``==`` / ``!=`` against an aggregation reads
      *"is equal to"* / *"is not equal to"* rather than the bare *"is"*.
    * **Temporality** — datetime operands select the temporal phrase variant.
    * **Negation / compactness** — flags forwarded to the vocabulary.

    :param comparator: The comparator expression.
    :param services: Shared verbalization state.
    :param negated: Outer negation (from a wrapping ``Not``).
    :param compact: Copula-less variant (HAVING clauses).  Defaults to
        ``services.configuration.compact_predicates`` when ``None``.
    :return: The operator fragment.
    """
    if compact is None:
        compact = services.configuration.compact_predicates
    operation = comparator.operation

    is_calculation = operation in (operator.eq, operator.ne) and (
        is_calculation_value(comparator.left) or is_calculation_value(comparator.right)
    )
    if is_calculation:
        calc_negated = (operation is operator.ne) ^ negated
        return Operators.CALC_EQ.select(
            negated=calc_negated, compact=compact
        ).as_fragment()

    temporal = is_temporal(comparator.left) or is_temporal(comparator.right)
    try:
        return (
            Operators.from_callable(operation)
            .select(negated=negated, compact=compact, temporal=temporal)
            .as_fragment()
        )
    except KeyError:
        name = comparator._name_
        return RoleFragment.for_operator(
            f"{Logicals.NOT.text} {name}" if negated else name
        )
