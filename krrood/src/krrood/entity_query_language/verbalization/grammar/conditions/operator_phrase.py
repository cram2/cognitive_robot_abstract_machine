from __future__ import annotations

import operator
from typing_extensions import TYPE_CHECKING, Optional

from krrood.entity_query_language.core.expression_structure import is_temporal
from krrood.entity_query_language.verbalization.fragments.base import (
    RoleFragment,
    Fragment,
)
from krrood.entity_query_language.verbalization.fragments.features import Number
from krrood.entity_query_language.verbalization.subquery import is_calculation_value
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Logicals,
    Operators,
    predicative_operator,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.operators.comparator import Comparator
    from krrood.entity_query_language.verbalization.context import MicroplanningServices


def coindexed_operator(operation) -> Fragment:
    """
    :param operation: A foldable comparison operator (``eq``/``gt``/``lt``/``ge``/``le``).
    :return: The plural copular operator fragment for the faithful co-indexed form — *"are equal
        to"* for ``eq`` (the calculation-equality reading, since two coordinated lists cannot read a
        bare *"are"*), *"are greater than"* for ``gt``, etc. The plural copula agrees because the
        coordinated terminals are the grammatical subject.
    """
    phrase = (
        Operators.CALC_EQ
        if operation is operator.eq
        else Operators.from_callable(operation)
    )
    return predicative_operator(phrase.value.standard, Number.PLURAL)


def comparator_operator(
    comparator: Comparator,
    services: MicroplanningServices,
    *,
    negated: bool = False,
    compact: Optional[bool] = None,
    number: Number = Number.SINGULAR,
) -> Fragment:
    """
    Select the operator fragment for *comparator* (e.g. *"is greater than"*,
    *"is not equal to"*, *"is before"*).

    Handles these orthogonal concerns declaratively:

    * **Calculation equality** — ``==`` / ``!=`` against an aggregation reads
      *"is equal to"* / *"is not equal to"* rather than the bare *"is"*.
    * **Temporality** — datetime operands select the temporal phrase variant.
    * **Negation / compactness** — flags forwarded to the vocabulary.
    * **Number agreement** — the predicative (non-compact) surface factors its copula out as an
      agreeing leaf (:func:`~…vocabulary.english.predicative_operator`), so a plural subject reads
      *"are greater than"* without a duplicated plural phrase.

    :param comparator: The comparator expression.
    :param services: Shared verbalization state.
    :param negated: Outer negation (from a wrapping ``Not``).
    :param compact: Copula-less variant (HAVING clauses).  Defaults to
        ``services.configuration.compact_predicates`` when ``None``.
    :param number: The grammatical number the predicative copula agrees with.
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
        word = Operators.CALC_EQ.select(negated=calc_negated, compact=compact)
        return word.as_fragment() if compact else predicative_operator(word.text, number)

    temporal = is_temporal(comparator.left) or is_temporal(comparator.right)
    try:
        word = Operators.from_callable(operation).select(
            negated=negated, compact=compact, temporal=temporal
        )
    except KeyError:
        name = comparator._name_
        return RoleFragment.for_operator(
            f"{Logicals.NOT.text} {name}" if negated else name
        )
    return word.as_fragment() if compact else predicative_operator(word.text, number)
