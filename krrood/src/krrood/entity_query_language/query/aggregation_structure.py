"""
Pure structural queries about aggregation subqueries and result quantifiers.

These helpers answer questions about an expression's aggregation *shape* — does it
select an aggregator, what does it aggregate over, is it a collapsible calculation —
without building anything or touching any rendering concern. They live at the query
layer (next to :class:`Entity` and the aggregators they inspect) because the facts they
expose are query-algebra knowledge usable by any consumer (evaluation, optimization,
verbalization, …), not a verbalization-only concern.
"""

from __future__ import annotations

from typing_extensions import Optional

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import Attribute
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.operators.aggregators import Aggregator
from krrood.entity_query_language.query.quantifiers import ResultQuantifier
from krrood.entity_query_language.query.query import Entity


def unwrap_result_quantifiers(expression: SymbolicExpression) -> SymbolicExpression:
    """
    Strip every enclosing result-quantifier wrapper (``An`` / ``The`` / …).

    :param expression: Any EQL expression.
    :return: The first non-quantifier descendant (or *expression* itself).
    """
    inner = expression
    while isinstance(inner, ResultQuantifier):
        inner = inner._child_
    return inner


def selected_aggregator(entity: SymbolicExpression) -> Optional[Aggregator]:
    """
    :param entity: Candidate expression.
    :return: The aggregator selected by *entity*, or ``None`` when *entity* is not an entity
        selecting one.
    """
    return entity.selected_aggregator if isinstance(entity, Entity) else None


def is_aggregation_subquery(entity: SymbolicExpression) -> bool:
    """
    :param entity: Candidate expression.
    :return: ``True`` when *entity* is an entity whose selected variable is an aggregator.
    """
    return selected_aggregator(entity) is not None


def is_calculation_value(expression: SymbolicExpression) -> bool:
    """
    :param expression: Candidate operand expression.
    :return: ``True`` when *expression* denotes a *calculation* — an aggregator or an
        aggregation sub-query (an entity selecting one), possibly wrapped in result quantifiers.
    """
    inner = unwrap_result_quantifiers(expression)
    if isinstance(inner, Aggregator):
        return True
    if isinstance(inner, Entity):
        return is_aggregation_subquery(inner)
    return False


def is_collapsible_aggregation_subquery(entity: SymbolicExpression) -> bool:
    """
    :param entity: Candidate expression.
    :return: ``True`` when *entity* is an unconstrained aggregation subquery — one that may be
        collapsed to a compact aggregate noun phrase because it adds no filtering beyond the
        aggregation itself.
    """
    return is_aggregation_subquery(entity) and not entity.is_constrained_or_grouped


def aggregation_leaf_attribute(entity: SymbolicExpression) -> Optional[Attribute]:
    """
    :param entity: Candidate expression.
    :return: The leaf attribute of the aggregator's child chain (e.g. the ``amount`` node behind
        ``max(t.amount_details.amount)``), or ``None`` when the aggregator's child is not an
        attribute chain (e.g. ``max(x)`` over a bare variable).
    """
    aggregator = selected_aggregator(entity)
    return aggregator._leaf_attribute_ if aggregator is not None else None


def aggregation_source_root(entity: SymbolicExpression) -> Optional[Variable]:
    """
    :param entity: Candidate expression.
    :return: The root variable that an aggregation subquery aggregates over (e.g. the
        ``BankTransaction`` variable behind ``max(t.amount_details.amount)``), or ``None``.
    """
    aggregator = selected_aggregator(entity)
    return aggregator._source_root_ if aggregator is not None else None
