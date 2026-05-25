"""
Recognising and inspecting *value subqueries* — nested
:class:`~krrood.entity_query_language.query.query.Entity` queries that denote a
value rather than a top-level request.

A nested ``Entity`` whose selected variable is an
:class:`~krrood.entity_query_language.operators.aggregators.Aggregator` is an
*aggregation value subquery*.  When it carries no ``WHERE`` / ``HAVING`` /
``GROUP BY`` it is *unconstrained* and may be collapsed to a compact aggregate
noun phrase (e.g. *"the maximum amount"*); otherwise its filter is load-bearing
and must be preserved.

These predicates are the single source of truth for this concept, shared by the
disambiguation pre-scan
(:func:`~krrood.entity_query_language.verbalization.context._build_disambiguation_map`)
and the nested-entity renderer
(:meth:`~krrood.entity_query_language.verbalization.entity_verbalizer.EntityVerbalizer.verbalize_nested`).
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from krrood.entity_query_language.operators.aggregators import Aggregator
    from krrood.entity_query_language.core.variable import Variable


def selected_aggregator(entity) -> Optional["Aggregator"]:
    """
    Return the :class:`~krrood.entity_query_language.operators.aggregators.Aggregator`
    selected by *entity*, or ``None`` when *entity* is not an
    :class:`~krrood.entity_query_language.query.query.Entity` selecting one.

    :param entity: Candidate expression.
    :returns: The selected aggregator, or ``None``.
    :rtype: ~krrood.entity_query_language.operators.aggregators.Aggregator or None
    """
    from krrood.entity_query_language.operators.aggregators import Aggregator
    from krrood.entity_query_language.query.query import Entity

    if not isinstance(entity, Entity):
        return None
    var = entity.selected_variable
    return var if isinstance(var, Aggregator) else None


def is_aggregation_subquery(entity) -> bool:
    """
    Return ``True`` when *entity* is an
    :class:`~krrood.entity_query_language.query.query.Entity` whose selected
    variable is an aggregator.

    :param entity: Candidate expression.
    :rtype: bool
    """
    return selected_aggregator(entity) is not None


def is_constrained_query(entity) -> bool:
    """
    Return ``True`` when *entity* carries a ``WHERE``, ``HAVING``, or non-empty
    ``GROUP BY`` clause.

    :param entity: An :class:`~krrood.entity_query_language.query.query.Entity`.
    :rtype: bool
    """
    if entity._where_expression_ is not None:
        return True
    if entity._having_expression_ is not None:
        return True
    grouped = entity._grouped_by_expression_
    return grouped is not None and bool(grouped.variables_to_group_by)


def is_collapsible_aggregation_subquery(entity) -> bool:
    """
    Return ``True`` when *entity* is an unconstrained aggregation subquery — one
    that may be collapsed to a compact aggregate noun phrase because it adds no
    filtering beyond the aggregation itself.

    :param entity: Candidate expression.
    :rtype: bool
    """
    return is_aggregation_subquery(entity) and not is_constrained_query(entity)


def aggregation_leaf_attribute(entity):
    """
    Return the leaf :class:`~krrood.entity_query_language.core.mapped_variable.Attribute`
    of the aggregator's child chain (e.g. the ``amount`` node behind
    ``max(t.amount_details.amount)``), or ``None`` when the aggregator's child is
    not an attribute chain (e.g. ``max(x)`` over a bare variable).

    :param entity: Candidate expression.
    :returns: The leaf attribute node, or ``None``.
    """
    from krrood.entity_query_language.core.mapped_variable import Attribute
    from krrood.entity_query_language.verbalization.chain_utils import walk_chain

    aggregator = selected_aggregator(entity)
    if aggregator is None:
        return None
    chain, _ = walk_chain(aggregator._child_)
    if chain and isinstance(chain[-1], Attribute):
        return chain[-1]
    return None


def aggregation_source_root(entity) -> Optional["Variable"]:
    """
    Return the root :class:`~krrood.entity_query_language.core.variable.Variable`
    that an aggregation subquery aggregates over (e.g. the ``BankTransaction``
    variable behind ``max(t.amount_details.amount)``), or ``None``.

    :param entity: Candidate expression.
    :returns: The chain-root variable of the aggregator's child, or ``None``.
    :rtype: ~krrood.entity_query_language.core.variable.Variable or None
    """
    from krrood.entity_query_language.core.variable import Variable
    from krrood.entity_query_language.verbalization.chain_utils import chain_root

    aggregator = selected_aggregator(entity)
    if aggregator is None:
        return None
    root = chain_root(aggregator._child_)
    return root if isinstance(root, Variable) else None
