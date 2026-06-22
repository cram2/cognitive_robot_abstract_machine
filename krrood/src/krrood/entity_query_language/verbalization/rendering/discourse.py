from __future__ import annotations

import uuid
from dataclasses import dataclass

from typing_extensions import Dict, FrozenSet, Optional, Protocol, runtime_checkable

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.query.query import Entity, Query
from krrood.entity_query_language.verbalization.grammar.query.planner import (
    QueryPlanner,
)
from krrood.entity_query_language.verbalization.microplanning.microplan import Microplan
from krrood.entity_query_language.verbalization.subquery import (
    aggregation_leaf_attribute,
)


@runtime_checkable
class DiscourseView(Protocol):
    """The narrow slice of discourse information the coreference pass consumes.

    The pass depends on this, not on ``QueryPlan`` or the planners (ISP/DIP): it only needs to know
    whether a fragment's source node opens a discourse scope, and who its focus referent is.
    """

    def is_scope(self, source: Optional[SymbolicExpression]) -> bool: ...

    def focus_of(self, source: Optional[SymbolicExpression]) -> Optional[uuid.UUID]: ...

    def is_selected_quantity(self, node_id: Optional[uuid.UUID]) -> bool: ...


@dataclass(frozen=True)
class DiscourseModel:
    """
    The focus referent of every query scope, projected once from the plans.

    A query's focus — the referent a possessive pronoun resolves to (*"its"*/*"their"*) — is a
    *what to say* fact already decided by the planner: the aggregation source population for an
    aggregation value-subquery, the WHERE subject for a plain query, and nothing for a set-of. That
    single decision lives here, derived from the read model, so neither the rules nor the
    coreference pass re-derive it.
    """

    _focus_by_scope: Dict[uuid.UUID, Optional[uuid.UUID]]

    _selected_quantities: FrozenSet[uuid.UUID] = frozenset()
    """Node ids of the quantities a query *selects* — an aggregation's measured attribute (the
    ``battery`` behind ``average(m.assigned_to.battery)``). A later mention of one (a WHERE on that
    same attribute) reduces to a bare *"the battery"*."""

    @classmethod
    def from_expression(
        cls, expression: SymbolicExpression, microplan: Microplan
    ) -> DiscourseModel:
        """
        :param expression: The root EQL expression.
        :param microplan: The shared plan read model (query plans are taken from it).
        :return: A discourse model mapping each query scope to its focus referent (``None`` for a
            scope with no single subject, e.g. a set-of).
        """
        focus: Dict[uuid.UUID, Optional[uuid.UUID]] = {}
        selected_quantities: set[uuid.UUID] = set()
        for node in expression._all_expressions_:
            if isinstance(node, Query) and getattr(node, "_id_", None) is not None:
                focus[node._id_] = cls._focus(microplan.plan_for(node, QueryPlanner))
            leaf = (
                aggregation_leaf_attribute(node) if isinstance(node, Entity) else None
            )
            if leaf is not None:
                selected_quantities.add(leaf._id_)
        return cls(focus, frozenset(selected_quantities))

    @staticmethod
    def _focus(plan) -> Optional[uuid.UUID]:
        """:return: The focus referent id for a query plan — the aggregation source for an
        aggregation value-subquery, else the WHERE subject, else the single common chain root (so a
        subject-less query like a ``set_of`` still pronominalises *"its …"*), else ``None``.
        """
        if plan.is_aggregation_subquery and plan.aggregation_data is not None:
            source = plan.aggregation_data.source
            return source._id_ if source is not None else None
        if plan.subject is not None:
            return plan.subject._id_
        return plan.discourse_root

    def is_scope(self, source: Optional[SymbolicExpression]) -> bool:
        """:return: ``True`` when *source* is a query node that opens a discourse scope."""
        return getattr(source, "_id_", None) in self._focus_by_scope

    def focus_of(self, source: Optional[SymbolicExpression]) -> Optional[uuid.UUID]:
        """:return: The focus referent of *source*'s scope, or ``None``."""
        return self._focus_by_scope.get(getattr(source, "_id_", None))

    def is_selected_quantity(self, node_id: Optional[uuid.UUID]) -> bool:
        """:return: ``True`` when *node_id* is a quantity the query selects (e.g. an aggregation's
        measured attribute), so its repeat mention reduces to a bare *"the <attribute>"*.
        """
        return node_id is not None and node_id in self._selected_quantities


EMPTY_DISCOURSE = DiscourseModel({})
"""A discourse model with no scopes — for local sub-tree realisation (opaque templates)."""
