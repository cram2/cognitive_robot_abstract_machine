from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from typing_extensions import List, Optional

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import MappedVariable
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.operators.aggregators import Aggregator
from krrood.entity_query_language.operators.core_logical_operators import (
    AND,
    flatten_operands,
)
from krrood.entity_query_language.query.quantifiers import The
from krrood.entity_query_language.query.query import Entity, Query, SetOf
from krrood.entity_query_language.verbalization.grammar.framework.planner import Planner
from krrood.entity_query_language.verbalization.grammar.conditions.restriction import (
    restriction_subject,
)
from krrood.entity_query_language.verbalization.subquery import (
    aggregation_leaf_attribute,
    aggregation_source_root,
    is_aggregation_subquery,
    selected_aggregator,
)
from krrood.entity_query_language.verbalization.vocabulary.english import FallbackNouns


class SelectionKind(Enum):
    """The structural shape of a query's selection."""

    ENTITY_SELECTOR = auto()
    """The selected variable is itself an entity (a sub-query selector)."""
    EMPTY = auto()
    """No selected variable — the fallback *"entity"* form."""
    SUBJECT = auto()
    """A plain variable / aggregator selection that can carry restrictions."""
    SET_OF = auto()
    """A tuple selection over several variables."""


@dataclass(frozen=True)
class RestrictionPlan:
    """A subject's WHERE condition decomposed into its conjuncts — the *content* of the restriction.

    Everything downstream of the raw conjuncts (range-folding a bound pair, choosing each conjunct's
    surface form and slot) is the placement registry's concern at render time, so the plan carries
    only the flattened conjuncts, neither folded nor placed.
    """

    conditions: List[SymbolicExpression] = field(default_factory=list)
    """The subject's WHERE conjuncts (an ``AND`` flattened to a list, not range-folded)."""


@dataclass(frozen=True)
class AggregationData:
    """A collapsed aggregation subquery (*"the maximum amount among …"*)."""

    aggregator: Optional[Aggregator]
    """The selected aggregator."""

    leaf: Optional[MappedVariable]
    """The leaf attribute of the aggregator chain, or ``None``."""

    is_constrained_or_grouped: bool
    """``True`` when the aggregation is constrained by a WHERE/HAVING clause."""

    source: Optional[Variable]
    """The chain-root source variable aggregated over, or ``None``."""


@dataclass(frozen=True)
class QueryPlan:
    """Complete *what to say* decomposition of a query (the plan)."""

    kind: SelectionKind
    """The structural shape of the selection."""

    is_the: bool
    """``True`` when the query is uniqueness-quantified (*"the"* rather than *"a"*)."""

    selected_type: str
    """Display name of the selected type (e.g. ``"Robot"``)."""

    subject: Optional[Variable]
    """The variable the WHERE restricts, or ``None`` when there is no groupable subject."""

    subject_restriction: Optional[RestrictionPlan]
    """The subject's WHERE, range-folded into conjuncts for the assembler to place, or ``None``."""

    where_condition: Optional[SymbolicExpression]
    """The query's raw WHERE condition, or ``None``."""

    is_aggregation_subquery: bool
    """``True`` when the query is an aggregation value-subquery."""

    aggregation_data: Optional[AggregationData]
    """The collapsed aggregation details when this is an aggregation subquery, else ``None``."""


@dataclass
class QueryPlanner(Planner[Query, QueryPlan]):
    """
    Decompose an entity or set-of query into a ``QueryPlan`` (the *what to say* decisions): the
    selection shape, the definiteness (``is_the``), the restriction subject and its WHERE
    partition (grouped *"whose …"* predicates vs. residual *"such that …"*), and whether the
    entity is an aggregation value-subquery.

    Reference: Reiter & Dale (2000) — content/structure determination (microplanning).

    >>> QueryPlanner(entity(variable(Robot, []))).plan().selected_type
    'Robot'
    """

    def plan(self) -> QueryPlan:
        """:return: The plan: selection shape, definiteness, restriction partition, aggregation."""
        self.node.build()
        return QueryPlan(
            kind=self._kind(),
            is_the=self._is_the(),
            selected_type=self._selected_type(),
            subject=self._subject(),
            subject_restriction=self._subject_restriction(),
            where_condition=self._where_condition(),
            is_aggregation_subquery=is_aggregation_subquery(self.node),
            aggregation_data=self._aggregation_data(),
        )

    # ── selection shape ──────────────────────────────────────────────────────

    @property
    def _selected(self) -> Optional[SymbolicExpression]:
        return getattr(self.node, "selected_variable", None)

    def _kind(self) -> SelectionKind:
        if isinstance(self.node, SetOf):
            return SelectionKind.SET_OF
        selected = self._selected
        if isinstance(selected, Entity):
            return SelectionKind.ENTITY_SELECTOR
        if selected is None:
            return SelectionKind.EMPTY
        return SelectionKind.SUBJECT

    def _is_the(self) -> bool:
        builder = getattr(self.node, "_quantifier_builder_", None)
        return builder is not None and builder.type is The

    def _selected_type(self) -> str:
        selected = self._selected
        if selected is not None and getattr(selected, "_type_", None):
            return selected._type_.__name__
        return FallbackNouns.ENTITY.text

    # ── subject restriction (WHERE partition) ────────────────────────────────

    def _subject(self) -> Optional[Variable]:
        if not isinstance(self.node, Entity):
            return None
        return restriction_subject(self.node, self._selected)

    def _subject_restriction(self) -> Optional[RestrictionPlan]:
        """:return: The subject's WHERE flattened into conjuncts (folding and placement are the
        registry's concern), or ``None`` when there is no groupable subject or no WHERE.
        """
        condition = self._where_condition()
        subject = self._subject()
        if condition is None or subject is None:
            return None
        return RestrictionPlan(conditions=flatten_operands(condition, AND))

    # ── clauses ──────────────────────────────────────────────────────────────

    def _where_condition(self) -> Optional[SymbolicExpression]:
        where = getattr(self.node, "_where_expression_", None)
        return where.condition if where is not None else None

    # ── aggregation value-subquery ───────────────────────────────────────────

    def _aggregation_data(self) -> Optional[AggregationData]:
        if not is_aggregation_subquery(self.node):
            return None
        return AggregationData(
            aggregator=selected_aggregator(self.node),
            leaf=aggregation_leaf_attribute(self.node),
            is_constrained_or_grouped=self.node.is_constrained_or_grouped,
            source=aggregation_source_root(self.node),
        )
