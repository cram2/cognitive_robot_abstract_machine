from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto

from typing_extensions import List, Optional

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.expression_structure import (
    root_variable_ids,
    walk_chain,
)
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


class ReportKind(Enum):
    """What makes a query a report (a presentation of results) rather than a search."""

    AGGREGATION = auto()
    """The selection computes aggregates — *"Report the sum …"* / *"For each …, report …"*."""
    ORDERING = auto()
    """An unranked ordered listing — *"Report Employees ordered by their salary"* (plural, because
    ordering ranges over all the results)."""


class RankingDirection(Enum):
    """The ordering direction a ``limit`` selects over, if any."""

    NONE = auto()
    """A ``limit`` with no ``ordered_by`` — the first *n* in natural order."""
    ASCENDING = auto()
    """``ordered_by(…, descending=False)`` — the lowest / bottom *n*."""
    DESCENDING = auto()
    """``ordered_by(…, descending=True)`` — the highest / top *n*."""


class RankingKeyRelation(Enum):
    """How the order key relates to the selected variable — it changes the surface form."""

    SELF = auto()
    """The order key *is* the selected variable (*"the highest int"*)."""
    ATTRIBUTE = auto()
    """The order key is an attribute chain of the selection (*"… with the highest salary"*)."""
    OTHER = auto()
    """The order key roots at a different variable — no clean noun-relative form."""


@dataclass(frozen=True)
class RankingPlan:
    """The *what to say* decomposition of a query's ``limit`` (with optional ordering): how many,
    in which direction, and how the order key relates to the selection. Present only when the query
    has a ``limit``; the surface form is the ranking registry's concern at render time.
    """

    n: int
    """The limit (always ``>= 1``)."""

    direction: RankingDirection
    """The ordering direction, or ``NONE`` for a bare ``limit``."""

    relation: RankingKeyRelation
    """How the order key relates to the selection (``SELF`` when there is no ordering)."""

    order_key: Optional[SymbolicExpression]
    """The ``ordered_by`` key expression, or ``None`` for a bare ``limit``."""


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
class ReportPlan:
    """A query that *presents* its results — a report/listing — rather than *searching* for a match.

    Report-ness and conditions are orthogonal: a report may still be filtered (it just presents the
    filtered results). The opening verb is *"Report"* rather than *"Find"*; :attr:`kind` selects the
    body.
    """

    kind: ReportKind
    """Why the query is a report — it drives the body (aggregate columns vs. an ordered listing)."""

    group_keys: List[SymbolicExpression] = field(default_factory=list)
    """The GROUP BY keys, rendered as the *"for each <key>"* frame (``AGGREGATION`` only)."""

    columns: List[SymbolicExpression] = field(default_factory=list)
    """The reported aggregate selections, group keys removed (``AGGREGATION`` only)."""

    @property
    def is_grouped(self) -> bool:
        """:return: ``True`` when the report has a GROUP BY (the *"for each …"* frame applies)."""
        return bool(self.group_keys)


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

    ranking: Optional[RankingPlan]
    """The ``limit`` (+ ordering) decomposition, or ``None`` when the query has no ``limit``."""

    discourse_root: Optional[uuid.UUID]
    """The single variable every chain in this query's scope roots at, or ``None`` when the roots
    are not unique. Used as the pronominalisation focus (*"its …"*) for a query that has no
    restriction subject (e.g. a ``set_of``); kept separate from :attr:`subject` so it never triggers
    *"whose"*-folding."""

    report: Optional[ReportPlan]
    """The report decomposition when the set-of selection computes aggregates, else ``None`` (a
    plain *"Find …"* search)."""


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
        ranking = self._ranking()
        return QueryPlan(
            kind=self._kind(),
            is_the=self._is_the(),
            selected_type=self._selected_type(),
            subject=self._subject(),
            subject_restriction=self._subject_restriction(),
            where_condition=self._where_condition(),
            is_aggregation_subquery=is_aggregation_subquery(self.node),
            aggregation_data=self._aggregation_data(),
            ranking=ranking,
            discourse_root=self._discourse_root(),
            report=self._report(ranking),
        )

    def _report(self, ranking: Optional[RankingPlan]) -> Optional[ReportPlan]:
        """:return: The report decomposition when the query *presents* results rather than searching
        — a set-of computing aggregates, or any query with an ``ordered_by`` listing — else ``None``.

        A ranked query (``limit``) keeps the *"Find the top three …"* form — the ranking pre-head is
        a distinct, count-bearing surface — so it is never treated as a report.
        """
        if ranking is not None:
            return None
        aggregation = self._aggregation_report()
        if aggregation is not None:
            return aggregation
        if self.node._ordered_by_builder_ is not None:
            return ReportPlan(kind=ReportKind.ORDERING)
        return None

    def _aggregation_report(self) -> Optional[ReportPlan]:
        """:return: The ``AGGREGATION`` report for a set-of selecting at least one aggregate (its
        GROUP BY keys split out of the reported columns), else ``None``."""
        if not isinstance(self.node, SetOf):
            return None
        selections = list(self.node._selected_variables_)
        if not any(isinstance(selection, Aggregator) for selection in selections):
            return None
        grouped = self.node._grouped_by_expression_
        keys = list(grouped.variables_to_group_by) if grouped is not None else []
        key_ids = {key._id_ for key in keys}
        columns = [s for s in selections if s._id_ not in key_ids]
        return ReportPlan(kind=ReportKind.AGGREGATION, group_keys=keys, columns=columns)

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

    # ── ranking (limit + ordering) ───────────────────────────────────────────

    def _ranking(self) -> Optional[RankingPlan]:
        """:return: The ``limit`` (+ ordering) decomposition, or ``None`` when the query has no
        ``limit`` (ordering without a limit keeps the standalone *"ordered by …"* clause).
        """
        limit = getattr(self.node, "_limit_", None)
        if limit is None:
            return None
        builder = getattr(self.node, "_ordered_by_builder_", None)
        if builder is None:
            return RankingPlan(
                n=limit,
                direction=RankingDirection.NONE,
                relation=RankingKeyRelation.SELF,
                order_key=None,
            )
        direction = (
            RankingDirection.DESCENDING
            if builder.descending
            else RankingDirection.ASCENDING
        )
        return RankingPlan(
            n=limit,
            direction=direction,
            relation=self._key_relation(builder.variable),
            order_key=builder.variable,
        )

    def _key_relation(self, order_key: SymbolicExpression) -> RankingKeyRelation:
        """:return: How *order_key* relates to the selected variable — ``SELF`` (the key is the
        selection), ``ATTRIBUTE`` (a chain on it), or ``OTHER`` (a different root)."""
        chain, root = walk_chain(order_key)
        selected_id = getattr(self._selected, "_id_", None)
        if selected_id is None or getattr(root, "_id_", None) != selected_id:
            return RankingKeyRelation.OTHER
        return RankingKeyRelation.ATTRIBUTE if chain else RankingKeyRelation.SELF

    # ── discourse focus (pronominalisation) ──────────────────────────────────

    def _discourse_root(self) -> Optional[uuid.UUID]:
        """:return: The single variable every chain in this query roots at, or ``None`` when the
        roots are not unique — the pronominalisation focus for a query with no restriction subject
        (e.g. a ``set_of``), so its chains read *"its …"* instead of restating the full root.
        """
        roots = root_variable_ids(self.node._all_expressions_)
        return next(iter(roots)) if len(roots) == 1 else None
