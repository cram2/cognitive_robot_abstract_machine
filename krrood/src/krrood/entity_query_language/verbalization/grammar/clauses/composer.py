from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Optional

from krrood.entity_query_language.query.query import Query
from krrood.entity_query_language.verbalization.fragments.base import Fragment
from krrood.entity_query_language.verbalization.grammar.clauses.assembler import (
    GroupedByAssembler,
    HavingAssembler,
    OrderedByAssembler,
)
from krrood.entity_query_language.verbalization.grammar.conditions.forms import (
    as_subject_restrictions,
    RestrictionFragments,
)
from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import (
    RuleContext,
)
from krrood.entity_query_language.verbalization.grammar.query.planner import QueryPlan


@dataclass
class ClauseComposer:
    """
    The single place that knows *which* assembler renders a query/aggregation body's clauses — the
    subject restriction, GROUP BY, HAVING, ORDER BY — and *how* to call them.

    A body assembler asks the composer for a clause (a plan fact it legitimately knows it has) and
    decides its *placement* (selection modifier, block item, inline part) — its own structural
    form. It never names or constructs the per-clause assemblers itself.
    """

    context: RuleContext

    def restriction(self, plan: QueryPlan) -> Optional[RestrictionFragments]:
        """:return: The placed subject-restriction pieces (superlatives / whose / residual), or
        ``None`` when the query has no groupable subject restriction."""
        if plan.subject_restriction is None:
            return None
        return as_subject_restrictions(
            plan.subject_restriction.conditions, plan.subject, self.context
        )

    def grouped_by(self, node: Query) -> Optional[Fragment]:
        """:return: The *"grouped by …"* clause, or ``None`` when the query has no GROUP BY."""
        return GroupedByAssembler(self.context).clause(node)

    def having(self, node: Query) -> Optional[Fragment]:
        """:return: The *"having …"* clause, or ``None`` when the query has no HAVING."""
        return HavingAssembler(self.context).clause(node)

    def ordered_by(self, node: Query) -> Optional[Fragment]:
        """:return: The *"ordered by …"* clause, or ``None`` when the query has no ORDER BY."""
        return OrderedByAssembler(self.context).clause(node)
