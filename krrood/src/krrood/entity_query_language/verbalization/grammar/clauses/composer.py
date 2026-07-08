from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Optional

from krrood.entity_query_language.query.query import Query
from krrood.entity_query_language.verbalization.fragments.base import (
    VerbalizationFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    GrammaticalNumber,
)
from krrood.entity_query_language.verbalization.grammar.clauses.assembler import (
    GroupedByAssembler,
    HavingAssembler,
    OrderedByAssembler,
)
from krrood.entity_query_language.verbalization.grammar.conditions.placement import (
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
    """The per-node context (recursion entry and microplanning services) the clause assemblers are
    built with."""

    def restriction(
        self, plan: QueryPlan, number: GrammaticalNumber = GrammaticalNumber.SINGULAR
    ) -> Optional[RestrictionFragments]:
        """:return: The placed subject-restriction pieces (superlatives / whose / residual), or
        ``None`` when the query has no groupable subject restriction. The predicate agrees with
        *number* — a plural subject gives *"whose salaries are …"*.

        Within the shown sentence this routes to and supplies only the trailing restriction span
        *"whose battery is greater than 50"*; the leading *"Find a Robot"* comes from the assembler:

        >>> robot = variable(Robot, [])
        >>> verbalize_expression(an(entity(robot).where(robot.battery > 50)))
        'Find a Robot whose battery is greater than 50'
        """
        if plan.subject_restriction is None:
            return None
        return as_subject_restrictions(
            plan.subject_restriction.conditions, plan.subject, self.context, number
        )

    def grouped_by(self, node: Query) -> Optional[VerbalizationFragment]:
        """:return: The *"grouped by …"* clause, or ``None`` when the query has no GROUP BY.

        It routes to the grouped-by assembler for the grouping span; in this fronted report that span
        surfaces as the leading *"For each department, report"* frame:

        >>> employee = variable(Employee, [])
        >>> verbalize_expression(
        ...     a(set_of(employee.department, sum(employee.salary)).grouped_by(employee.department))
        ... )
        'For each department, report the sum of salaries of Employees'
        """
        return GroupedByAssembler(self.context).clause(node)

    def having(self, node: Query) -> Optional[VerbalizationFragment]:
        """:return: The trailing *"where …"* HAVING clause, or ``None`` when the query has no HAVING.

        It routes to the having assembler for the trailing *"where the sum is greater than 30000"*
        span. A grouped report fronts its HAVING onto the group key instead (*"For each department
        whose …"*), so this trailing form is used for a constrained aggregation scope.
        """
        return HavingAssembler(self.context).clause(node)

    def ordered_by(self, node: Query) -> Optional[VerbalizationFragment]:
        """:return: The *"ordered by …"* clause, or ``None`` when the query has no ORDER BY.

        It routes to the ordered-by assembler, which contributes only the trailing *"ordered by their
        salaries from lowest to highest"* span after the *"Report Employees"* selection:

        >>> employee = variable(Employee, [])
        >>> verbalize_expression(an(entity(employee).ordered_by(employee.salary)))
        'Report Employees ordered by their salaries from lowest to highest'
        """
        return OrderedByAssembler(self.context).clause(node)
