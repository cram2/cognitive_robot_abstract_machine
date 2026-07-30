from __future__ import annotations

from typing_extensions import List, Optional

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.query.builders import OrderedByBuilder
from krrood.entity_query_language.query.query import Query
from krrood.entity_query_language.verbalization.fragments.base import (
    oxford_comma,
    PhraseFragment,
    VerbalizationFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    GrammaticalNumber,
    Separator,
)
from krrood.entity_query_language.verbalization.grammar.framework.assembler import (
    Assembler,
)
from krrood.entity_query_language.verbalization.grammar.clauses.planner import (
    GroupedByPlanner,
    GroupPlan,
)
from krrood.entity_query_language.verbalization.grammar.query.planner import (
    SortDirection,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Articles,
    Conjunctions,
    Copulas,
    Keywords,
    OrderingRangeWords,
)

_ORDERING_RANGE_WORDS = {
    SortDirection.ASCENDING: OrderingRangeWords.LOWEST_TO_HIGHEST,
    SortDirection.DESCENDING: OrderingRangeWords.HIGHEST_TO_LOWEST,
}
"""
Maps a :class:`SortDirection` to the ORDER BY range prose it surfaces as.
"""


class GroupedByAssembler(Assembler[Query, GroupPlan]):
    """
    *"grouped by <keys>"* — or *"and the <aggregated> are grouped by <keys>"* in a query.

    Reference: :cite:t:`reiter2000building` — aggregation / clause structuring; :cite:t:`gatt2009simplenlg`,
    SimpleNLG — surface realisation.

    A grouped set-of *report* fronts its grouping as *"For each <key>, report …"* in the query
    assembler, so this clause is used for the in-query weave (*"and the <aggregated> are grouped by
    …"*).

    >>> employee = variable(Employee, [])
    >>> verbalize_expression(
    ...     a(set_of(employee.department, sum(employee.salary)).grouped_by(employee.department))
    ... )
    'For each department, report the sum of salaries of Employees'
    """

    planner = GroupedByPlanner

    def realize(self, node: Query, plan: GroupPlan) -> VerbalizationFragment:
        """
        :param node: The query being rendered.
        :param plan: The group plan (always carries keys — :meth:`clause` gates on their presence).
        :return: *"grouped by <keys>"* — or *"and the <aggregated> are grouped by <keys>"* when
            the query also selects aggregations.

        It renders the grouping key (*"department"*) that the report is grouped on. For the grouped
        set-of report in the class example, the query assembler fronts that key as the leading *"For
        each department, report"* frame and drops this trailing clause as redundant.
        """
        groups_phrase = self._keys_phrase(plan.keys)
        if plan.aggregated and plan.weaves_aggregated:
            aggregated_phrase = oxford_comma(
                [
                    self.context.child(expression, number=GrammaticalNumber.PLURAL)
                    for expression in plan.aggregated
                ],
                Conjunctions.AND.as_fragment(),
            )
            return PhraseFragment(
                parts=[
                    Conjunctions.AND.as_fragment(),
                    Articles.THE.as_fragment(),
                    aggregated_phrase,
                    Copulas.ARE.as_fragment(),
                    Keywords.GROUPED_BY.as_fragment(),
                    groups_phrase,
                ]
            )
        return PhraseFragment(parts=[Keywords.GROUPED_BY.as_fragment(), groups_phrase])

    def clause(self, node: Query) -> Optional[VerbalizationFragment]:
        """
        :param node: The query being rendered.
        :return: The in-query GROUP BY clause, or ``None`` when there are no group keys.

        It is the in-query grouped-by gate: it renders the clause only when group keys are present.
        For the grouped report shown the keys surface as the fronted *"For each department, report"*
        in the query assembler, so this trailing clause is suppressed there:

        >>> employee = variable(Employee, [])
        >>> verbalize_expression(
        ...     an(entity(employee).grouped_by(employee.department))
        ... )
        'For each department, report all Employees'
        """
        plan = self.plan(node)
        return self.realize(node, plan) if plan.has_keys else None

    def _keys_phrase(
        self, variables: List[SymbolicExpression]
    ) -> VerbalizationFragment:
        """
        :param variables: The group-by key expressions.
        :return: The comma-joined group keys *"<key1>, <key2>, …"*.

        It assembles the list of grouping keys (*"department"*, *"name"*). For the fronted report
        shown those same keys are coordinated as *"For each department and name, report"* by the
        query assembler:

        >>> employee = variable(Employee, [])
        >>> verbalize_expression(
        ...     a(set_of(employee.department, employee.name, sum(employee.salary)).grouped_by(employee.department, employee.name))
        ... )
        'For each department and name, report the sum of salaries of Employees'
        """
        return PhraseFragment(
            parts=[self.context.child(variable) for variable in variables],
            separator=Separator.COMMA,
        )


class OrderedByAssembler(Assembler[OrderedByBuilder, None]):
    """*"ordered by <variable> from lowest to highest / from highest to lowest"*. Realisation-only
    (no plan).

    >>> employee = variable(Employee, [])
    >>> verbalize_expression(a(set_of(employee).ordered_by(employee.salary, descending=True)))
    'Report Employees ordered by their salaries from highest to lowest'
    """

    def realize(
        self, node: OrderedByBuilder, plan: None = None
    ) -> VerbalizationFragment:
        """
        :param node: The ordered-by builder (exposing ``.variable`` and ``.descending``).
        :param plan: Unused (this assembler has no plan).
        :return: The clause *"ordered by <variable> from lowest to highest"* (ascending) or
            *"… from highest to lowest"* (descending).

        It emits only the trailing ordering span — here *"ordered by their salaries from highest to
        lowest"* — and chooses the descending direction word; *"Report Employees"* is supplied by the
        query.
        """
        direction = (
            SortDirection.DESCENDING if node.descending else SortDirection.ASCENDING
        )
        return PhraseFragment(
            parts=[
                Keywords.ORDERED_BY.as_fragment(),
                self.context.child(node.variable),
                _ORDERING_RANGE_WORDS[direction].as_fragment(),
            ]
        )

    def clause(self, query: Query) -> Optional[VerbalizationFragment]:
        """
        :param query: The query being rendered.
        :return: The in-query ORDER BY clause, or ``None`` when the query is unordered.

        It is the in-query ordering gate: present an order builder and it renders the trailing
        *"ordered by their salaries from lowest to highest"* span, otherwise nothing:

        >>> employee = variable(Employee, [])
        >>> verbalize_expression(an(entity(employee).ordered_by(employee.salary)))
        'Report Employees ordered by their salaries from lowest to highest'
        """
        builder = query._ordered_by_builder_
        return self.realize(builder) if builder is not None else None


class HavingAssembler(Assembler[Query, None]):
    """
    *"where <condition>"* — a HAVING filter as a full trailing clause.

    Realisation-only (no plan).     A *grouped report* fronts its HAVING onto the group
    key instead (*"For each department whose     <aggregate> is …"*, in the query
    assembler), so this trailing form serves the non-fronted scope —     a constrained
    aggregation value (*"the sum … among <population> … where …"*).
    """

    def realize(self, node: Query, plan: None = None) -> VerbalizationFragment:
        """
        :param node: The query whose HAVING condition to render.
        :param plan: Unused (this assembler has no plan).
        :return: *"where <condition>"* — the filter as a full clause keeping its copula (*"where the
            sum is greater than 30000"*), rather than a bare *"having …"* participle that misparses as
            modifying the population.
        """
        having_fragment = self.context.child(node._having_expression_.condition)
        return PhraseFragment(parts=[Keywords.WHERE.as_fragment(), having_fragment])

    def clause(self, query: Query) -> Optional[VerbalizationFragment]:
        """
        :param query: The query being rendered.
        :return: The trailing *"where <condition>"* HAVING clause, or ``None`` when there is no
            HAVING.
        """
        return self.realize(query) if query._having_expression_ is not None else None
