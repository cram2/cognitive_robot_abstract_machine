from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import List, Optional

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.query.operations import GroupedBy
from krrood.entity_query_language.query.query import Query, SetOf
from krrood.entity_query_language.verbalization.grammar.framework.planner import Planner


@dataclass(frozen=True)
class GroupPlan:
    """
    The GROUP BY keys and the expressions aggregated over them.
    """

    keys: List[SymbolicExpression]
    """
    The group-by key expressions (empty ⇒ a bare *"grouped"* / no clause).
    """

    aggregated: List[SymbolicExpression]
    """
    Selected expressions aggregated (not group keys) — rendered plural; empty when the
    query selects no aggregations.
    """

    weaves_aggregated: bool = False
    """
    ``True`` when the query weaves its aggregated selections into the clause (*"and the
    <aggregated> are grouped by …"*) — an entity query.

    A set-of renders its selections in the tuple, so it does not. This is the *what to
    say* decision the assembler used to take by reading the query type
    (``isinstance(node, SetOf)``) at render time.
    """

    @property
    def has_keys(self) -> bool:
        """:return: ``True`` when the query carries at least one group-by key.

        >>> GroupPlan(keys=[], aggregated=[]).has_keys
        False
        """
        return bool(self.keys)


@dataclass
class GroupedByPlanner(Planner[Query, GroupPlan]):
    """
    Decompose a query's GROUP BY into a ``GroupPlan``.

    The query-algebra facts — which keys group the results and which selections are aggregated over
    them — are owned by the EQL expressions (``GroupedBy.group_key_root_ids`` and
    ``Query.aggregated_selections``); this planner only packages them into the verbalization plan.

    Reference: :cite:t:`reiter2000building` — content/structure determination (microplanning).

    >>> employee = variable(Employee, [])
    >>> GroupedByPlanner(
    ...     a(set_of(employee.department, sum(employee.salary)).grouped_by(employee.department))
    ... ).plan().has_keys
    True
    """

    def plan(self) -> GroupPlan:
        """:return: The group plan: the group-by keys and the expressions aggregated over them.

        >>> GroupedByPlanner(an(entity(variable(Employee, [])))).plan().has_keys
        False
        """
        grouped = self._grouped_by()
        if grouped is None or not grouped.variables_to_group_by:
            return GroupPlan(keys=[], aggregated=[])
        return GroupPlan(
            keys=list(grouped.variables_to_group_by),
            aggregated=self.node.aggregated_selections(grouped.group_key_root_ids),
            weaves_aggregated=not isinstance(self.node, SetOf),
        )

    def _grouped_by(self) -> Optional[GroupedBy]:
        """:return: The query's grouped-by expression (``None`` when the query is ungrouped).

        >>> GroupedByPlanner(an(entity(variable(Employee, []))))._grouped_by() is None
        True
        """
        return self.node._grouped_by_expression_
