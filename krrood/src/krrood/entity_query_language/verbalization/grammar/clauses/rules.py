from __future__ import annotations

from krrood.entity_query_language.query.operations import GroupedBy, OrderedBy
from krrood.entity_query_language.verbalization.fragments.base import Fragment
from krrood.entity_query_language.verbalization.grammar.clauses.assembler import (
    GroupedByAssembler,
    OrderedByAssembler,
)
from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import (
    PhraseRule,
    RuleContext,
)


class GroupedByRule(PhraseRule):
    """GroupedBy → *"grouped by <key1>, <key2>, …"* (or *"grouped"* when keyless).

    >>> employee = variable(Employee, [])
    >>> verbalize_expression(
    ...     a(set_of(employee.department, sum(employee.salary)).grouped_by(employee.department))
    ... )
    'For each department, report the sum of salaries of Employees'
    """

    construct = GroupedBy
    name = "grouped-by"

    def build(self, node: GroupedBy, context: RuleContext) -> Fragment:
        return GroupedByAssembler(context).assemble(node)


class OrderedByRule(PhraseRule):
    """OrderedBy → *"ordered by <variable> (ascending|descending)"*.

    >>> employee = variable(Employee, [])
    >>> verbalize_expression(a(set_of(employee).ordered_by(employee.salary, descending=True)))
    'Report Employees ordered by their salary (descending)'
    """

    construct = OrderedBy
    name = "ordered-by"

    def build(self, node: OrderedBy, context: RuleContext) -> Fragment:
        return OrderedByAssembler(context).assemble(node)
