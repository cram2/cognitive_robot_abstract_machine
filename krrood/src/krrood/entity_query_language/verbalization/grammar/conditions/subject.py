from __future__ import annotations

from abc import abstractmethod
from typing_extensions import Optional

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.operators.aggregators import Aggregator
from krrood.entity_query_language.verbalization.grammar.framework.specificity import (
    SpecificityRule,
)
from krrood.entity_query_language.query.aggregation_structure import (
    aggregation_source_root,
)

# %% restriction-subject rules (which variable does the WHERE restrict?)


class RestrictionSubjectRule(SpecificityRule):
    """
    Resolve which variable a query's selection restricts, so the selection's ``WHERE`` can fold
    into a post-nominal *"whose …"* modifier on it.  A selection matched by no rule has no
    groupable subject — its ``WHERE`` stays a full *"such that …"* clause.
    """

    @classmethod
    @abstractmethod
    def applies(
        cls, expression: SymbolicExpression, selected_variable: SymbolicExpression
    ) -> bool:
        """
        :param expression: The query whose restriction subject to resolve.
        :param selected_variable: The query's selected variable.
        :return: ``True`` when this rule can name the restriction subject of *expression*.

        >>> robot = variable(Robot, [])
        >>> verbalize_expression(an(entity(robot).where(robot.battery > 50)))
        'Find a Robot whose battery is greater than 50'
        """

    @classmethod
    @abstractmethod
    def subject(
        cls, expression: SymbolicExpression, selected_variable: SymbolicExpression
    ) -> Optional[Variable]:
        """
        :param expression: The query whose restriction subject to resolve.
        :param selected_variable: The query's selected variable.
        :return: The variable the ``WHERE`` restricts.

        >>> robot = variable(Robot, [])
        >>> query = an(entity(robot).where(robot.battery > 50))
        >>> restriction_subject(query, query.selected_variable) is robot
        True
        """


class SelectedVariableSubjectRule(RestrictionSubjectRule):
    """The selection is a plain variable → it is its own subject."""

    @classmethod
    def applies(
        cls, expression: SymbolicExpression, selected_variable: SymbolicExpression
    ) -> bool:
        """Fires when the selection is a plain variable.

        >>> robot = variable(Robot, [])
        >>> query = an(entity(robot).where(robot.battery > 50))
        >>> SelectedVariableSubjectRule.applies(query, query.selected_variable)
        True
        """
        return isinstance(selected_variable, Variable)

    @classmethod
    def subject(
        cls, expression: SymbolicExpression, selected_variable: SymbolicExpression
    ) -> Optional[Variable]:
        """The plain selection is its own restriction subject.

        >>> robot = variable(Robot, [])
        >>> query = an(entity(robot).where(robot.battery > 50))
        >>> SelectedVariableSubjectRule.subject(query, query.selected_variable) is robot
        True
        """
        return selected_variable


class AggregationSourceSubjectRule(RestrictionSubjectRule):
    """
    The selection aggregates over a single source variable's chain (e.g.
    ``max(t.amount_details.amount)``); the ``WHERE`` restricts that aggregated entity,
    whose noun ends the selection so a *"whose …"* modifier attaches grammatically.

    >>> employee = variable(Employee, [])
    >>> verbalize_expression(
    ...     an(entity(max(employee.salary)).where(employee.department == 'Sales'))
    ... )
    "Find the maximum of the salary of an Employee whose department is 'Sales'"
    """

    @classmethod
    def applies(
        cls, expression: SymbolicExpression, selected_variable: SymbolicExpression
    ) -> bool:
        """Fires when the selection aggregates over a single source variable's chain.

        >>> employee = variable(Employee, [])
        >>> query = an(entity(max(employee.salary)).where(employee.department == 'Sales'))
        >>> AggregationSourceSubjectRule.applies(query, query.selected_aggregator)
        True
        """
        return (
            isinstance(selected_variable, Aggregator)
            and aggregation_source_root(expression) is not None
        )

    @classmethod
    def subject(
        cls, expression: SymbolicExpression, selected_variable: SymbolicExpression
    ) -> Optional[Variable]:
        """The aggregated source variable is the restriction subject.

        >>> employee = variable(Employee, [])
        >>> query = an(entity(max(employee.salary)).where(employee.department == 'Sales'))
        >>> AggregationSourceSubjectRule.subject(query, query.selected_aggregator) is employee
        True
        """
        return aggregation_source_root(expression)


def restriction_subject(
    expression: SymbolicExpression, selected_variable: SymbolicExpression
) -> Optional[Variable]:
    """
    :param expression: The query whose restriction subject to resolve.
    :param selected_variable: The query's selected variable.
    :return: The variable a selection's ``WHERE`` restricts (most-specific rule wins), or
        ``None``.

    >>> robot = variable(Robot, [])
    >>> query = an(entity(robot).where(robot.battery > 50))
    >>> restriction_subject(query, query.selected_variable) is robot
    True
    """
    chosen = RestrictionSubjectRule.most_applicable(expression, selected_variable)
    return chosen.subject(expression, selected_variable) if chosen is not None else None
