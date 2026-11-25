"""
Facts - conditions that fix the value of ground terms - and the assignments they state.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass

from typing_extensions import Any, Iterable, Mapping

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.variable import InstantiatedVariable, Literal
from krrood.entity_query_language.ground_term import GroundTerm, GroundTermExpression
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.operators.core_logical_operators import Not
from krrood.entity_query_language.predicate import Predicate
from krrood.planning.exceptions import (
    AssignedValueHasWrongType,
    AssignmentBypassesSetter,
    ComparisonIsNotAnEquality,
    ConditionIsNotAFact,
    FactHasVariables,
    MissingGroundTerm,
)

# %% facts


@dataclass(kw_only=True, frozen=True)
class Fact:
    """
    The value of the ground term an expression reads, for every binding of the
    expression's operands.
    """

    expression: GroundTermExpression
    """
    The expression whose ground terms are given the value.
    """

    value: Any
    """
    The value the fact gives the ground term, either a constant or an expression read
    together with the operands.

    In ``block.support.clear == True`` it is ``True`` for every block; in ``block.name
    == block.support.name`` each block takes the name of its own support.
    """

    @classmethod
    def from_condition(cls, *, condition: SymbolicExpression) -> Fact:
        """
        :param condition: ``expression == value``, a predicate, which holds, or
            :func:`~krrood.entity_query_language.factories.not_` of a predicate, which
            does not.
        :return: The fact *condition* states.
        :raises ComparisonIsNotAnEquality: If *condition* compares a ground term by other
            than equality.
        :raises ConditionIsNotAFact: If *condition* has none of these forms.
        """
        if cls._is_predicate(condition=condition):
            return cls(expression=condition, value=True)
        if isinstance(condition, Not) and cls._is_predicate(
            condition=condition._child_
        ):
            return cls(expression=condition._child_, value=False)
        if not (
            isinstance(condition, Comparator)
            and isinstance(condition.left, GroundTermExpression)
        ):
            raise ConditionIsNotAFact(condition=condition)
        if condition.operation is not operator.eq:
            raise ComparisonIsNotAnEquality(condition=condition)
        if isinstance(condition.right, Literal):
            return cls(expression=condition.left, value=condition.right._value_)
        return cls(expression=condition.left, value=condition.right)

    @staticmethod
    def _is_predicate(*, condition: SymbolicExpression) -> bool:
        """
        :param condition: A condition given as a fact.
        :return: Whether *condition* applies a predicate.
        """
        return (
            isinstance(condition, InstantiatedVariable)
            and isinstance(condition._type_, type)
            and issubclass(condition._type_, Predicate)
        )

    @property
    def selected_expressions(self) -> tuple[SymbolicExpression, ...]:
        """
        :return: The expressions a query selects to give :meth:`assignments` its
            bindings: the operands of :attr:`expression`, and :attr:`value` where it is
            an expression. A binding maps each of them to its value, so their order
            carries no meaning.
        """
        operands = self.expression._ground_term_operands_
        if isinstance(self.value, SymbolicExpression):
            return (*operands, self.value)
        return operands

    @property
    def ground_term(self) -> GroundTerm:
        """
        :return: The one ground term :attr:`expression` reads, for a fact whose operands
            are all literals.
        :raises FactHasVariables: If an operand of :attr:`expression` is not a literal.
        """
        operands = self.expression._ground_term_operands_
        if not all(isinstance(operand, Literal) for operand in operands):
            raise FactHasVariables(expression=self.expression)
        return self.expression._ground_term_(
            operand_values=tuple(operand._value_ for operand in operands)
        )

    def assignments(
        self, *, bindings: Iterable[Mapping[SymbolicExpression, Any]]
    ) -> dict[GroundTerm, Any]:
        """
        :param bindings: Values of :attr:`selected_expressions`.
        :return: The value of :attr:`value` under each of *bindings*, for the ground term
            :attr:`expression` reads under it.
        :raises MissingGroundTerm: If one of these ground terms does not exist.
        :raises AssignmentBypassesSetter: If live objects set one of these ground terms
            through code of their own class.
        :raises AssignedValueHasWrongType: If a value is not of the type the class
            diagram declares for its ground term.
        """
        operands = self.expression._ground_term_operands_
        assignments = {}
        for binding in bindings:
            term = self.expression._ground_term_(
                operand_values=tuple(binding[operand] for operand in operands)
            )
            value = (
                binding[self.value]
                if isinstance(self.value, SymbolicExpression)
                else self.value
            )
            if not term.has_live_value():
                raise MissingGroundTerm(term=term)
            if term.setting_runs_code():
                raise AssignmentBypassesSetter(term=term)
            if not term.accepts_value(value=value):
                raise AssignedValueHasWrongType(term=term, value=value)
            assignments[term] = value
        return assignments
