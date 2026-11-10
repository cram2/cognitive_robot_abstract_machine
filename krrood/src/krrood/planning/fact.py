"""
Facts - conditions that fix the value of ground terms - and the assignments they state.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass

from typing_extensions import Any

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.variable import InstantiatedVariable, Literal
from krrood.entity_query_language.factories import set_of
from krrood.entity_query_language.ground_term import GroundTerm, GroundTermExpression
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.operators.core_logical_operators import Not
from krrood.entity_query_language.predicate import Predicate
from krrood.planning.exceptions import (
    AssignmentBypassesSetter,
    ComparisonIsNotAnEquality,
    ConditionIsNotAFact,
    FactValueIsNotConstant,
    MissingGroundTerm,
)

# %% facts


@dataclass(frozen=True)
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
    The value every ground term of :attr:`expression` has.
    """

    @classmethod
    def from_condition(cls, condition: SymbolicExpression) -> Fact:
        """
        :param condition: ``expression == value``, a predicate, which holds, or
            :func:`~krrood.entity_query_language.factories.not_` of a predicate, which
            does not.
        :return: The fact *condition* states.
        :raises ComparisonIsNotAnEquality: If *condition* compares a ground term by other
            than equality.
        :raises ConditionIsNotAFact: If *condition* has none of these forms.
        :raises FactValueIsNotConstant: If *condition* compares with an expression
            instead of a value.
        """
        if cls._is_predicate(condition):
            return cls(condition, True)
        if isinstance(condition, Not) and cls._is_predicate(condition._child_):
            return cls(condition._child_, False)
        if not (
            isinstance(condition, Comparator)
            and isinstance(condition.left, GroundTermExpression)
        ):
            raise ConditionIsNotAFact(condition)
        if condition.operation is not operator.eq:
            raise ComparisonIsNotAnEquality(condition)
        if not isinstance(condition.right, Literal):
            raise FactValueIsNotConstant(condition)
        return cls(condition.left, condition.right._value_)

    @staticmethod
    def _is_predicate(condition: SymbolicExpression) -> bool:
        """
        :param condition: A condition given as a fact.
        :return: Whether *condition* applies a predicate.
        """
        return (
            isinstance(condition, InstantiatedVariable)
            and isinstance(condition._type_, type)
            and issubclass(condition._type_, Predicate)
        )

    def assignments(self) -> dict[GroundTerm, Any]:
        """
        :return: :attr:`value` for the ground term :attr:`expression` reads under each
            binding of its operands.
        :raises MissingGroundTerm: If one of these ground terms does not exist.
        :raises AssignmentBypassesSetter: If live objects set one of these ground terms
            through code of their own class.
        """
        operands = self.expression._ground_term_operands_
        assignments = {}
        for row in set_of(*operands).tolist():
            term = self.expression._ground_term_(
                *(row[operand] for operand in operands)
            )
            if not term.has_live_value():
                raise MissingGroundTerm(term)
            if term.setting_runs_code():
                raise AssignmentBypassesSetter(term)
            assignments[term] = self.value
        return assignments
