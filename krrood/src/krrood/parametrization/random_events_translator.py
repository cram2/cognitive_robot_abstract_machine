from __future__ import annotations

import itertools
import operator
from dataclasses import dataclass
from functools import cached_property
from typing import assert_never, Dict

import numpy as np

import random_events
import random_events.variable
from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import MappedVariable
from krrood.entity_query_language.core.variable import Literal
from krrood.entity_query_language.factories import ConditionType
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.operators.core_logical_operators import OR, AND, Not
from krrood.entity_query_language.operators.logical_quantifiers import (
    QuantifiedConditional,
)
from krrood.parametrization.exceptions import (
    WhereExpressionHasNoRandomEventRepresentation,
    WhereExpressionIsFirstOrder,
)
from random_events.interval import closed_open, closed, open
from random_events.product_algebra import Event, SimpleEvent


# %% translating a where expression into a random event


@dataclass
class WhereExpressionToRandomEventTranslator:
    """
    Class that translates a query into a random event.

    Conjunction, disjunction and negation are translated into the corresponding
    operations of the product algebra, which normalizes the result itself, so the query
    may have any propositional shape. Quantified conditions are refused, since a product
    algebra constrains a fixed set of variables and states nothing about the objects a
    quantifier ranges over.
    """

    conditions_root: ConditionType
    """
    The query to translate.
    """

    @cached_property
    def variables(self) -> Dict[MappedVariable, random_events.variable.Variable]:
        result = {}
        if self.conditions_root is None:
            return result
        for comparator in itertools.chain(
            [self.conditions_root], self.conditions_root._descendants_
        ):
            if not is_literal_comparator(comparator):
                continue
            result[comparator.left] = (
                random_events.variable.variable_from_name_and_type(
                    comparator.left._name_, comparator.left._type_
                )
            )
        return result

    def _get_variable_from_comparator(
        self, expression: Comparator
    ) -> random_events.variable.Variable:
        """
        :param expression: The comparator to get the parameterization variable from.
        :return: The parameterization variable that corresponds to the comparator's left side.
        """
        return self.variables[expression.left]

    def translate(self) -> Event:
        """
        :return: The random event that corresponds to the query.
        :raises WhereExpressionIsFirstOrder: When the query quantifies over a variable.
        :raises WhereExpressionHasNoRandomEventRepresentation: When a part of the query
            constrains nothing that a random event variable stands for.
        """
        return self._translate_expression(self.conditions_root).simplify()

    def _translate_expression(self, expression: SymbolicExpression) -> Event:
        """
        Translate a where expression of any propositional shape into a random event.

        :param expression: The expression to translate.
        :return: The random event that holds exactly where the expression holds.
        :raises WhereExpressionIsFirstOrder: When the expression quantifies over a
            variable.
        :raises WhereExpressionHasNoRandomEventRepresentation: When the expression
            constrains nothing that a random event variable stands for.
        """
        if isinstance(expression, QuantifiedConditional):
            raise WhereExpressionIsFirstOrder(expression)

        if isinstance(expression, AND):
            return self._translate_expression(
                expression.left
            ) & self._translate_expression(expression.right)

        if isinstance(expression, OR):
            return self._translate_expression(
                expression.left
            ) | self._translate_expression(expression.right)

        if isinstance(expression, Not):
            return self._translate_expression(expression._child_).complement()

        if isinstance(expression, Literal):
            return (
                self.unconstrained_event()
                if expression._value_
                else self.impossible_event()
            )

        if not is_literal_comparator(expression):
            raise WhereExpressionHasNoRandomEventRepresentation(expression)

        return self._translate_comparator(expression)

    def _translate_comparator(self, comparator: Comparator) -> Event:
        """
        Translate a comparison between a variable and a literal into a random event.

        The event is built over every variable of the query rather than over the
        compared one alone, because the product algebra combines events by the variables
        they already carry and would otherwise drop the variables an operand does not
        mention.

        :param comparator: The comparison to translate.
        :return: The random event that holds exactly where the comparison holds.
        """
        result = self.unconstrained_simple_event()
        variable = self._get_variable_from_comparator(comparator)

        if isinstance(comparator.right._value_, type(Ellipsis)):
            return result.as_composite_set()

        match comparator.operation:
            case operator.eq:
                self._translate_equal(comparator, variable, result)
            case operator.ne:
                self._translate_not_equal(comparator, variable, result)
            case operator.gt:
                self._translate_greater_than(comparator, variable, result)
            case operator.lt:
                self._translate_less_than(comparator, variable, result)
            case operator.ge:
                self._translate_greater_than_or_equal(comparator, variable, result)
            case operator.le:
                self._translate_less_than_or_equal(comparator, variable, result)
            case _:
                assert_never(comparator.operation)

        return result.as_composite_set()

    def unconstrained_simple_event(self) -> SimpleEvent:
        """
        :return: The simple event that leaves every variable of the query on its full
            domain.
        """
        return SimpleEvent.from_data(
            {variable: variable.domain for variable in self.variables.values()}
        )

    def unconstrained_event(self) -> Event:
        """
        :return: The event that holds everywhere, the identity of intersection.
        """
        return self.unconstrained_simple_event().as_composite_set()

    def impossible_event(self) -> Event:
        """
        :return: The event that holds nowhere, the identity of union.
        """
        return self.unconstrained_event().complement()

    def _translate_equal(
        self,
        comparator: Comparator,
        parametrization_variable: random_events.variable.Variable,
        result: SimpleEvent,
    ) -> None:
        result[parametrization_variable] &= parametrization_variable.make_value(
            comparator.right._value_
        )

    def _translate_not_equal(
        self,
        comparator: Comparator,
        parametrization_variable: random_events.variable.Variable,
        result: SimpleEvent,
    ) -> None:
        result[parametrization_variable] &= parametrization_variable.make_value(
            comparator.right._value_
        ).complement()

    def _translate_greater_than(
        self,
        comparator: Comparator,
        parametrization_variable: random_events.variable.Variable,
        result: SimpleEvent,
    ) -> None:
        result[parametrization_variable] &= open(comparator.right._value_, np.inf)

    def _translate_less_than(
        self,
        comparator: Comparator,
        parametrization_variable: random_events.variable.Variable,
        result: SimpleEvent,
    ) -> None:
        result[parametrization_variable] &= closed_open(
            -np.inf,
            comparator.right._value_,
        )

    def _translate_less_than_or_equal(
        self,
        comparator: Comparator,
        parametrization_variable: random_events.variable.Variable,
        result: SimpleEvent,
    ) -> None:
        result[parametrization_variable] &= closed(
            -np.inf,
            comparator.right._value_,
        )

    def _translate_greater_than_or_equal(
        self,
        comparator: Comparator,
        parametrization_variable: random_events.variable.Variable,
        result: SimpleEvent,
    ) -> None:
        result[parametrization_variable] &= closed(
            comparator.right._value_,
            np.inf,
        )


# %% shape checks on where expressions


def is_literal_comparator(expression: Comparator) -> bool:
    """
    Checks if the given expression is a literal comparator.

    :param expression: The expression to check.
    :return: True if the expression is a literal comparator, False otherwise.
    """
    if not isinstance(expression, Comparator):
        return False
    if not isinstance(expression.left, MappedVariable):
        return False
    if not isinstance(expression.right, Literal):
        return False
    return True
