from __future__ import annotations

import itertools
import operator
from collections import deque
from dataclasses import dataclass
from functools import cached_property
from typing import assert_never, List, Dict

import numpy as np

import random_events
import random_events.variable
from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import MappedVariable
from krrood.entity_query_language.core.variable import Literal
from krrood.entity_query_language.factories import ConditionType
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.operators.core_logical_operators import OR, AND
from krrood.parametrization.exceptions import (
    WhereExpressionNotInDisjunctiveNormalForm,
)
from random_events.interval import closed_open, closed, open
from random_events.product_algebra import Event, SimpleEvent


@dataclass
class WhereExpressionToRandomEventTranslator:
    """
    Class that translates a query into a random event.
    Requires that the query is in disjunctive normal form.

    Check the documentation of `is_disjunctive_normal_form` for more information.
    """

    conditions_root: ConditionType
    """
    The query in disjunctive normal form to translate.
    """

    def __post_init__(self):
        if self.conditions_root is not None and not is_disjunctive_normal_form(
            self.conditions_root
        ):
            raise WhereExpressionNotInDisjunctiveNormalForm(self.conditions_root)

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
        """

        simple_events = []

        # Traverse the logical tree starting from the conditions root
        queue = deque([self.conditions_root])

        while queue:
            expression = queue.popleft()

            if isinstance(expression, OR):
                queue.extend(expression._children_)
                continue

            elif isinstance(expression, AND):
                simple_event = self._translate_conjunction(expression)
            elif isinstance(expression, Comparator):
                simple_event = SimpleEvent.from_data(
                    {v: v.domain for v in self.variables.values()}
                )

                self._translate_comparators(
                    self._get_variable_from_comparator(expression),
                    [expression],
                    simple_event,
                )
            else:
                assert_never(expression)
            simple_events.append(simple_event)
        return Event.from_simple_sets(*simple_events).simplify()

    def _translate_conjunction(self, expression: AND) -> SimpleEvent:
        """
        Translate a conjunction expression into a random event.
        The conjunction must not contain any disjunctions anymore.

        :param expression: The conjunction expression to translate.
        :return: The random event corresponding to the conjunction.
        """
        result = SimpleEvent.from_data()

        # check that it is always a comparison between a variable and a literal
        for variable, comparators in self.comparators_grouped_by_variable(
            expression
        ).items():
            self._translate_comparators(variable, comparators, result)

        return result

    def comparators_grouped_by_variable(
        self, expression: SymbolicExpression
    ) -> Dict[random_events.variable.Variable, List[Comparator]]:
        """
        Group comparators by their variable given an expression.

        :param expression: The expression where all comparators in the descendents should be grouped by variables.
        :return: A dictionary mapping ObjectAccessVariables to lists of their corresponding comparators.
        """

        # Collect all Comparator descendants and group them by their accessed variable
        grouped: Dict[random_events.variable.Variable, List[Comparator]] = {}
        for expr in expression._descendants_:
            if not isinstance(expr, Comparator):
                continue
            key = self._get_variable_from_comparator(expr)
            grouped.setdefault(key, []).append(expr)
        return grouped

    def _translate_comparators(
        self,
        variable: random_events.variable.Variable,
        comparators: List[Comparator],
        result: SimpleEvent,
    ) -> None:
        """
        Translate comparators for a given variable into a random event in-place.

        :param variable: The variable for which to translate comparators.
        :param comparators: The comparators to translate.
        :param result: The random event to update in-place.
        :return: None
        """

        result[variable] = variable.domain
        for comparator in comparators:

            if isinstance(comparator.right._value_, type(Ellipsis)):
                continue

            match comparator.operation:
                case operator.eq:
                    self._translate_eq(comparator, variable, result)
                case operator.ne:
                    self._translate_ne(comparator, variable, result)
                case operator.gt:
                    self._translate_gt(comparator, variable, result)
                case operator.lt:
                    self._translate_lt(comparator, variable, result)
                case operator.ge:
                    self._translate_ge(comparator, variable, result)
                case operator.le:
                    self._translate_le(comparator, variable, result)
                case _:
                    assert_never(comparator.operation)

    def _translate_eq(
        self,
        comparator: Comparator,
        parametrization_variable: random_events.variable.Variable,
        result: SimpleEvent,
    ) -> None:
        result[parametrization_variable] &= parametrization_variable.make_value(
            comparator.right._value_
        )

    def _translate_ne(
        self,
        comparator: Comparator,
        parametrization_variable: random_events.variable.Variable,
        result: SimpleEvent,
    ) -> None:
        result[parametrization_variable] &= parametrization_variable.make_value(
            comparator.right._value_
        ).complement()

    def _translate_gt(
        self,
        comparator: Comparator,
        parametrization_variable: random_events.variable.Variable,
        result: SimpleEvent,
    ) -> None:
        result[parametrization_variable] &= open(comparator.right._value_, np.inf)

    def _translate_lt(
        self,
        comparator: Comparator,
        parametrization_variable: random_events.variable.Variable,
        result: SimpleEvent,
    ) -> None:
        result[parametrization_variable] &= closed_open(
            -np.inf,
            comparator.right._value_,
        )

    def _translate_le(
        self,
        comparator: Comparator,
        parametrization_variable: random_events.variable.Variable,
        result: SimpleEvent,
    ) -> None:
        result[parametrization_variable] &= closed(
            -np.inf,
            comparator.right._value_,
        )

    def _translate_ge(
        self,
        comparator: Comparator,
        parametrization_variable: random_events.variable.Variable,
        result: SimpleEvent,
    ) -> None:
        result[parametrization_variable] &= closed(
            comparator.right._value_,
            np.inf,
        )


def is_disjunctive_normal_form(condition_root: ConditionType) -> bool:
    """
    Checks if the given query is disjunctive normal form (DNF).

    A query is in DNF if the following 3 statements are true:
    1. All its comparators are literal comparators, i.e. comparators between one variable and one literal
    2. All of its conjunctions (AND statements) only have literal comparators as children
    3. There is at most one disjunction (OR statement) which has to be at the root.

    Example:
        (x > 3) is DNF

        (x > 3) & (y < 5) is DNF

        (x > 3) | (y < 5) is DNF

        (x > 3) | ((y > 5) & (z < 2)) is DNF

        (x > 3) & ((y > 5) | (z < 2)) is not DNF

    :param condition_root: The condition root of the query to check
    :return: True if the query is disjunctive normal form, False otherwise
    """

    return (
        is_disjunction_of_conjunction_of_literal_comparators(condition_root)
        or is_conjunction_of_literal_comparators(condition_root)
        or is_literal_comparator(condition_root)
    )


def is_disjunction_of_conjunction_of_literal_comparators(expression: OR) -> bool:
    """
    Checks if the given expression is a disjunction of conjunctions of literal comparators.

    :param expression: The expression to check.
    :return: True if the expression is a disjunction of conjunctions of literal comparators, False otherwise.
    """
    if not isinstance(expression, OR):
        return False
    for child in expression._children_:
        if not (
            is_disjunction_of_conjunction_of_literal_comparators(child)
            or is_conjunction_of_literal_comparators(child)
            or is_literal_comparator(child)
        ):
            return False
    return True


def is_conjunction_of_literal_comparators(expression: AND) -> bool:
    """
    Checks if the given expression is a conjunction of literal comparators.

    :param expression: The expression to check.
    :return: True if the expression is a conjunction of literal comparators, False otherwise.
    """
    if not isinstance(expression, AND):
        return False
    for child in expression._children_:
        if not (
            is_conjunction_of_literal_comparators(child) or is_literal_comparator(child)
        ):
            return False

    return True


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
