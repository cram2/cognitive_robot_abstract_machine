"""
This module turns the EQL condition a task states into a
:mod:`random_events` event.

Each proposition a condition states becomes one boolean symbolic variable meaning "this
proposition holds". What the proposition *means* is deliberately ignored here: bounding
a distance and asserting a containment both become one variable, so what is represented
is the condition's logical skeleton alone. Refining a bounded value into the interval it
actually describes is a separate question, asked separately.

An :class:`~random_events.product_algebra.Event` is a disjunctive normal form, so the
number of terms a condition needs is a property of that skeleton and is what the survey
reports as the size of a condition's representation.

.. note::
    The product algebra is propositional: an event constrains a fixed set of variables.
    A condition quantifying over the objects a scene provides states something about those
    objects instead, and is refused rather than instantiated over some chosen number of
    them, since the number instantiated -- and not the condition -- would decide the size
    of the result.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.operators.core_logical_operators import AND, OR, Not
from krrood.entity_query_language.operators.logical_quantifiers import (
    QuantifiedConditional,
)
from krrood.exceptions import DataclassException
from krrood.parametrization.exceptions import WhereExpressionIsFirstOrder
from random_events.product_algebra import Event, SimpleEvent
from random_events.set import Set
from random_events.variable import Symbolic

from experiments.random_events_experiments.task_condition_survey.task_conditions import (
    ConstantCondition,
    StatedProposition,
    StatedTaskCondition,
)

# %% conditions outside the propositional algebra


@dataclass
class ConditionNotFullyRead(DataclassException):
    """
    Raised when a condition containing a part the survey could not read is asked to be
    represented, since what that part constrains is unknown.
    """

    unread: str
    """
    What was found in place of a condition the survey represents.
    """

    def error_message(self) -> str:
        return (
            f"{self.unread} constrains something unknown, so it has no representation."
        )

    def suggest_correction(self) -> str:
        return (
            "Survey only the conditions that were read in full, so a partial reading is "
            "never reported as a complete one."
        )


# %% representing a condition as a random event


@dataclass
class ConditionEventTranslator:
    """
    Represents an EQL condition as a random event over one boolean variable per
    proposition the condition states.

    Every leaf is built over the whole variable set rather than over the single variable
    it constrains, because the product algebra's operations combine events by the
    variables they already carry and would otherwise silently drop the variables an
    operand does not mention.

    ..note::
        :class:`~krrood.parametrization.random_events_translator.WhereExpressionToRandomEventTranslator`
        does not translate these conditions. Its leaves are comparisons between an
        attribute of a queried object and a literal value, which it represents as the
        interval or the set of states that attribute is left with. A condition read from
        a suite's source has no such leaf: it applies a relation, or compares a value the
        suite measures against whatever bound the source writes, a name as often as a
        number. What is asked of it here is the size of its logical skeleton, for which
        one boolean variable per proposition is enough.
    """

    def translate(self, condition: StatedTaskCondition) -> Event:
        """
        :param condition: The condition to represent.
        :return: The event holding exactly where the condition holds.
        :raises ConditionNotFullyRead: When part of the condition was never read.
        :raises WhereExpressionIsFirstOrder: When the condition quantifies.
        """
        unread = condition.unread_parts()
        if unread:
            raise ConditionNotFullyRead(unread=", ".join(unread))
        variables = self.variables_of(condition)
        return self.event_of(condition.expression, variables)

    def simple_set_count(self, condition: StatedTaskCondition) -> int:
        """
        :param condition: The condition to represent.
        :return: Number of terms the condition's disjunctive normal form needs.
        """
        return len(self.translate(condition).simple_sets)

    @staticmethod
    def variables_of(condition: StatedTaskCondition) -> list[Symbolic]:
        """
        :param condition: A propositional condition.
        :return: One boolean variable per proposition the condition states, in first
            statement order.
        :raises WhereExpressionIsFirstOrder: When the condition quantifies.
        """
        quantifications = condition.quantifications()
        if quantifications:
            raise WhereExpressionIsFirstOrder(quantifications[0])
        names = []
        for proposition in condition.propositions():
            if proposition.variable_name not in names:
                names.append(proposition.variable_name)
        return [
            Symbolic(name, domain=Set.from_iterable([True, False])) for name in names
        ]

    def event_of(
        self, expression: SymbolicExpression, variables: list[Symbolic]
    ) -> Event:
        """
        :param expression: A propositional EQL condition.
        :param variables: The variables the event is represented over.
        :return: The event holding exactly where the condition holds.
        """
        if isinstance(expression, AND):
            return self.event_of(expression.left, variables) & self.event_of(
                expression.right, variables
            )
        if isinstance(expression, OR):
            return self.event_of(expression.left, variables) | self.event_of(
                expression.right, variables
            )
        if isinstance(expression, Not):
            return self.event_of(expression._child_, variables).complement()
        if isinstance(expression, QuantifiedConditional):
            raise WhereExpressionIsFirstOrder(expression)

        constant = ConstantCondition.stated_by(expression)
        if constant is not None:
            return (
                self.everything(variables)
                if constant is ConstantCondition.ALWAYS
                else self.nothing(variables)
            )

        proposition = StatedProposition.stated_by(expression)
        if proposition is not None:
            return self.proposition_holds(proposition.variable_name, variables)

        raise ConditionNotFullyRead(unread=type(expression).__name__)

    @staticmethod
    def proposition_holds(variable_name: str, variables: list[Symbolic]) -> Event:
        """
        :param variable_name: What the proposition constrains.
        :param variables: The variables the event is represented over.
        :return: The event where that proposition holds and every other is unconstrained.
        """
        return SimpleEvent.from_data(
            {
                variable: (
                    variable.make_value(True)
                    if variable.name == variable_name
                    else variable.domain
                )
                for variable in variables
            }
        ).as_composite_set()

    @staticmethod
    def everything(variables: list[Symbolic]) -> Event:
        """
        :param variables: The variables the event is represented over.
        :return: The event holding everywhere, the identity of intersection.
        """
        return SimpleEvent.from_data(
            {variable: variable.domain for variable in variables}
        ).as_composite_set()

    @classmethod
    def nothing(cls, variables: list[Symbolic]) -> Event:
        """
        :param variables: The variables the event is represented over.
        :return: The event holding nowhere, the identity of union.
        """
        return cls.everything(variables).complement()
