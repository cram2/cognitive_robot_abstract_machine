from __future__ import annotations

import typing
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, Optional

import numpy as np
from typing_extensions import Any

import random_events.variable
from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.factories import and_
from krrood.entity_query_language.query.match import MatchVariable
from krrood.parametrization.random_events_translator import (
    WhereExpressionToRandomEventTranslator,
)
from random_events.product_algebra import Event
from random_events.set import Set


@dataclass
class UnderspecifiedParameters:
    """
    A class that extracts all necessary information from a {py:class}`~krrood.entity_query_language.query.match.Match`
    and binds it together. Instances of this can be used to parameterize objects with underspecified variables using
    generative models. This generally serves as glue between `ProbabilisticModel` and `Match`.
    """

    statement: MatchVariable
    """
    The UnderspecifiedVariable to extract information from.
    """

    _random_event_compiler: Optional[WhereExpressionToRandomEventTranslator] = field(
        init=False
    )
    """
    The translator that extracts a random event from the where conditions.
    Only exists if the statement has a where condition.
    """

    truncation_event: Optional[Event] = field(init=False, default=None)
    """
    The where condition as random event.
    Only exists if the statement has a where condition.
    """

    def __post_init__(self):
        self.statement.expression.build()
        self._random_event_compiler = WhereExpressionToRandomEventTranslator(
            and_(*self.statement._where_conditions_)
        )
        if self.statement._where_conditions_:
            self.truncation_event = self._random_event_compiler.translate()

    @cached_property
    def variables(self) -> Dict[str, random_events.variable.Variable]:
        """
        :return: A dictionary that maps variable names to random events variables that appear in
        the `where` or `Match` statement.
        """
        result = {v.name: v for v in self._random_event_compiler.variables.values()}

        for attribute_match in self.statement.matches_with_variables:
            name = attribute_match.name_from_variable_access_path

            if isinstance(attribute_match.assigned_value, SymbolicExpression):
                random_events_variable = random_events.variable.Symbolic(
                    name=name,
                    domain=Set.from_iterable(attribute_match.assigned_value.tolist()),
                )
                result[random_events_variable.name] = random_events_variable
                continue
            if attribute_match.assigned_variable._type_ is None or not issubclass(
                attribute_match.assigned_variable._type_,
                random_events.variable.compatible_types,
            ):
                continue

            random_events_variable = random_events.variable.variable_from_name_and_type(
                name, attribute_match.assigned_variable._type_
            )

            result[random_events_variable.name] = random_events_variable
        return result

    @property
    def assignments_for_conditioning(
        self,
    ) -> Dict[random_events.variable.Variable, Any]:
        """
        :return: A dictionary that contains all facts from the statement and that can be directly used for
        conditioning a probabilistic model. These values ignore the `where` conditions.
        """
        result = {}
        for literal in self.statement.matches_with_variables:
            variable = self.variables.get(literal.assigned_variable._name_, None)
            if variable is None or isinstance(
                literal.assigned_variable._value_, (type(Ellipsis), SymbolicExpression)
            ):
                continue

            result[variable] = literal.assigned_variable._value_
        return result

    def create_instance_from_variables_and_sample(
        self,
        variables: typing.Iterable[random_events.variable.Variable],
        sample: np.ndarray,
    ) -> Dict[random_events.variable.Variable, Any]:
        """
        Create an instance from a sample of a probabilistic model.

        :param variables: The variables from a probabilistic model.
        :param sample: A sample from the same model-
        :return: The instance
        """

        for variable_, value in zip(variables, sample):
            mapped_variable = self.statement._get_mapped_variable_by_name(
                variable_.name
            )

            if not variable_.is_numeric:
                [value] = [
                    domain_value.element
                    for domain_value in variable_.domain
                    if hash(domain_value) == value
                ]
            else:
                value = value.item()
            mapped_variable._value_ = value

        self.statement._update_kwargs_from_literal_values()
        result = self.statement.construct_instance()
        return result
