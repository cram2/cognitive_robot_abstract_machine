from __future__ import annotations

import typing
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, Optional, Tuple

import numpy as np
from typing_extensions import Any

import random_events.variable
from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.factories import and_
from krrood.entity_query_language.core.mapped_variable import (
    Attribute,
    Index,
    Call,
    MappedVariable,
)
from krrood.entity_query_language.predicate import symbolic_function
from krrood.entity_query_language.query.match import MatchVariable, AttributeMatch
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.data_access_objects.to_dao import ToDataAccessObjectState
from krrood.parametrization.random_events_translator import (
    WhereExpressionToRandomEventTranslator,
)
from probabilistic_model.probabilistic_circuit.relational.learn_rspn import (
    get_features_of_class,
    FeatureExtractor,
)
from random_events.product_algebra import Event, SimpleEvent
from random_events.set import Set
from semantic_digital_twin.world_description.world_entity import Body


@symbolic_function
def symbolic_hash(value: Any) -> int:
    return hash(value)


def get_clean_name_from_mapped_variable(variable: MappedVariable) -> str:
    """
    Get a clean name from a mapped variable by joining its attribute names.

    :param variable: The mapped variable.
    :return: The clean name.
    """
    names = []
    for step in variable._access_path_:
        if isinstance(step, Attribute):
            names.append(step._attribute_name_)
        elif isinstance(step, Index):
            names.append(f"[{step._key_}]")
        elif isinstance(step, Call):
            names.append(f"()")
    return ".".join(names)


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

    generated_events: typing.List[Event] = field(init=False, default_factory=list)

    _symbolic_expression_event_cache: Dict[
        SymbolicExpression, Tuple[Event, Dict[str, random_events.variable.Variable]]
    ] = field(init=False, default_factory=dict)

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
                variables = self._create_variables_from_symbolic_expression(
                    attribute_match
                )
                result.update(variables)
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

    def _create_variables_from_symbolic_expression(
        self, attribute_match: AttributeMatch
    ) -> Dict[str, random_events.variable.Variable]:

        if attribute_match.assigned_value in self._symbolic_expression_event_cache:
            return self._symbolic_expression_event_cache[attribute_match.assigned_value]

        result = {}

        state = ToDataAccessObjectState()
        domain_objects = attribute_match.assigned_value.tolist()
        hashes = [hash(obj) for obj in domain_objects]
        data_access_objects = [to_dao(obj, state=state) for obj in domain_objects]

        features = get_features_of_class(
            data_access_objects[0],
            attribute_match.assigned_variable,
            [],
            set(),
        )
        extractor = FeatureExtractor(features)

        # extract feature variables
        for feature in extractor.features:
            relative_feature_name = get_clean_name_from_mapped_variable(feature)
            name = f"{attribute_match.name_from_variable_access_path}.{relative_feature_name}"
            random_events_variable = random_events.variable.variable_from_name_and_type(
                name=name, type_=feature._type_
            )
            result[random_events_variable.name] = random_events_variable

        identifier_name = f"{attribute_match.name_from_variable_access_path}"
        identifier_variable = random_events.variable.Symbolic(
            name=identifier_name, domain=Set.from_iterable(hashes)
        )

        result[identifier_variable.name] = identifier_variable

        simple_events = []
        for index, (hash_, dao) in enumerate(zip(hashes, data_access_objects)):
            current_features = extractor.apply_mapping(dao)

            current_simple_event = SimpleEvent.from_data(
                {
                    identifier_variable: hash_,
                    **{
                        result[
                            f"{attribute_match.name_from_variable_access_path}.{get_clean_name_from_mapped_variable(f)}"
                        ]: v
                        for f, v in zip(features, current_features)
                    },
                }
            )
            simple_events.append(current_simple_event)

        resulting_event = Event.from_simple_sets(*simple_events)
        self.generated_events.append(resulting_event)
        self._symbolic_expression_event_cache[attribute_match.assigned_value] = (
            resulting_event,
            result,
        )

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
        sample_mapping = dict(zip(variables, sample))
        for variable_, value in sample_mapping.items():
            mapped_variable = self.statement._get_mapped_variable_by_name(
                variable_.name
            )
            if mapped_variable is None:
                continue

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
