from __future__ import annotations

from dataclasses import dataclass, field
from enum import EnumType
from functools import cached_property
from types import UnionType, EllipsisType

import numpy as np
from typing_extensions import Any, Iterable, Optional, Union, get_args
from krrood.parametrization.exceptions import EmptyVariableDomain, InvalidEllipsis
import random_events.variable
from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.variable import Literal, Variable
from krrood.entity_query_language.factories import and_
from krrood.entity_query_language.query.match import MatchVariable, AttributeMatch
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.data_access_objects.to_dao import ToDataAccessObjectState
from krrood.parametrization.random_events_translator import (
    WhereExpressionToRandomEventTranslator,
)
from krrood.parametrization.feature_extraction.feature_extractor import FeatureExtractor
from random_events.interval import singleton
from random_events.product_algebra import Event, SimpleEvent
from random_events.set import Set
from random_events.variable import (
    compatible_types,
    variable_from_name_and_type,
    most_appropriate_variable_type,
)


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

    truncation_assignments_from_where_conditions: Optional[Event] = field(
        init=False, default=None
    )
    """
    The where condition as random event.
    Only exists if the statement has a where condition.
    """

    conditioning_assignments_from_literal_values: dict[
        random_events.variable.Variable, Any
    ] = field(init=False, default_factory=dict)
    """
    Dictionary of events that are created from literal values, e.g. actual values. These are the assignments, that the probabilistic model is *conditioned* on.
    """

    truncation_assignments_from_krrood_variables: list[Event] = field(
        init=False, default_factory=list
    )
    """
    List of events that are created from symbolic expressions, e.g. variable assignments. These are the assignments, that the probabilistic model is *truncated* on. They may contain literals in their domains, however, the difference to `_events_from_literal_values` is, that these events are not directly used for conditioning, but rather for truncation. This means, that the events of this list do not change the weights of sum nodes in probabilistic circuits.
    """

    _symbolic_expression_event_cache: dict[
        SymbolicExpression, tuple[Event, dict[str, random_events.variable.Variable]]
    ] = field(init=False, default_factory=dict)
    """
    A cache for events that are created from symbolic expressions.
    """

    def __post_init__(self):
        self.statement.expression.build()
        self._random_event_compiler = WhereExpressionToRandomEventTranslator(
            and_(*self.statement._where_conditions_)
        )
        if self.statement._where_conditions_:
            self.truncation_assignments_from_where_conditions = (
                self._random_event_compiler.translate()
            )
        _ = self.variables  # make variables available

    @cached_property
    def variables(self) -> dict[str, random_events.variable.Variable]:
        """
        :return: A dictionary that maps variable names to random events variables that appear in
        the `where` or `Match` statement.
        """
        result = {v.name: v for v in self._random_event_compiler.variables.values()}

        for attribute_match in self.statement.matches_with_variables:
            if attribute_match.assigned_value is None:
                continue

            result.update(self._extract_variables_from_attribute_match(attribute_match))

        return result

    def _extract_variables_from_attribute_match(
        self, attribute_match: AttributeMatch
    ) -> dict[str, random_events.variable.Variable]:
        """
        Extract variables from an attribute match by dispatching to specific handlers.

        :param attribute_match: The attribute match to extract variables from.
        :return: A dictionary of extracted variables.
        """
        krrood_variable = attribute_match.assigned_variable

        if isinstance(krrood_variable, Literal):
            return self._handle_literal_attribute_match(attribute_match)

        if isinstance(krrood_variable, Variable):
            return self._handle_variable_attribute_match(attribute_match)

        return {}

    def _handle_literal_attribute_match(
        self, attribute_match: AttributeMatch
    ) -> dict[str, random_events.variable.Variable]:
        """
        Handle attribute matches where the assigned value is a literal.

        :param attribute_match: The attribute match with a literal assigned value.
        :return: A dictionary of extracted variables.
        """
        name = attribute_match.name_from_variable_access_path
        value = attribute_match.assigned_value
        krrood_variable = attribute_match.assigned_variable
        type_ = self._process_attribute_match_type(krrood_variable._type_)

        if (
            issubclass(type_, compatible_types)
            and not isinstance(value, compatible_types)
            and not isinstance(value, (list, tuple, set))
            and not isinstance(value, EllipsisType)
        ):
            raise TypeError(
                f"The attribute type is {type_}, but the assigned value is of type {type(value)}."
                f"Please enter a value of a valid type. Valid types are {type_} or Ellipsis"
            )

        if isinstance(value, EllipsisType) and not issubclass(type_, compatible_types):
            raise InvalidEllipsis(type_)

        if isinstance(value, EllipsisType):
            return {name: variable_from_name_and_type(name=name, type_=type_)}

        if isinstance(value, compatible_types):
            result = {name: variable_from_name_and_type(name=name, type_=type(value))}
            self.conditioning_assignments_from_literal_values[result[name]] = value
            return result

        if isinstance(value, (list, tuple, set)):
            return self._extract_variables_from_iterable_literal(name, value)

        if isinstance(value, compatible_types) or isinstance(type_, compatible_types):
            return self._handle_compatible_type_literal(name, value, type_)

        return self._extract_variables_from_non_primitive_literal(value, name)

    def _handle_compatible_type_literal(
        self,
        name: str,
        value: Any,
        type_: type,
    ) -> dict[str, random_events.variable.Variable]:
        """
        Handle a literal whose value or declared type is a primitive compatible type.

        Picks ``type_`` when it is a concrete compatible type; falls back to
        ``type(value)`` otherwise.

        :param name: The variable name derived from the attribute access path.
        :param value: The literal value being matched.
        :param type_: The processed declared type of the attribute.
        :return: A dictionary containing the extracted variable.
        """
        effective_type = (
            type_
            if (
                type_ is not None
                and isinstance(type_, type)
                and issubclass(type_, compatible_types)
            )
            else type(value)
        )
        result = {name: variable_from_name_and_type(name=name, type_=effective_type)}
        self.conditioning_assignments_from_literal_values[result[name]] = value
        return result

    def _extract_variables_from_iterable_literal(
        self, name: str, value: Union[list, tuple]
    ) -> dict[str, random_events.variable.Variable]:
        """
        Extract variables from an iterable literal by processing each element with an
        indexed name prefix (e.g. ``walls[0].height``, ``walls[1].start_point.x``).

        Primitive elements become a single conditioning variable; non-primitive elements
        are decomposed via
        :py:class:`~krrood.parametrization.feature_extraction.feature_extractor.FeatureExtractor`.

        :param name: Base variable name derived from the attribute access path.
        :param value: The iterable assigned value.
        :return: A dictionary of extracted variables.
        """
        result = {}
        for index, element in enumerate(value):
            if element is None:
                continue

            type_ = self._process_attribute_match_type(type(element))
            if isinstance(element, EllipsisType) and not issubclass(
                type_, compatible_types
            ):
                raise InvalidEllipsis(type_)

            indexed_name = f"{name}[{index}]"
            if isinstance(element, compatible_types):
                result.update(self._handle_compatible_type_literal(indexed_name, element, type_))
            else:
                result.update(
                    self._extract_variables_from_non_primitive_literal(element, indexed_name)
                )
        return result

    def _extract_variables_from_non_primitive_literal(
        self, value: Any, name_prefix: str
    ) -> dict[str, random_events.variable.Variable]:
        """
        Extract variables from a single non-primitive literal value.

        Converts ``value`` to a DAO, runs feature extraction, and registers a
        conditioning assignment for every discovered feature.

        :param value: The non-primitive literal to decompose.
        :param name_prefix: Attribute access path used to namespace the feature names
            (e.g. ``"obj"`` yields ``"obj.position.x"``).
        :return: A dictionary mapping prefixed feature names to their variables.
        """
        result = {}
        dao_state = ToDataAccessObjectState()
        extractor = FeatureExtractor.from_instances([to_dao(value, dao_state)])
        for feature in extractor.features:
            feature_name = f"{name_prefix}.{feature.get_clean_name_from_mapped_variable()}"
            random_events_variable = variable_from_name_and_type(
                name=feature_name, type_=feature._type_
            )
            result[feature_name] = random_events_variable
            mapping = feature.apply_mapping_on_external_root(value)
            if not isinstance(mapping, feature._type_):
                mapping = feature._type_(mapping)
            self.conditioning_assignments_from_literal_values[random_events_variable] = mapping
        return result

    def _handle_variable_attribute_match(
        self, attribute_match: AttributeMatch
    ) -> dict[str, random_events.variable.Variable]:
        """
        Handle attribute matches where the assigned value is a KRROOD variable.

        :param attribute_match: The attribute match with a KRROOD variable assigned value.
        :return: A dictionary of extracted variables.
        """
        type_ = self._process_attribute_match_type(
            attribute_match.assigned_variable._type_
        )
        if attribute_match.assigned_value in self._symbolic_expression_event_cache:
            return self._symbolic_expression_event_cache[
                attribute_match.assigned_value
            ][1]

        domain_objects = attribute_match.assigned_value.tolist()

        if not domain_objects:
            raise EmptyVariableDomain(attribute_match.variable)

        if not type_ is None and issubclass(type_, compatible_types):
            return self._extract_variables_from_primitive_krrood_variable(
                attribute_match, domain_objects
            )

        return self._extract_variables_from_non_primitive_krrood_variable(
            attribute_match, domain_objects
        )

    def _extract_variables_from_primitive_krrood_variable(
        self, attribute_match: AttributeMatch, domain_objects: list[Any]
    ) -> dict[str, random_events.variable.Variable]:
        """
        Extract variables from a KRROOD variable with a primitive type.

        :param attribute_match: The attribute match.
        :param domain_objects: The objects in the variable's domain.
        :return: A dictionary of extracted variables.
        """
        name = attribute_match.name_from_variable_access_path
        type_ = self._process_attribute_match_type(
            attribute_match.assigned_variable._type_
        )
        re_variable = variable_from_name_and_type(name=name, type_=type_)
        result = {re_variable.name: re_variable}

        if isinstance(type_, EnumType):
            simple_events = [
                SimpleEvent.from_data({re_variable: Set.from_iterable(domain_objects)})
            ]
        else:
            simple_events = [
                SimpleEvent.from_data({re_variable: singleton(obj)})
                for obj in domain_objects
            ]
        self.truncation_assignments_from_krrood_variables.append(
            Event.from_simple_sets(*simple_events)
        )
        return result

    def _extract_variables_from_non_primitive_krrood_variable(
        self, attribute_match: AttributeMatch, domain_objects: list[Any]
    ) -> dict[str, random_events.variable.Variable]:
        """
        Extract variables from a KRROOD variable with a non-primitive type.

        :param attribute_match: The attribute match.
        :param domain_objects: The objects in the variable's domain.
        :return: A dictionary of extracted variables.
        """
        state = ToDataAccessObjectState()
        hashes = [hash(obj) for obj in domain_objects]
        data_access_objects = [to_dao(obj, state=state) for obj in domain_objects]

        extractor = FeatureExtractor.from_instances(data_access_objects)

        result = {}

        for feature in extractor.features:
            feature_name = feature.get_clean_name_from_mapped_variable()
            name = (
                f"{attribute_match.name_from_variable_access_path}." f"{feature_name}"
            )
            re_variable = variable_from_name_and_type(name=name, type_=feature._type_)
            result[re_variable.name] = re_variable

        identifier_name = f"{attribute_match.name_from_variable_access_path}"
        identifier_variable = random_events.variable.Symbolic(
            name=identifier_name, domain=Set.from_iterable(hashes)
        )
        result[identifier_variable.name] = identifier_variable

        simple_events = []
        for hash_, dao in zip(hashes, data_access_objects):
            current_feature_values = extractor.apply_mapping(dao)

            data = {identifier_variable: hash_}
            for feature, value in zip(extractor.features, current_feature_values):
                feature_name = feature.get_clean_name_from_mapped_variable()
                name = (
                    f"{attribute_match.name_from_variable_access_path}.{feature_name}"
                )
                data[result[name]] = value

            simple_events.append(SimpleEvent.from_data(data))

        resulting_event = Event.from_simple_sets(*simple_events)
        self.truncation_assignments_from_krrood_variables.append(resulting_event)
        self._symbolic_expression_event_cache[attribute_match.assigned_value] = (
            resulting_event,
            result,
        )

        return result

    def construct_instance_from_model_sample(
        self,
        variables: Iterable[random_events.variable.Variable],
        sample: np.ndarray,
    ) -> dict[random_events.variable.Variable, Any]:
        """
        Construct an instance from a sample of a probabilistic model.

        :param variables: The variables from a probabilistic model.
        :param sample: A sample from the same model.
        :return: The constructed instance.
        """
        sample_mapping = dict(zip(variables, sample))
        for variable_, value in sample_mapping.items():
            mapped_variable = self.statement._get_mapped_variable_by_name(
                variable_.name
            )
            attribute_match = [
                match
                for match in self.statement.matches_with_variables
                if match.name_from_variable_access_path == variable_.name
            ]
            attribute_match = attribute_match[0] if attribute_match else None
            if attribute_match is None:
                continue
            if mapped_variable is None:
                continue

            if attribute_match and isinstance(
                attribute_match.assigned_value, SymbolicExpression
            ):
                [domain_index] = [
                    val
                    for index, val in variable_.domain.hash_map.items()
                    if index == value
                ]
                [value] = [
                    domain_value
                    for domain_value in attribute_match.assigned_value.tolist()
                    if hash(domain_value) == domain_index
                ]
            elif not variable_.is_numeric:
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

    @staticmethod
    def _process_attribute_match_type(type_):
        """
        Process the type of an attribute matches assigned variable such that there are no unions.
        :param type_: The type to process
        :return: The processed type.
        """
        if isinstance(type_, UnionType):
            types = get_args(type_)
            return most_appropriate_variable_type(types)
        else:
            return type_
