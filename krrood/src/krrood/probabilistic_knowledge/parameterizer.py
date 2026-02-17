from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from random_events.product_algebra import SimpleEvent
from random_events.set import Set
from random_events.variable import Continuous, Integer, Symbolic, Variable
from sqlalchemy import inspect, Column
from sqlalchemy.orm import Relationship
from typing_extensions import List, Optional, assert_never, Any, Tuple, Type

from probabilistic_model.probabilistic_circuit.rx.helper import fully_factorized
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from ..adapters.json_serializer import list_like_classes
from ..class_diagrams.class_diagram import WrappedClass
from ..class_diagrams.wrapped_field import WrappedField
from ..ormatic.dao import DataAccessObject, get_dao_class, to_dao


@dataclass
class Parameterization:
    """
    A class that contains the variables and simple event resulting from parameterizing a DataAccessObject.
    """

    variables: List[Variable] = field(default_factory=list)
    """
    A list of random event variables that are being parameterized.
    """
    simple_event: SimpleEvent = field(default_factory=lambda: SimpleEvent({}))
    """
    A SimpleEvent containing the values of the variables.
    """

    def fill_missing_variables(self):
        self.simple_event.fill_missing_variables(self.variables)

    def extend_variables(self, variables: List[Variable]):
        """
        Update the variables by extending them with the given variables.
        """
        self.variables.extend(variables)

    def update_simple_event(self, simple_event: SimpleEvent):
        """
        Update the simple event by extending it with the given simple event.
        """
        self.simple_event.update(simple_event)

    def create_fully_factorized_distribution(self) -> ProbabilisticCircuit:
        """
        Create a fully factorized probabilistic circuit over the variables in the parameterization.
        """
        distribution_variables = [
            v for v in self.variables if not isinstance(v, Integer)
        ]

        return fully_factorized(
            distribution_variables,
            means={v: 0.0 for v in distribution_variables if isinstance(v, Continuous)},
            variances={
                v: 1.0 for v in distribution_variables if isinstance(v, Continuous)
            },
        )

    def merge_parameterization(self, other: Parameterization):
        """
        Update the parameterization with another parameterization by extending the variables and updating the simple event.

        :param other: The parameterization to update with.
        """
        self.variables.extend(other.variables)
        self.simple_event.update(other.simple_event)


@dataclass
class Parameterizer:
    """
    A class that can be used to parameterize a DataAccessObject into random event variables and a simple event
    containing the values of the variables.
    The resulting variables and simple event can then be used to create a probabilistic circuit.
    """

    parameterization: Parameterization = field(default_factory=Parameterization)
    """
    Parameterization containing the variables and simple event resulting from parameterizing a DataAccessObject.
    """

    def parameterize(self, object: Any, prefix: str) -> Parameterization:
        """
        Create variables for all fields of an object.

        :param object: The object to parameterize.
        :param prefix: The prefix to use for variable names.

        :return: Parameterization containing the variables and simple event.
        """
        if type(object) in list_like_classes:
            for i, value in enumerate(object):
                self.parameterize(value, f"{prefix}[{i}]")
            return self.parameterization
        else:
            dao = to_dao(object)
            return self.parameterize_dao(dao, prefix)

    def parameterize_dao(self, dao: DataAccessObject, prefix: str) -> Parameterization:
        """
        Create variables for all fields of a DataAccessObject.

        :param dao: The DataAccessObject to parameterize.
        :param prefix: The prefix to use for variable names.

        :return: A Parameterization containing the variables and simple event.
        """
        sql_alchemy_mapper = inspect(dao).mapper

        for wrapped_field in WrappedClass(dao.original_class()).fields:
            for relationship in sql_alchemy_mapper.relationships:
                self._process_relationship(relationship, wrapped_field, dao, prefix)

            for column in sql_alchemy_mapper.columns:
                variables, attribute_values = self._process_column(
                    column, wrapped_field, dao, prefix
                )
                self._update_variables_and_event(variables, attribute_values)

        self.parameterization.fill_missing_variables()
        return self.parameterization

    def _update_variables_and_event(
        self, variables: List[Variable], attribute_values: List[Any]
    ):
        """
        Update the current parameterization by the given variables and attribute values.

        :param variables: The variables to add to the variables list.
        :param attribute_values: The attribute values to add to the simple event.
        """
        for variable, attribute_value in zip(variables, attribute_values):
            if variable is None:
                continue
            self.parameterization.extend_variables([variable])
            if attribute_value is None:
                continue

            self.parameterization.update_simple_event(
                SimpleEvent({variable: attribute_value})
            )

    def _process_relationship(
        self,
        relationship: Relationship,
        wrapped_field: WrappedField,
        dao: DataAccessObject,
        prefix: str,
    ):
        """
        Process a SQLAlchemy relationship and add variables and events for it.

        ..Note:: This method is recursive and will process all relationships of a relationship. Optional relationships that are None will be skipped, as we decided that they should not be included in the model.

        :param relationship: The SQLAlchemy relationship to process.
        :param wrapped_field: The WrappedField potentially corresponding to the relationship.
        :param dao: The DataAccessObject containing the relationship.
        :param prefix: The prefix to use for variable names.
        """
        attribute_name = relationship.key

        # %% Skip attributes that are not of interest.
        if not self._is_attribute_of_interest(attribute_name, dao, wrapped_field):
            return

        attribute_dao = getattr(dao, attribute_name)

        # %% one to many relationships
        if wrapped_field.is_one_to_many_relationship:
            for value in attribute_dao:
                self.parameterize_dao(
                    dao=value.target, prefix=f"{prefix}.{attribute_name}"
                )
            return

        # %% one to one relationships
        if wrapped_field.is_one_to_one_relationship:
            if attribute_dao is None:
                attribute_dao = get_dao_class(wrapped_field.type_endpoint)()
            self.parameterize_dao(
                dao=attribute_dao,
                prefix=f"{prefix}.{attribute_name}",
            )
            return

        else:
            assert_never(wrapped_field)

    def _process_column(
        self,
        column: Column,
        wrapped_field: WrappedField,
        dao: DataAccessObject,
        prefix: str,
    ) -> Tuple[List[Variable], List[Any]]:
        """
        Process a SQLAlchemy column and create variables and events for it.

        :param column: The SQLAlchemy column to process.
        :param wrapped_field: The WrappedField potentially corresponding to the column.
        :param dao: The DataAccessObject containing the column.
        :param prefix: The prefix to use for variable names.

        :return: A tuple containing a list of variables and a list of corresponding attribute values.
        """
        attribute_name = self._column_attribute_name(column, dao)

        # %% Skip attributes that are not of interest.
        if not self._is_attribute_of_interest(attribute_name, dao, wrapped_field):
            return [], []

        attribute = getattr(dao, attribute_name)

        # %% one to many relationships
        if wrapped_field.is_collection_of_builtins:
            variables = [
                self._create_variable_from_type(
                    wrapped_field.type_endpoint, f"{prefix}.{value}"
                )
                for value in attribute
            ]
            return variables, attribute

        # %% one to one relationships
        if wrapped_field.is_builtin_type or wrapped_field.is_enum:
            var = self._create_variable_from_type(
                wrapped_field.type_endpoint, f"{prefix}.{attribute_name}"
            )
            return [var], [attribute]

        else:
            assert_never(wrapped_field)

    def _is_attribute_of_interest(
        self,
        attribute_name: Optional[str],
        dao: DataAccessObject,
        wrapped_field: WrappedField,
    ) -> bool:
        """
        Check if the correct attribute is being inspected, and if yes, if it should be included in the model

        ..info:: Included are only attributes that are not primary keys, foreign keys, and that are not optional with
        a None value. Additionally, attributes of type uuid.UUID and str are excluded.

        :param attribute_name: The name of the attribute to check.
        :param dao: The DataAccessObject containing the attribute.
        :param wrapped_field: The WrappedField corresponding to the attribute.

        :return: True if the attribute is of interest, False otherwise.
        """
        return (
            attribute_name
            and wrapped_field.public_name == attribute_name
            and not wrapped_field.type_endpoint in (datetime, uuid.UUID, str)
            and not (wrapped_field.is_optional and getattr(dao, attribute_name) is None)
        )

    def _column_attribute_name(
        self, column: Column, dao: DataAccessObject
    ) -> Optional[str]:
        """
        Get the attribute name corresponding to a SQLAlchemy Column, if it is not a primary key, foreign key, or polymorphic type.

        :return: The attribute name or None if the column is not of interest.
        """
        if hasattr(dao, "__mapper_args__") and column.key == dao.__mapper_args__.get(
            "polymorphic_on", None
        ):
            return None

        if column.primary_key or column.foreign_keys:
            return None

        return column.name

    def _create_variable_from_type(
        self,
        field_type: Type[enum.Enum] | Type[bool] | Type[int] | Type[float],
        name: str,
    ) -> Variable:
        """
        Create a random event variable based on its type.

        :param field_type: The type of the field for which to create the variable. Usually accessed through WrappedField.type_endpoint.
        :param name: The name of the variable.

        :return: A random event variable or raise error if the type is not supported.
        """

        if issubclass(field_type, enum.Enum):
            return Symbolic(name, Set.from_iterable(list(field_type)))
        elif issubclass(field_type, bool):
            return Symbolic(name, Set.from_iterable([True, False]))
        elif issubclass(field_type, int):
            return Integer(name)
        elif issubclass(field_type, float):
            return Continuous(name)
        else:
            assert_never(field_type)

    def create_fully_factorized_distribution(
        self,
    ) -> ProbabilisticCircuit:
        """
        Create a fully factorized probabilistic circuit over the given variables.

        :return: A fully factorized probabilistic circuit.
        """
        return self.parameterization.create_fully_factorized_distribution()
