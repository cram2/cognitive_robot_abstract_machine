from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable

import numpy as np

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

from .object_access_variable import ObjectAccessVariable, AttributeAccessLike
from ..adapters.json_serializer import list_like_classes
from ..class_diagrams.class_diagram import WrappedClass
from ..class_diagrams.wrapped_field import WrappedField
from ..entity_query_language.factories import variable_from, variable
from ..entity_query_language.core.mapped_variable import Index, Selectable
from ..ormatic.dao import (
    DataAccessObject,
    get_dao_class,
    to_dao,
    ToDataAccessObjectState,
)


@dataclass
class Parameterization:
    """
    A class that contains the variables and simple event resulting from parameterizing a DataAccessObject.
    """

    variables: List[ObjectAccessVariable] = field(default_factory=list)
    """
    A list of random event variables that are being parameterized.
    """

    assignments: Dict[ObjectAccessVariable, Any] = field(default_factory=dict)
    """
    A dict containing the assignments of the variables to concrete values.
    This may contain less variables than `variables` if some variables are not being specified (using ...).
    These assignments are intended for conditioning probabilistic models.
    """

    @property
    def random_events_variables(self) -> List[Variable]:
        return [v.variable for v in self.variables]

    @property
    def assignments_for_conditioning(self) -> Dict[Variable, Any]:
        return {v.variable: value for v, value in self.assignments.items()}

    def extend_variables(self, variables: List[ObjectAccessVariable]):
        """
        Update the variables by extending them with the given variables.
        """
        self.variables.extend(variables)

    def create_fully_factorized_distribution(self) -> ProbabilisticCircuit:
        """
        Create a fully factorized probabilistic circuit over the variables in the parameterization.
        """
        distribution_variables = [
            v for v in self.random_events_variables if not isinstance(v, Integer)
        ]

        return fully_factorized(
            distribution_variables,
            means={v: 0.0 for v in distribution_variables if isinstance(v, Continuous)},
            variances={
                v: 1.0 for v in distribution_variables if isinstance(v, Continuous)
            },
        )

    def _parameterize_data_access_object_with_sample(
        self, dao: DataAccessObject, sample: Dict[ObjectAccessVariable, Any]
    ):
        """
        Parameterize a DataAccessObject with a sample in place.
        The structure of `dao` has to be compatible with the access patterns of this parameterizers variables.

        :param dao: The DataAccessObject to parameterize in place.
        :param sample: The sample to apply to the object.
        """
        for variable, value in sample.items():
            variable.set_value(dao, value)

    def parameterize_object_with_sample(
        self, obj: Any, sample: Dict[ObjectAccessVariable, Any]
    ) -> Any:
        """
        Parameterize an object with a sample.

        :param obj: The object to parameterize.
        :param sample: The sample to parameterize the object with.
        :return: A new copy of the object with the parameters.
        """
        dao = to_dao(obj)
        self._parameterize_data_access_object_with_sample(dao, sample)
        result = dao.from_dao()
        return result

    def get_variable_by_name(self, name: str) -> ObjectAccessVariable:
        """
        Get an ObjectAccessVariable by its name.

        :param name: The name of the variable to retrieve.
        :return: The variable with the specified name.
        """
        [result] = [v for v in self.variables if v.variable.name == name]
        return result

    def create_assignment_from_variables_and_sample(
        self, variables: Iterable[Variable], sample: np.ndarray
    ) -> Dict[ObjectAccessVariable, Any]:
        """
        The sample has to be constructed from a circuit that matches the variables of this parameterizer.
        """
        result = {}
        for variable, value in zip(variables, sample):
            object_access_variable = self.get_variable_by_name(variable.name)

            if not object_access_variable.variable.is_numeric:
                value = [
                    domain_value.element
                    for domain_value in object_access_variable.variable.domain
                    if hash(domain_value) == value
                ][0]
            else:
                value = value.item()
            result[object_access_variable] = value

        return result


@dataclass
class Parameterizer:
    """
    A class that can be used to parameterize an object into object access variables and an assignment event
    containing the values of the variables.

    For this, the target object first is converted into a DataAccessObject. Use the Ellipsis (...) to signal that a
    field should be parameterized. Use None to signal that a field should not be parameterized.

    For example

    .. code-block:: python

        parameterization = Parameterizer().parameterize(Position(x=..., y=0.69, z=None))

    will create 2 variables for the `x` and `y` fields of the Position class and an assignment containing ``{y: 0.69}``.
    `z` will not be parameterized as its set to `None`.

    The resulting variables and assignments can then be used to create probabilistic models.
    """

    parameterization: Parameterization = field(default_factory=Parameterization)
    """
    Parameterization containing the variables and simple event resulting from parameterizing a DataAccessObject.
    """

    def parameterize(self, obj: Any) -> Parameterization:
        """
        Create variables for all fields of an object.

        :param obj: The object to generate the parametrization from.

        :return: Parameterization containing the variables and simple event.
        """
        if type(obj) in list_like_classes:
            raise NotImplementedError(
                "Parameterization of list-like types is not supported directly."
            )

        dao = to_dao(obj)

        dao_variable = variable(type(dao), [dao])

        self._parameterize_dao(dao, dao_variable)

        return self.parameterization

    def _parameterize_dao(
        self, dao: DataAccessObject, dao_variable: Selectable
    ) -> Parameterization:
        """
        Create variables for all fields of a DataAccessObject.

        :param dao: The DataAccessObject to extract the parameters from.
        :param dao_variable: The EQL variable corresponding to the DataAccessObject for symbolic access.

        :return: A Parameterization containing the variables and simple event.
        """
        sql_alchemy_mapper = inspect(dao).mapper

        for wrapped_field in WrappedClass(dao.original_class()).fields:

            relationship = sql_alchemy_mapper.relationships.get(
                wrapped_field.name, None
            )
            if relationship is not None:
                self._process_relationship(
                    relationship, wrapped_field, dao, dao_variable
                )
                continue

            column = sql_alchemy_mapper.columns.get(wrapped_field.name, None)
            if column is not None:
                variables, attribute_values = self._process_column(
                    column, wrapped_field, dao, dao_variable
                )
                self._update_variables_and_assignments(variables, attribute_values)

        return self.parameterization

    def _update_variables_and_assignments(
        self, variables: List[ObjectAccessVariable], attribute_values: List[Any]
    ):
        """
        Update the current parameterization by the given variables and attribute values.

        :param variables: The variables to add to the variables list.
        :param attribute_values: The attribute values to add to the simple event.
        """
        for variable, attribute_value in zip(variables, attribute_values):

            self.parameterization.extend_variables([variable])
            if attribute_value == Ellipsis:
                continue

            self.parameterization.assignments[variable] = attribute_value

    def _process_relationship(
        self,
        relationship: Relationship,
        wrapped_field: WrappedField,
        dao: DataAccessObject,
        dao_variable: Selectable,
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
        attribute_dao = getattr(dao, attribute_name)
        symbolic_attribute_access = getattr(dao_variable, attribute_name)

        if attribute_dao is None:
            return

        # %% one to many relationships
        if wrapped_field.is_one_to_many_relationship:
            for index, value in enumerate(attribute_dao):
                if value.target is None:
                    continue
                self._parameterize_dao(
                    dao=value.target,
                    dao_variable=symbolic_attribute_access[index].target,
                )
            return

        # %% one to one relationships
        if wrapped_field.is_one_to_one_relationship:
            self._parameterize_dao(
                dao=attribute_dao,
                dao_variable=symbolic_attribute_access,
            )
            return

        else:
            assert_never(wrapped_field)

    def _process_column(
        self,
        column: Column,
        wrapped_field: WrappedField,
        dao: DataAccessObject,
        dao_variable: Selectable,
    ) -> Tuple[List[ObjectAccessVariable], List[Any]]:
        """
        Process an SQLAlchemy column and create variables and events for it.

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
        symbolic_attribute_access = getattr(dao_variable, attribute_name)

        if wrapped_field.is_collection_of_builtins:
            variables = [
                self._create_variable_from_type(
                    wrapped_field.type_endpoint, symbolic_attribute_access[index]
                )
                for index, value in enumerate(attribute)
            ]
            return variables, attribute

        if wrapped_field.is_builtin_type or wrapped_field.is_enum:
            var = self._create_variable_from_type(
                wrapped_field.type_endpoint, symbolic_attribute_access
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
            not wrapped_field.type_endpoint in (datetime, uuid.UUID, str)
            and getattr(dao, attribute_name) is not None
        )

    def _column_attribute_name(
        self, column: Column, dao: DataAccessObject
    ) -> Optional[str]:
        """
        Get the attribute name corresponding to a SQLAlchemy Column if it is not a primary key, foreign key, or polymorphic type.

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
        symbolic_access_variable: AttributeAccessLike,
    ) -> ObjectAccessVariable:
        """
        Create an object access variable based on a python type.

        :param field_type: The type of the field for which to create the variable. Usually accessed through WrappedField.type_endpoint.
        :param symbolic_access_variable: The EQL statement that accesses the field.

        :return: A object access variable or raise error if the type is not supported.
        """

        if issubclass(field_type, enum.Enum):
            result = Symbolic(
                str(symbolic_access_variable), Set.from_iterable(field_type)
            )
        elif issubclass(field_type, bool):
            result = Symbolic(
                str(symbolic_access_variable), Set.from_iterable([True, False])
            )
        elif issubclass(field_type, int):
            result = Integer(str(symbolic_access_variable))
        elif issubclass(field_type, float):
            result = Continuous(str(symbolic_access_variable))
        else:
            assert_never(field_type)

        return ObjectAccessVariable(result, symbolic_access_variable)

    def create_fully_factorized_distribution(
        self,
    ) -> ProbabilisticCircuit:
        """
        Create a fully factorized probabilistic circuit over the given variables.

        :return: A fully factorized probabilistic circuit.
        """
        return self.parameterization.create_fully_factorized_distribution()
