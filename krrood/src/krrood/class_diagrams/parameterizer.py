from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing_extensions import List, Optional, Dict

from probabilistic_model.probabilistic_circuit.rx.helper import fully_factorized
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from random_events.set import Set
from random_events.variable import Continuous, Integer, Symbolic, Variable

from ..class_diagrams.class_diagram import WrappedClass
from ..class_diagrams.wrapped_field import WrappedField


@dataclass
class Parameterizer:
    """
    Parameterizer for creating random event variables from WrappedClass instances.

    This class provides methods to recursively inspect a class structure (via WrappedClass)
    and generate corresponding random variables for probabilistic modeling.
    """

    def __call__(self, wrapped_class: WrappedClass) -> List[Variable]:
        """
        Create random event variables from a WrappedClass.

        """
        return self.parameterize(wrapped_class, prefix=wrapped_class.clazz.__name__)

    def parameterize(self, wrapped_class: WrappedClass, prefix: str) -> List[Variable]:
        """
        Create variables for all fields of a WrappedClass.

        :return: A list of random event variables.
        """

        variables_by_name: Dict[str, Variable] = {}

        for wrapped_field in wrapped_class.fields:
            for var in self._parameterize_wrapped_field(wrapped_field, prefix):
                if var.name not in variables_by_name:
                    variables_by_name[var.name] = var

        return list(variables_by_name.values())

    def _parameterize_wrapped_field(
        self, wrapped_field: WrappedField, prefix: str
    ) -> List[Variable]:
        """
        Create variables for a single WrappedField.

        :return: A list of variables
        """
        field_name = f"{prefix}.{wrapped_field.name}"

        if self.skip_field(wrapped_field):
            return []

        if wrapped_field.is_one_to_one_relationship and not wrapped_field.is_enum:
            return self._parameterize_relationship(wrapped_field, field_name)

        variable = self._create_variable_from_field(wrapped_field, field_name)

        return [variable]

    def skip_field(self, wrapped_field: WrappedField) -> bool:
        """
        Currently skipping the 'datetime' fields.
        """
        return wrapped_field.type_endpoint is datetime

    def _parameterize_relationship(
        self, wrapped_field: WrappedField, field_name: str
    ) -> List[Variable]:
        """
        Create variables for a one-to-one relationship field.

        :return: A list of variables from the related class.
        """
        if not hasattr(wrapped_field.clazz, "_class_diagram"):
            return []

        class_diagram = wrapped_field.clazz._class_diagram
        if not class_diagram:
            return []

        target_type = wrapped_field.type_endpoint
        if target_type not in class_diagram._cls_wrapped_cls_map:
            return []

        target_wrapped_class = class_diagram.get_wrapped_class(target_type)
        return self.parameterize(target_wrapped_class, prefix=field_name)

    def _create_variable_from_field(
        self, wrapped_field: WrappedField, field_name: str
    ) -> Optional[Variable]:
        """
        Create a random event variable from a WrappedField based on its type.

        :return: A random event variable or raise error if the type is not supported.
        """
        endpoint_type = wrapped_field.type_endpoint

        if wrapped_field.is_enum:
            return Symbolic(field_name, Set.from_iterable(list(endpoint_type)))

        elif endpoint_type is int:
            return Integer(field_name)

        elif endpoint_type is float:
            return Continuous(field_name)

        elif endpoint_type is bool:
            return Symbolic(field_name, Set.from_iterable([True, False]))

        else:
            raise NotImplementedError(
                f"No conversion between {endpoint_type} and random_events.Variable is known."
            )

    def create_fully_factorized_distribution(
        self,
        variables: List[Variable],
    ) -> ProbabilisticCircuit:
        """
        Create a fully factorized probabilistic circuit over the given variables.

        :return: A fully factorized probabilistic circuit.
        """
        distribution_variables = [v for v in variables if not isinstance(v, Integer)]

        return fully_factorized(
            distribution_variables,
            means={v: 0.0 for v in distribution_variables if isinstance(v, Continuous)},
            variances={
                v: 1.0 for v in distribution_variables if isinstance(v, Continuous)
            },
        )
