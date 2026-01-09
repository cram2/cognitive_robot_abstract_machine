from typing import List
from datetime import datetime
from random_events.variable import Continuous, Integer, Symbolic, Variable
from random_events.set import Set
from ..class_diagrams.class_diagram import WrappedClass
from ..class_diagrams.wrapped_field import WrappedField
from probabilistic_model.probabilistic_circuit.rx.helper import fully_factorized
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)


class Parameterizer:
    """
    Parameterizer for creating random event variables from WrappedClass instances.
    """

    def __init__(self):
        self._variables: List[Variable] = []

    def __call__(self, wrapped_class: WrappedClass) -> List[Variable]:
        """
        Parameterize a WrappedClass instance and return its variables.
        """
        self._variables = []
        self._parameterize_wrapped_class(
            wrapped_class, prefix=wrapped_class.clazz.__name__
        )
        return self._variables

    def _parameterize_wrapped_class(self, wrapped_class: WrappedClass, prefix: str):
        """
        Parameterize all fields of a WrappedClass recursively.
        """
        for wrapped_field in wrapped_class.fields:
            self._parameterize_wrapped_field(wrapped_field, prefix)

    def _parameterize_wrapped_field(self, wrapped_field: WrappedField, prefix: str):
        """
        Parameterize a single WrappedField.
        """
        field_name = f"{prefix}.{wrapped_field.name}"
        endpoint_type = wrapped_field.type_endpoint

        if endpoint_type is datetime:
            return

        if wrapped_field.is_optional:
            endpoint_type = wrapped_field.contained_type

        if (
            wrapped_field.is_one_to_one_relationship
            and wrapped_field.clazz._class_diagram
        ):
            if not wrapped_field.is_enum:
                target_wrapped_class: WrappedClass = (
                    wrapped_field.clazz._class_diagram.get_wrapped_class(endpoint_type)
                )
                self._parameterize_wrapped_class(
                    target_wrapped_class, prefix=field_name
                )
                return

        if wrapped_field.is_enum:
            enum_values = list(endpoint_type)
            self._variables.append(Symbolic(field_name, Set.from_iterable(enum_values)))

        elif endpoint_type == int:
            self._variables.append(Integer(field_name))

        elif endpoint_type == float:
            self._variables.append(Continuous(field_name))

    def create_fully_factorized_distribution(
        self,
        variables: List[Variable],
    ) -> ProbabilisticCircuit:
        """
        Create a fully factorized probabilistic circuit over the given variables.
        """
        return fully_factorized(
            variables,
            means={v: 0.0 for v in variables if isinstance(v, Continuous)},
            variances={v: 1.0 for v in variables if isinstance(v, Continuous)},
        )
