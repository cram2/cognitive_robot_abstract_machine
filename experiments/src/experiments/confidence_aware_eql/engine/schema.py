from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum

import numpy as np
from random_events.variable import Continuous, Variable
from typing_extensions import Any, List, Type


@dataclass
class FeatureSchema:
    """
    The ordered random variables that describe one kind of instance.

    The schema is derived from a dataclass whose public fields are the features,
    and it turns an instance of that class into the numeric row that a
    probabilistic model scores.

    .. note::
        Every field becomes a :class:`random_events.variable.Continuous`
        variable. Enumeration fields are represented by the numeric value of
        their member, because the Gaussian mixture used here can only place
        continuous distributions on its leaves. A model with symbolic leaves,
        such as a joint probability tree, would allow
        :class:`random_events.variable.Symbolic` variables instead.
    """

    variables: List[Variable]
    """The random variables of the schema, in the order used for encoding."""

    @property
    def variable_names(self) -> List[str]:
        """
        The variable names in schema order.
        """
        return [variable.name for variable in self.variables]

    @classmethod
    def from_dataclass(cls, instance_class: Type) -> FeatureSchema:
        """
        Derive a schema from the public fields of a dataclass.

        Fields whose names begin with an underscore are treated as internal and
        excluded, so framework-injected attributes never become features.

        :param instance_class: The dataclass whose fields describe the features.
        :return: The schema over the public fields of the class.
        """
        variables = [
            Continuous(field_definition.name)
            for field_definition in fields(instance_class)
            if not field_definition.name.startswith("_")
        ]
        return cls(variables)

    def encode(self, instance: Any) -> np.ndarray:
        """
        Encode an instance into a numeric row in schema order.

        Features whose attribute is ``None`` are encoded as ``numpy.nan``, which marks
        them as unobserved so the caller can marginalise them out.

        :param instance: The instance whose attributes are read by name.
        :return: One row of feature values in schema order.
        """
        return np.array(
            [
                self._encode_value(getattr(instance, variable.name))
                for variable in self.variables
            ],
            dtype=float,
        )

    def observed_variables(self, row: np.ndarray) -> List[Variable]:
        """
        Return the variables that were actually observed in a row.

        :param row: An encoded row, where unobserved features are ``numpy.nan``.
        :return: The variables whose value is present.
        """
        return [
            variable
            for variable, value in zip(self.variables, row)
            if not np.isnan(value)
        ]

    @staticmethod
    def _encode_value(value: Any) -> float:
        """
        Encode a single attribute value to its numeric representation.

        :param value: The attribute value, possibly ``None`` or an enumeration member.
        :return: The numeric encoding, or ``numpy.nan`` when the value is absent.
        """
        if value is None:
            return np.nan
        if isinstance(value, Enum):
            return float(value.value)
        return float(value)
