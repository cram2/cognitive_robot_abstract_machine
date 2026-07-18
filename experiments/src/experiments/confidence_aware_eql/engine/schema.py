from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum

import numpy as np
from random_events.variable import Continuous
from typing_extensions import List, Optional, Type


@dataclass
class EncodedInstance:
    """The numeric encoding of one instance against a feature schema."""

    row: Optional[np.ndarray]
    """Feature values in schema order, or ``None`` when the instance is incomplete."""

    missing_features: List[str] = field(default_factory=list)
    """Names of features that were absent on the instance."""

    @property
    def is_complete(self) -> bool:
        """Whether every feature was present and encodable."""
        return not self.missing_features


@dataclass
class FeatureSchema:
    """The ordered random variables that describe one kind of instance.

    A schema is derived from a dataclass whose public fields are the features.
    Each field becomes a :class:`random_events.variable.Continuous` variable, so
    the variables used for encoding are exactly the ones the probabilistic
    circuit is built over. Enumeration fields are encoded by the numeric value of
    their member, which keeps the whole feature vector continuous and therefore
    representable by the Gaussian mixture.
    """

    variables: List[Continuous]
    """The random variables of the schema, in the order used for encoding."""

    @property
    def feature_names(self) -> List[str]:
        """The variable names in schema order."""
        return [variable.name for variable in self.variables]

    @classmethod
    def from_dataclass(cls, instance_class: Type) -> FeatureSchema:
        """Derive a schema from the public fields of a dataclass.

        Fields whose names begin with an underscore are treated as internal and
        excluded, so framework-injected attributes never become features.

        :param instance_class: The dataclass whose fields describe the features.
        """
        variables = [
            Continuous(field_definition.name)
            for field_definition in fields(instance_class)
            if not field_definition.name.startswith("_")
        ]
        return cls(variables)

    def encode(self, instance: object) -> EncodedInstance:
        """Encode an instance into a numeric row in schema order.

        A feature is recorded as missing when its attribute is ``None``, so an
        incomplete instance is reported rather than silently mis-encoded.

        :param instance: The instance whose attributes are read by name.
        """
        missing_features: List[str] = []
        values: List[float] = []
        for variable in self.variables:
            value = getattr(instance, variable.name)
            if value is None:
                missing_features.append(variable.name)
                continue
            values.append(self._encode_value(value))
        if missing_features:
            return EncodedInstance(row=None, missing_features=missing_features)
        return EncodedInstance(row=np.array(values, dtype=float))

    @staticmethod
    def _encode_value(value: object) -> float:
        """Encode a single attribute value to its numeric representation."""
        if isinstance(value, Enum):
            return float(value.value)
        return float(value)
