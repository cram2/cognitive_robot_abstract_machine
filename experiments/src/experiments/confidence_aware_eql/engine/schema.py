from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum

import numpy as np
from typing_extensions import Dict, List, Optional, Type

from experiments.confidence_aware_eql.engine.feature import Feature, FeatureKind


@dataclass
class EncodedInstance:
    """The numeric encoding of one instance against a feature schema."""

    row: Optional[np.ndarray]
    """Feature values in schema order, or ``None`` when the instance is incomplete."""

    missing_features: List[str] = field(default_factory=list)
    """Names of features that were absent or carried an unrecognised value."""

    @property
    def is_complete(self) -> bool:
        """Whether every feature was present and encodable."""
        return not self.missing_features


@dataclass
class FeatureSchema:
    """The ordered features that describe one kind of instance.

    A schema is derived from a dataclass whose public fields are the features:
    floating-point fields become continuous features and enumeration fields
    become categorical features. The schema then encodes instances of that class
    into the numeric rows the probabilistic circuit consumes.
    """

    features: List[Feature]
    """The features in the order used for encoding."""

    @property
    def feature_names(self) -> List[str]:
        """The feature names in schema order."""
        return [feature.name for feature in self.features]

    @classmethod
    def from_dataclass(cls, instance_class: Type) -> FeatureSchema:
        """Derive a schema from the public fields of a dataclass.

        Fields whose names begin with an underscore are treated as internal and
        excluded, so framework-injected attributes never become features.
        """
        features = [
            cls._feature_from_field(field_definition.name, field_definition.type)
            for field_definition in fields(instance_class)
            if not field_definition.name.startswith("_")
        ]
        return cls(features)

    def encode(self, instance: object) -> EncodedInstance:
        """Encode an instance into a numeric row in schema order.

        A feature is recorded as missing when its attribute is ``None``, so an
        incomplete instance is reported rather than silently mis-encoded.
        """
        missing_features: List[str] = []
        values: List[float] = []
        for feature in self.features:
            value = getattr(instance, feature.name)
            if value is None:
                missing_features.append(feature.name)
                continue
            values.append(self._encode_value(feature, value))
        if missing_features:
            return EncodedInstance(row=None, missing_features=missing_features)
        return EncodedInstance(row=np.array(values, dtype=float))

    @staticmethod
    def _feature_from_field(name: str, field_type: type) -> Feature:
        """Build a feature from a dataclass field's name and type."""
        if isinstance(field_type, type) and issubclass(field_type, Enum):
            categories = {member.name: float(member.value) for member in field_type}
            return Feature(name, FeatureKind.CATEGORICAL, categories)
        return Feature(name, FeatureKind.CONTINUOUS)

    @staticmethod
    def _encode_value(feature: Feature, value: object) -> float:
        """Encode a single attribute value to its numeric representation."""
        if isinstance(value, Enum):
            return float(value.value)
        return float(value)
