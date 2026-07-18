from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from typing_extensions import Dict, Optional

from experiments.confidence_aware_eql.exceptions import UnknownFeatureValueError


class FeatureKind(Enum):
    """The measurement type of a feature."""

    CONTINUOUS = "continuous"
    """A real-valued feature modelled by a continuous distribution."""

    CATEGORICAL = "categorical"
    """A discrete feature whose values are mapped to numeric codes."""


@dataclass
class Feature:
    """A single named dimension of an instance, such as weight or material.

    A feature knows how to turn a raw value into the numeric encoding the
    probabilistic circuit consumes. Continuous features pass their value
    through; categorical features look the value up in :attr:`categories`.
    """

    name: str
    """Identifier of the feature, used as the circuit variable name."""

    kind: FeatureKind = FeatureKind.CONTINUOUS
    """Whether the feature is continuous or categorical."""

    categories: Optional[Dict[str, float]] = None
    """Mapping from category label to numeric code, for categorical features."""

    def encode(self, value: object) -> float:
        """Return the numeric encoding of ``value`` for this feature.

        :raises UnknownFeatureValueError: if a categorical value is not known.
        """
        if self.kind is FeatureKind.CONTINUOUS:
            return float(value)
        if value not in self.categories:
            raise UnknownFeatureValueError(self.name, value)
        return float(self.categories[value])
