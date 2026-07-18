from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum

import numpy as np
from typing_extensions import Dict, List, Type

from experiments.confidence_aware_eql.engine.schema import FeatureSchema


@dataclass
class InstancePrototype:
    """A representative instance together with the spread of its continuous features.

    A prototype describes one cluster of familiar instances, such as "a cup".
    Sampling draws continuous features from a normal distribution around the
    prototype's values and keeps categorical features fixed.
    """

    instance: object
    """The representative instance whose feature values anchor the cluster."""

    continuous_spread: Dict[str, float]
    """Standard deviation to sample around each continuous feature."""


@dataclass
class TrainingSampler:
    """Generates familiar training instances from a set of prototypes."""

    instance_class: Type
    """The dataclass whose instances are produced."""

    prototypes: List[InstancePrototype]
    """The clusters of familiar instances to sample from."""

    def sample(self, instances_per_prototype: int, random_seed: int = 0) -> List[object]:
        """Return sampled familiar instances drawn around every prototype."""
        generator = np.random.default_rng(random_seed)
        return [
            self._sample_one(prototype, generator)
            for prototype in self.prototypes
            for _ in range(instances_per_prototype)
        ]

    def encode_all(self, schema: FeatureSchema, instances: List[object]) -> np.ndarray:
        """Encode instances into a training matrix in schema order."""
        return np.vstack([schema.encode(instance).row for instance in instances])

    def _sample_one(self, prototype: InstancePrototype, generator: np.random.Generator) -> object:
        """Draw a single instance around a prototype."""
        arguments = {}
        for field_definition in fields(self.instance_class):
            if field_definition.name.startswith("_"):
                continue
            value = getattr(prototype.instance, field_definition.name)
            arguments[field_definition.name] = self._sample_value(
                field_definition.name, value, prototype, generator
            )
        return self.instance_class(**arguments)

    @staticmethod
    def _sample_value(
        name: str,
        value: object,
        prototype: InstancePrototype,
        generator: np.random.Generator,
    ) -> object:
        """Sample one feature value, jittering continuous features only."""
        if isinstance(value, Enum):
            return value
        spread = prototype.continuous_spread[name]
        return float(generator.normal(value, spread))
