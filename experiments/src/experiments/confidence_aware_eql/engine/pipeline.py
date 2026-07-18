from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import List, Type

from experiments.confidence_aware_eql.engine.circuit_model import (
    GaussianMixtureCircuit,
    fit_gaussian_mixture_circuit,
)
from experiments.confidence_aware_eql.engine.evaluator import ConfidenceAwareEvaluator
from experiments.confidence_aware_eql.engine.schema import FeatureSchema
from experiments.confidence_aware_eql.engine.threshold import (
    FamiliarityThreshold,
    PercentileThreshold,
)
from experiments.confidence_aware_eql.engine.training import TrainingSampler


@dataclass
class ConfidenceModelBuilder:
    """Assembles a confidence-aware evaluator from a domain's training sampler.

    The builder derives the feature schema from the instance class, samples
    familiar training instances, fits a Gaussian mixture circuit, and fits a
    familiarity threshold on the training log-likelihoods.
    """

    instance_class: Type
    """The dataclass whose instances the evaluator will score."""

    sampler: TrainingSampler
    """The source of familiar training instances."""

    instances_per_prototype: int = 80
    """How many instances to draw around each prototype."""

    random_seed: int = 0
    """Seed used for sampling and mixture fitting."""

    def build(
        self, threshold: FamiliarityThreshold = None
    ) -> ConfidenceAwareEvaluator:
        """Build and return a fitted confidence-aware evaluator."""
        if threshold is None:
            threshold = PercentileThreshold(percentile=1.0)
        schema = FeatureSchema.from_dataclass(self.instance_class)
        instances = self.sampler.sample(self.instances_per_prototype, self.random_seed)
        training_matrix = self.sampler.encode_all(schema, instances)
        circuit = fit_gaussian_mixture_circuit(
            training_matrix, schema.feature_names, random_seed=self.random_seed
        )
        threshold.fit(circuit.log_likelihood(training_matrix))
        return ConfidenceAwareEvaluator(schema, circuit, threshold)
