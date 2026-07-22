from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Optional, Type

from experiments.confidence_aware_eql.engine.circuit_model import (
    fit_gaussian_mixture_circuit,
)
from experiments.confidence_aware_eql.engine.evaluator import ConfidenceAwareEvaluator
from experiments.confidence_aware_eql.engine.schema import FeatureSchema
from experiments.confidence_aware_eql.engine.threshold import (
    FamiliarityThreshold,
    PercentileThreshold,
)
from experiments.confidence_aware_eql.engine.training import TrainingDataGenerator


@dataclass
class ConfidenceModelBuilder:
    """
    Assembles a confidence-aware evaluator from a set of cluster prototypes.

    The builder derives the feature schema from the instance class, generates familiar
    training instances from the prototypes, fits a mixture circuit on them, and
    calibrates the familiarity threshold on the training log-likelihoods.
    """

    instance_class: Type
    """The dataclass whose instances the evaluator will score."""

    generator: TrainingDataGenerator
    """
    The source of familiar training instances.
    """

    instances_per_prototype: int = 80
    """
    How many instances to draw from each prototype.
    """

    random_seed: int = 0
    """
    The seed used for generating the training set and fitting the mixture.
    """

    def build(
        self, threshold: Optional[FamiliarityThreshold] = None
    ) -> ConfidenceAwareEvaluator:
        """
        Build and return a fitted confidence-aware evaluator.

        :param threshold: The calibration strategy, defaulting to the first percentile
            of the training log-likelihoods.
        :return: An evaluator ready to score instances of the instance class.
        """
        if threshold is None:
            threshold = PercentileThreshold(percentile=1.0)
        schema = FeatureSchema.from_dataclass(self.instance_class)
        training_data = self.generator.sample(
            self.instances_per_prototype, schema.variables, self.random_seed
        )
        circuit = fit_gaussian_mixture_circuit(
            training_data, schema.variables, random_seed=self.random_seed
        )
        threshold.fit(circuit.log_likelihood(training_data))
        return ConfidenceAwareEvaluator(schema, circuit, threshold)
