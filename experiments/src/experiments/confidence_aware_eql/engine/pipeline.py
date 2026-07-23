from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import List, Optional, Type

from experiments.confidence_aware_eql.engine.circuit_model import (
    fit_gaussian_mixture_circuit,
)
from experiments.confidence_aware_eql.engine.evaluator import ConfidenceAwareEvaluator
from experiments.confidence_aware_eql.engine.schema import FeatureSchema
from experiments.confidence_aware_eql.engine.threshold import (
    FamiliarityThreshold,
    PercentileThreshold,
)
from experiments.confidence_aware_eql.engine.training import (
    FamiliarCluster,
    TrainingDataGenerator,
)


@dataclass
class ConfidenceModelBuilder:
    """
    Assembles a confidence-aware evaluator from a set of familiar clusters.

    The builder derives the feature schema from the instance class, generates familiar
    training instances from the clusters through the generative backend of the query
    language, fits a mixture circuit on them, and calibrates the familiarity threshold
    on the training log-likelihoods.
    """

    instance_class: Type
    """The dataclass whose instances the evaluator will score."""

    clusters: List[FamiliarCluster]
    """
    The clusters of familiar instances the model is learned from.
    """

    instances_per_cluster: int = 80
    """
    How many instances to draw from each cluster.
    """

    random_seed: int = 0
    """
    The seed used for fitting the mixture.
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
        generator = TrainingDataGenerator(schema, self.clusters)
        training_data = generator.sample(self.instances_per_cluster)
        circuit = fit_gaussian_mixture_circuit(
            training_data, schema.variables, random_seed=self.random_seed
        )
        threshold.fit(circuit.log_likelihood(training_data))
        return ConfidenceAwareEvaluator(schema, circuit, threshold)
