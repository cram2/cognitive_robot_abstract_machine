"""
Speed of the layered numpy circuit compared to the rustworkx circuit that the Gaussian
mixture learning method creates, whose components are multivariate Gaussian leaves: the
queries that leave the circuit as it is, truncation to boxes, the queries of the
truncated circuit, and conditioning on a partial point, for a growing number of
components.
"""

from __future__ import annotations

import enum
import functools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from random_events.interval import Interval, closed
from random_events.product_algebra import Event, SimpleEvent
from random_events.variable import Continuous
from sklearn.mixture import GaussianMixture
from typing_extensions import List, Self

from experiments.experiment_definitions import (
    ExperimentResult,
    ExperimentsTable,
    MeanAndStandardDeviation,
    TypstRenderer,
)
from experiments.probabilistic_model_experiments.layered_circuit_speed import (
    TimedCall,
    raise_if_answers_disagree,
    speedup_of,
    staircase_of_boxes,
)
from krrood.entity_query_language.backends import (
    EntityQueryLanguageGenerativeBackend,
)
from krrood.entity_query_language.factories import a, variable_from
from probabilistic_model.adapters.rustworkx_tensorized.rustworkx_to_tensorized import (
    RustworkxCircuitToLayeredCircuitConverter,
)
from probabilistic_model.learning.gaussian_mixture.gaussian_mixture_model import (
    GaussianMixtureModel,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit as RustworkxProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)


class GaussianMixtureQuery(enum.Enum):
    """
    The queries measured on both circuits.
    """

    LOG_LIKELIHOOD = "log_likelihood, 1000 events"
    LOG_LIKELIHOOD_OF_MANY = "log_likelihood, 10000 events"
    SAMPLE = "sample, 1000"
    PROBABILITY_OF_A_BOX = "probability of a box"
    TRUNCATION_TO_A_BOX = "truncation to a box"
    TRUNCATION_TO_SEVERAL_BOXES = "truncation to several boxes"
    LOG_LIKELIHOOD_AFTER_TRUNCATION = "log_likelihood after truncation, 1000 events"
    SAMPLE_AFTER_TRUNCATION = "sample after truncation, 1000"
    CONDITIONING_ON_ONE_VARIABLE = "conditioning on one variable"
    CONDITIONING_ON_ALL_BUT_ONE_VARIABLE = "conditioning on all but one variable"


@dataclass
class GaussianMixtureConfiguration:
    """
    The size of one Gaussian mixture the benchmark measures, and how its answers are
    compared.
    """

    number_of_components: int
    """
    How many Gaussians the mixture has.
    """

    number_of_boxes: int = 10
    """
    How many disjoint boxes the event of the batched truncation is made of.
    """

    numerical_integration_tolerance: float = 1e-3
    """
    The absolute difference allowed between two probabilities that
    :mod:`scipy.stats.multivariate_normal` integrated numerically.
    """

    @classmethod
    def variants(cls) -> List[Self]:
        """
        :return: The mixtures the benchmark measures, from the fewest components to the
            most.
        """
        configurations = a(cls)(number_of_components=variable_from([5, 20, 80]))
        return list(
            configurations.evaluate(backend=EntityQueryLanguageGenerativeBackend())
        )


@dataclass
class ClusteredSamplesFactory:
    """
    Draws samples from well separated, correlated clusters and fits a Gaussian mixture
    to them.
    """

    number_of_variables: int = 4
    """
    Number of continuous variables of the samples.
    """

    number_of_samples: int = 10000
    """
    Number of samples drawn to fit the mixture.
    """

    def samples(self, number_of_clusters: int) -> pd.DataFrame:
        """
        :param number_of_clusters: How many clusters to draw from.
        :return: The samples, one column per variable.
        """
        generator = np.random.default_rng(69)
        per_cluster = self.number_of_samples // number_of_clusters
        clusters = []
        for _ in range(number_of_clusters):
            factor = generator.uniform(-1, 1, (self.number_of_variables,) * 2)
            clusters.append(
                generator.multivariate_normal(
                    generator.uniform(-10, 10, self.number_of_variables),
                    factor @ factor.T + 0.1 * np.eye(self.number_of_variables),
                    per_cluster,
                )
            )
        return pd.DataFrame(
            np.concatenate(clusters),
            columns=[f"x_{index}" for index in range(self.number_of_variables)],
        )

    def learn_circuit(self, number_of_components: int) -> RustworkxProbabilisticCircuit:
        """
        :param number_of_components: How many Gaussians the mixture has.
        :return: The mixture as the Gaussian mixture learning method creates it.
        """
        return GaussianMixtureModel(
            model=GaussianMixture(n_components=number_of_components, random_state=0)
        ).fit(self.samples(number_of_components))


@dataclass
class GaussianMixtureQueryDurationResult(ExperimentResult):
    """
    Wall-clock duration of one query on a Gaussian mixture, rustworkx vs the layered
    numpy circuit.
    """

    number_of_components: int
    """
    How many Gaussians the mixture has.
    """

    query: GaussianMixtureQuery
    """
    The query.
    """

    rustworkx_duration: MeanAndStandardDeviation
    """
    Duration of the runs on the rustworkx circuit.
    """

    layered_duration: MeanAndStandardDeviation
    """
    Duration of the runs on the layered circuit.
    """

    speedup: float
    """
    How many times faster the layered circuit answered than rustworkx, by mean duration.
    """


@dataclass
class GaussianMixtureBenchmark:
    """
    The measurements on one mixture, converted once into a layered circuit.
    """

    configuration: GaussianMixtureConfiguration
    """
    The size of the mixture and how its answers are compared.
    """

    rustworkx_circuit: RustworkxProbabilisticCircuit
    """
    The mixture as the learning method created it.
    """

    layered: LayeredProbabilisticCircuit
    """
    The same mixture as a layered circuit.
    """

    @property
    def variables(self) -> List[Continuous]:
        """
        :return: The variables of the mixture, in the order of the circuit.
        """
        return list(self.layered.variables)

    def result_of(
        self,
        query: GaussianMixtureQuery,
        rustworkx_call: TimedCall,
        layered_call: TimedCall,
    ) -> GaussianMixtureQueryDurationResult:
        """
        :param query: The query that was timed.
        :param rustworkx_call: The timed query on the rustworkx circuit.
        :param layered_call: The timed query on the layered circuit.
        :return: The result.
        """
        return GaussianMixtureQueryDurationResult(
            number_of_components=self.configuration.number_of_components,
            query=query,
            rustworkx_duration=rustworkx_call.duration,
            layered_duration=layered_call.duration,
            speedup=speedup_of(rustworkx_call.duration, layered_call.duration),
        )

    def central_box(self) -> Event:
        """
        :return: A box that bounds the first two variables around the middle of the
            data and leaves the others free.
        """
        first, second = self.variables[:2]
        return SimpleEvent.from_data(
            {
                first: self.window_of(first, 0.25, 0.75),
                second: self.window_of(second, 0.25, 0.75),
            }
        ).as_composite_set()

    @functools.cached_property
    def range_of_samples(self) -> np.ndarray:
        """
        :return: The 1% and 99% quantile of every variable in samples of the mixture,
            shape (2, #variables).
        """
        return np.quantile(self.rustworkx_circuit.sample(2000), [0.01, 0.99], axis=0)

    def window_of(self, variable: Continuous, lower: float, upper: float) -> Interval:
        """
        :param variable: A variable of the circuit.
        :param lower: Where the window starts, as a fraction of the range of samples.
        :param upper: Where the window ends, as a fraction of the range of samples.
        :return: The window.
        """
        low, high = self.range_of_samples[:, self.variables.index(variable)]
        return closed(low + lower * (high - low), low + upper * (high - low))

    def measure(self) -> List[GaussianMixtureQueryDurationResult]:
        """
        :return: One result per query.
        """
        return (
            self.measure_queries_without_truncation()
            + self.measure_truncation()
            + self.measure_conditioning()
        )

    def measure_queries_without_truncation(
        self,
    ) -> List[GaussianMixtureQueryDurationResult]:
        """
        Time the likelihood, sampling and probability of a box on both circuits, which
        leave the circuit as it is.

        :return: One result per query.
        """
        rustworkx_circuit, layered = self.rustworkx_circuit, self.layered
        results = []
        for query, amount in (
            (GaussianMixtureQuery.LOG_LIKELIHOOD, 1000),
            (GaussianMixtureQuery.LOG_LIKELIHOOD_OF_MANY, 10000),
        ):
            samples = rustworkx_circuit.sample(amount)
            rustworkx_call = TimedCall.of(
                lambda: rustworkx_circuit.log_likelihood(samples)
            )
            layered_call = TimedCall.of(lambda: layered.log_likelihood(samples))
            raise_if_answers_disagree(
                query.value, rustworkx_call.result, layered_call.result
            )
            results.append(self.result_of(query, rustworkx_call, layered_call))

        results.append(
            self.result_of(
                GaussianMixtureQuery.SAMPLE,
                TimedCall.of(lambda: rustworkx_circuit.sample(1000)),
                TimedCall.of(lambda: layered.sample(1000)),
            )
        )

        [box] = self.central_box().simple_sets
        box = box.__deepcopy__()
        box.fill_missing_variables(set(self.variables))
        rustworkx_call = TimedCall.of(
            lambda: rustworkx_circuit.probability_of_simple_event(box)
        )
        layered_call = TimedCall.of(lambda: layered.probability_of_simple_event(box))
        raise_if_answers_disagree(
            GaussianMixtureQuery.PROBABILITY_OF_A_BOX.value,
            rustworkx_call.result,
            layered_call.result,
            self.configuration.numerical_integration_tolerance,
        )
        results.append(
            self.result_of(
                GaussianMixtureQuery.PROBABILITY_OF_A_BOX, rustworkx_call, layered_call
            )
        )
        return results

    def measure_truncation(self) -> List[GaussianMixtureQueryDurationResult]:
        """
        Time truncating both circuits to one box and to several boxes, and the
        likelihood and sampling of the circuit truncated to one box.

        :return: One result per query.
        """
        rustworkx_circuit, layered = self.rustworkx_circuit, self.layered
        box = self.central_box()
        rustworkx_call = TimedCall.of(
            lambda: rustworkx_circuit.truncated(box.__deepcopy__())
        )
        layered_call = TimedCall.of(lambda: layered.truncated(box.__deepcopy__()))
        rustworkx_truncated, rustworkx_probability = rustworkx_call.result
        layered_truncated, layered_probability = layered_call.result
        raise_if_answers_disagree(
            GaussianMixtureQuery.TRUNCATION_TO_A_BOX.value,
            rustworkx_probability,
            layered_probability,
            self.configuration.numerical_integration_tolerance,
        )
        results = [
            self.result_of(
                GaussianMixtureQuery.TRUNCATION_TO_A_BOX, rustworkx_call, layered_call
            )
        ]

        first, second = self.variables[:2]
        staircase = staircase_of_boxes(
            first,
            second,
            self.window_of(first, 0.0, 1.0).simple_sets[0],
            self.window_of(second, 0.0, 1.0).simple_sets[0],
            self.configuration.number_of_boxes,
        )
        rustworkx_call = TimedCall.of(
            lambda: rustworkx_circuit.truncated(staircase.__deepcopy__()), repeats=1
        )
        layered_call = TimedCall.of(
            lambda: layered.truncated(staircase.__deepcopy__()), repeats=1
        )
        raise_if_answers_disagree(
            GaussianMixtureQuery.TRUNCATION_TO_SEVERAL_BOXES.value,
            rustworkx_call.result[1],
            layered_call.result[1],
            self.configuration.numerical_integration_tolerance,
        )
        results.append(
            self.result_of(
                GaussianMixtureQuery.TRUNCATION_TO_SEVERAL_BOXES,
                rustworkx_call,
                layered_call,
            )
        )

        samples = rustworkx_truncated.sample(1000)
        rustworkx_call = TimedCall.of(
            lambda: rustworkx_truncated.log_likelihood(samples)
        )
        layered_call = TimedCall.of(lambda: layered_truncated.log_likelihood(samples))
        raise_if_answers_disagree(
            GaussianMixtureQuery.LOG_LIKELIHOOD_AFTER_TRUNCATION.value,
            rustworkx_call.result,
            layered_call.result,
            absolute_tolerance=1e-2,
        )
        results.append(
            self.result_of(
                GaussianMixtureQuery.LOG_LIKELIHOOD_AFTER_TRUNCATION,
                rustworkx_call,
                layered_call,
            )
        )
        results.append(
            self.result_of(
                GaussianMixtureQuery.SAMPLE_AFTER_TRUNCATION,
                TimedCall.of(lambda: rustworkx_truncated.sample(1000), repeats=1),
                TimedCall.of(lambda: layered_truncated.sample(1000), repeats=1),
            )
        )
        return results

    def measure_conditioning(self) -> List[GaussianMixtureQueryDurationResult]:
        """
        Time conditioning both circuits on a sample of one variable and of all but one
        variable.

        :return: One result per query.
        """
        rustworkx_circuit, layered = self.rustworkx_circuit, self.layered
        sample = rustworkx_circuit.sample(1)[0]
        results = []
        for query, number_of_conditioned in (
            (GaussianMixtureQuery.CONDITIONING_ON_ONE_VARIABLE, 1),
            (
                GaussianMixtureQuery.CONDITIONING_ON_ALL_BUT_ONE_VARIABLE,
                len(self.variables) - 1,
            ),
        ):
            point = {
                variable: sample[index]
                for index, variable in enumerate(self.variables[:number_of_conditioned])
            }
            rustworkx_call = TimedCall.of(
                lambda: rustworkx_circuit.log_conditional(point)
            )
            layered_call = TimedCall.of(lambda: layered.log_conditional(point))
            raise_if_answers_disagree(
                query.value, rustworkx_call.result[1], layered_call.result[1]
            )
            results.append(self.result_of(query, rustworkx_call, layered_call))
        return results


def main():
    factory = ClusteredSamplesFactory()
    results = []
    for configuration in GaussianMixtureConfiguration.variants():
        rustworkx_circuit = factory.learn_circuit(configuration.number_of_components)
        layered = RustworkxCircuitToLayeredCircuitConverter.convert(rustworkx_circuit)
        results.extend(
            GaussianMixtureBenchmark(
                configuration, rustworkx_circuit, layered
            ).measure()
        )
    print(
        TypstRenderer(ExperimentsTable(results)).render_figure(
            f"Query durations on a Gaussian mixture over "
            f"{factory.number_of_variables} variables with a growing number of "
            f"components, rustworkx vs the layered numpy circuit."
        )
    )


if __name__ == "__main__":
    main()
