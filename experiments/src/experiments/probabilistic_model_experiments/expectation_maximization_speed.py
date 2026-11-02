"""
Speed of expectation maximization on a mixture of factorized Gaussians compared to
scikit-learn's diagonal Gaussian mixture, both started from the same parameters and run
for the same number of iterations on scikit-learn datasets.
"""

from __future__ import annotations

# %% imports
import warnings
from dataclasses import dataclass, field

import numpy as np
from random_events.variable import Continuous
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
from typing_extensions import Any, Callable

from experiments.experiment_definitions import (
    ExperimentResult,
    ExperimentsTable,
    MeanAndStandardDeviation,
    TypstRenderer,
)
from experiments.probabilistic_circuit_experiments.gaussian_mixture_benchmark import (
    Dataset,
)
from experiments.probabilistic_model_experiments.layered_circuit_speed import (
    TimedCall,
)
from probabilistic_model.learning.expectation_maximization import (
    ExpectationMaximization,
    ExpectationMaximizationLearning,
)
from probabilistic_model.probabilistic_circuit.rx.helper import (
    factorized_gaussian_mixture,
)


# %% starting parameters
@dataclass
class StartingParameters:
    """
    The parameters both mixtures start from: equal weights, the means of randomly chosen
    rows and the standard deviation of every feature over all rows.
    """

    means: np.ndarray
    """
    The mean of every feature, one row per mixture component.
    """

    scales: np.ndarray
    """
    The standard deviation of every feature, one row per mixture component.
    """

    @classmethod
    def from_random_rows(
        cls,
        data: np.ndarray,
        number_of_components: int,
        minimum_scale: float,
        generator: np.random.Generator,
    ) -> StartingParameters:
        """
        Center every component at a distinct random row.

        :param data: One row per sample.
        :param number_of_components: Number of mixture components.
        :param minimum_scale: Lower bound for every standard deviation, so constant
            features keep a positive scale.
        :param generator: The source of the random rows.
        :return: The starting parameters.
        """
        rows = generator.choice(len(data), size=number_of_components, replace=False)
        scales = np.maximum(np.std(data, axis=0), minimum_scale)
        return cls(means=data[rows], scales=np.tile(scales, (number_of_components, 1)))


# %% results
@dataclass
class ExpectationMaximizationDurationResult(ExperimentResult):
    """
    Duration and quality of both fits on one dataset.
    """

    dataset: Dataset
    """
    The dataset both mixtures were fitted on.
    """

    number_of_rows: int
    """
    Number of rows of the dataset.
    """

    number_of_features: int
    """
    Number of features of the dataset.
    """

    number_of_components: int
    """
    Number of mixture components, one per class of the dataset.
    """

    expectation_maximization_duration: MeanAndStandardDeviation
    """
    Duration of a fit with :class:`ExpectationMaximizationLearning`.
    """

    scikit_learn_duration: MeanAndStandardDeviation
    """
    Duration of a fit with scikit-learn's :class:`GaussianMixture`.
    """

    expectation_maximization_average_log_likelihood: float
    """
    Average log-likelihood of the rows under the circuit fitted by expectation
    maximization.
    """

    scikit_learn_average_log_likelihood: float
    """
    Average log-likelihood of the rows under scikit-learn's fitted mixture.
    """


# %% comparison
@dataclass
class ExpectationMaximizationSpeedComparison:
    """
    Fits both mixtures repeatedly on a dataset and measures their durations and average
    log-likelihoods.
    """

    iterations: int = 20
    """
    Number of expectation maximization iterations of every fit.
    """

    repetitions: int = 5
    """
    Number of timed fits per dataset and method.
    """

    warmup_repetitions: int = 1
    """
    Number of untimed fits per dataset and method before the timed ones.
    """

    minimum_gaussian_variance: float = 1e-6
    """
    Lower bound for the variances expectation maximization learns; scikit-learn adds it
    to every variance instead.
    """

    generator: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(69)
    )
    """
    The source of the random starting rows.
    """

    def measure(self, dataset: Dataset) -> ExpectationMaximizationDurationResult:
        """
        Fit both mixtures on a dataset from the same starting parameters.

        :param dataset: A dataset whose target is a class.
        :return: The durations and average log-likelihoods of both fits.
        """
        bunch = dataset.bunch()
        data = bunch.data
        number_of_components = bunch.target.nunique()
        starting_parameters = StartingParameters.from_random_rows(
            data.to_numpy(),
            number_of_components,
            np.sqrt(self.minimum_gaussian_variance),
            self.generator,
        )

        learning = ExpectationMaximizationLearning(
            factorized_gaussian_mixture(
                [Continuous(name) for name in data.columns],
                starting_parameters.means,
                starting_parameters.scales,
            ),
            iterations=self.iterations,
            expectation_maximization=ExpectationMaximization(
                minimum_gaussian_variance=self.minimum_gaussian_variance
            ),
        )
        scikit_learn_mixture = GaussianMixture(
            n_components=number_of_components,
            covariance_type="diag",
            weights_init=np.full(number_of_components, 1 / number_of_components),
            means_init=starting_parameters.means,
            precisions_init=1 / starting_parameters.scales**2,
            max_iter=self.iterations,
            tol=0.0,
            reg_covar=self.minimum_gaussian_variance,
        )

        expectation_maximization_call = self.timed(lambda: learning.fit(data))
        scikit_learn_call = self.timed(
            lambda: scikit_learn_mixture.fit(data.to_numpy())
        )
        circuit = expectation_maximization_call.result
        circuit_data = data[[variable.name for variable in circuit.variables]]

        return ExpectationMaximizationDurationResult(
            dataset=dataset,
            number_of_rows=len(data),
            number_of_features=len(data.columns),
            number_of_components=number_of_components,
            expectation_maximization_duration=expectation_maximization_call.duration,
            scikit_learn_duration=scikit_learn_call.duration,
            expectation_maximization_average_log_likelihood=float(
                np.mean(circuit.log_likelihood(circuit_data.to_numpy()))
            ),
            scikit_learn_average_log_likelihood=float(
                scikit_learn_mixture.score(data.to_numpy())
            ),
        )

    def timed(self, fit: Callable[[], Any]) -> TimedCall:
        """
        Run a fit untimed :attr:`warmup_repetitions` times, then time it
        :attr:`repetitions` times.

        :param fit: The fit to run.
        :return: The duration of the timed fits and the result of the last one.
        """
        for _ in range(self.warmup_repetitions):
            fit()
        return TimedCall.of(fit, repeats=self.repetitions)


# %% main
def main():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    comparison = ExpectationMaximizationSpeedComparison()
    results = [
        comparison.measure(dataset)
        for dataset in (Dataset.IRIS, Dataset.WINE, Dataset.BREAST_CANCER)
    ]
    print(
        TypstRenderer(ExperimentsTable(results)).render_figure(
            f"Durations of {comparison.iterations} expectation maximization iterations "
            f"on a mixture of factorized Gaussians with one component per class, "
            f"the rustworkx circuit vs scikit-learn's diagonal Gaussian mixture, both "
            f"started from the same parameters."
        )
    )


if __name__ == "__main__":
    main()
