"""
Times fitting a mixture of factorized Gaussians with expectation maximization on scikit-
learn datasets, against scikit-learn's diagonal Gaussian mixture started from the same
parameters and run for the same number of iterations.
"""

from __future__ import annotations

# %% imports
import warnings
from dataclasses import asdict, dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd
import sklearn.datasets
from random_events.variable import Continuous
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
from sklearn.utils import Bunch
from typing_extensions import Any, Callable

from probabilistic_model.learning.expectation_maximization import (
    ExpectationMaximization,
    ExpectationMaximizationLearning,
)
from probabilistic_model.probabilistic_circuit.rx.helper import (
    factorized_gaussian_mixture,
)
from probabilistic_model.utils import timeit


# %% datasets
class Dataset(StrEnum):
    """
    The scikit-learn datasets with continuous features the comparison runs on.
    """

    IRIS = "iris"
    WINE = "wine"
    BREAST_CANCER = "breast_cancer"
    DIGITS = "digits"

    def load(self) -> Bunch:
        """
        Load the dataset with its features as a dataframe.

        :return: The features under ``data`` and the class of every row under
            ``target``.
        """
        match self:
            case Dataset.IRIS:
                return sklearn.datasets.load_iris(as_frame=True)
            case Dataset.WINE:
                return sklearn.datasets.load_wine(as_frame=True)
            case Dataset.BREAST_CANCER:
                return sklearn.datasets.load_breast_cancer(as_frame=True)
            case Dataset.DIGITS:
                return sklearn.datasets.load_digits(as_frame=True)


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
        component_count: int,
        minimum_scale: float,
        generator: np.random.Generator,
    ) -> StartingParameters:
        """
        Center every component at a distinct random row.

        :param data: One row per sample.
        :param component_count: Number of mixture components.
        :param minimum_scale: Lower bound for every standard deviation, so constant
            features keep a positive scale.
        :param generator: The source of the random rows.
        :return: The starting parameters.
        """
        rows = generator.choice(len(data), size=component_count, replace=False)
        scales = np.maximum(np.std(data, axis=0), minimum_scale)
        return cls(means=data[rows], scales=np.tile(scales, (component_count, 1)))


# %% timing
@dataclass
class ComparisonResult:
    """
    The timing and quality of both fits on one dataset.
    """

    dataset: Dataset
    """
    The dataset both mixtures were fitted on.
    """

    row_count: int
    """
    Number of rows of the dataset.
    """

    feature_count: int
    """
    Number of features of the dataset.
    """

    component_count: int
    """
    Number of mixture components, one per class of the dataset.
    """

    expectation_maximization_seconds: float
    """
    Mean duration of a fit with :class:`ExpectationMaximizationLearning`.
    """

    expectation_maximization_seconds_deviation: float
    """
    Standard deviation of the duration of a fit with
    :class:`ExpectationMaximizationLearning`.
    """

    scikit_learn_seconds: float
    """
    Mean duration of a fit with scikit-learn's :class:`GaussianMixture`.
    """

    scikit_learn_seconds_deviation: float
    """
    Standard deviation of the duration of a fit with scikit-learn's
    :class:`GaussianMixture`.
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


@dataclass
class ExpectationMaximizationSpeedComparison:
    """
    Fits both mixtures repeatedly on every dataset and reports their durations and
    average log-likelihoods.
    """

    datasets: tuple[Dataset, ...] = tuple(Dataset)
    """
    The datasets to compare on.
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

    def run(self) -> pd.DataFrame:
        """
        Compare both methods on every dataset.

        :return: One row per dataset with the fields of :class:`ComparisonResult`.
        """
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        return pd.DataFrame(
            [asdict(self.compare_on(dataset)) for dataset in self.datasets]
        )

    def compare_on(self, dataset: Dataset) -> ComparisonResult:
        """
        Fit both mixtures on a dataset from the same starting parameters.

        :param dataset: The dataset to fit.
        :return: The durations and average log-likelihoods of both fits.
        """
        bunch = dataset.load()
        data = bunch.data
        component_count = bunch.target.nunique()
        starting_parameters = StartingParameters.from_random_rows(
            data.to_numpy(),
            component_count,
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
            n_components=component_count,
            covariance_type="diag",
            weights_init=np.full(component_count, 1 / component_count),
            means_init=starting_parameters.means,
            precisions_init=1 / starting_parameters.scales**2,
            max_iter=self.iterations,
            tol=0.0,
            reg_covar=self.minimum_gaussian_variance,
        )

        circuit, expectation_maximization_seconds = self._time(learning.fit, data)
        _, scikit_learn_seconds = self._time(scikit_learn_mixture.fit, data.to_numpy())
        circuit_data = data[[variable.name for variable in circuit.variables]]

        return ComparisonResult(
            dataset=dataset,
            row_count=len(data),
            feature_count=len(data.columns),
            component_count=component_count,
            expectation_maximization_seconds=float(
                np.mean(expectation_maximization_seconds)
            ),
            expectation_maximization_seconds_deviation=float(
                np.std(expectation_maximization_seconds)
            ),
            scikit_learn_seconds=float(np.mean(scikit_learn_seconds)),
            scikit_learn_seconds_deviation=float(np.std(scikit_learn_seconds)),
            expectation_maximization_average_log_likelihood=float(
                np.mean(circuit.log_likelihood(circuit_data.to_numpy()))
            ),
            scikit_learn_average_log_likelihood=float(
                scikit_learn_mixture.score(data.to_numpy())
            ),
        )

    def _time(self, fit: Callable[[Any], Any], data: Any) -> tuple[Any, list[float]]:
        """
        Run a fit repeatedly, timing all but the warm-up runs.

        :param fit: The fit to run.
        :param data: What the fit is given.
        :return: The result of the last fit and the duration of every timed fit in
            seconds.
        """
        timed_fit = timeit(fit)
        seconds = []
        for repetition in range(self.warmup_repetitions + self.repetitions):
            result, duration = timed_fit(data)
            if repetition >= self.warmup_repetitions:
                seconds.append(duration.total_seconds())
        return result, seconds


# %% main
if __name__ == "__main__":
    with pd.option_context("display.max_columns", None, "display.width", None):
        print(ExpectationMaximizationSpeedComparison().run())
