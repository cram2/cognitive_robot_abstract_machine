"""
Gaussian mixtures and joint probability trees, fitted and scored on the same rows.

The continuous columns are standardized and cast to single precision, the precision
:class:`~random_events.interval.SimpleInterval` stores its bounds at, so that no tree
leaves the extreme training rows out of its support.
"""

from __future__ import annotations

import enum
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tqdm
from sklearn import datasets
from sklearn.utils import Bunch
from typing_extensions import List, Optional, Sequence

from experiments.experiment_definitions import (
    ExperimentResult,
    ExperimentsTable,
    TypstRenderer,
)
from probabilistic_model.learning.gaussian_mixture.gaussian_mixture_model import (
    GaussianMixtureModel,
)
from probabilistic_model.learning.gaussian_mixture.step_mix_model import StepMixModel
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.learning_method import LearningMethod
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)


class Dataset(enum.StrEnum):
    """
    The scikit-learn datasets the methods are compared on.
    """

    IRIS = "iris"
    """
    Four measurements of 150 flowers and their species.
    """

    WINE = "wine"
    """
    Thirteen measurements of 178 wines and their cultivar.
    """

    BREAST_CANCER = "breast_cancer"
    """
    Thirty measurements of 569 tumours and whether they are malignant.
    """

    DIABETES = "diabetes"
    """
    Nine measurements of 442 patients and their sex.
    """

    CALIFORNIA_HOUSING = "california_housing"
    """
    Eight measurements of 20640 districts. Downloaded on first use.
    """

    @property
    def symbolic_column(self) -> Optional[str]:
        """
        :return: The name of the symbolic column, if there is one.
        """
        match self:
            case Dataset.CALIFORNIA_HOUSING:
                return None
            case Dataset.DIABETES:
                return "sex"
        return "label"

    def load(self) -> pd.DataFrame:
        """
        :return: The dataset, with its continuous columns standardized.
        """
        bunch = self.bunch()
        features = bunch.data.astype(float)
        match self:
            case Dataset.CALIFORNIA_HOUSING:
                return standardized(features)
            case Dataset.DIABETES:
                sex = np.where(features.pop("sex") > 0, "first", "second")
                return standardized(features).assign(sex=sex)
        labels = np.asarray(bunch.target_names)[bunch.target.to_numpy()]
        return standardized(features).assign(label=labels.astype(str))

    def bunch(self) -> Bunch:
        """
        :return: The dataset as scikit-learn provides it, with pandas frames.
        """
        match self:
            case Dataset.IRIS:
                return datasets.load_iris(as_frame=True)
            case Dataset.WINE:
                return datasets.load_wine(as_frame=True)
            case Dataset.BREAST_CANCER:
                return datasets.load_breast_cancer(as_frame=True)
            case Dataset.DIABETES:
                return datasets.load_diabetes(as_frame=True)
            case Dataset.CALIFORNIA_HOUSING:
                return datasets.fetch_california_housing(as_frame=True)

    def settings(self) -> List[Setting]:
        """
        :return: The configurations the dataset is fitted with.
        """
        mixture = (
            Method.GAUSSIAN_MIXTURE if self.symbolic_column is None else Method.STEPMIX
        )
        return [MixtureSetting(mixture, components) for components in (1, 3, 5, 10)] + [
            TreeSetting(share) for share in (0.2, 0.1, 0.05, 0.02)
        ]


def standardized(features: pd.DataFrame) -> pd.DataFrame:
    """
    :return: The columns with zero mean and unit standard deviation, at single
        precision.
    """
    return ((features - features.mean()) / features.std()).astype(np.float32)


class Method(enum.Enum):
    """
    The learning methods compared.
    """

    GAUSSIAN_MIXTURE = GaussianMixtureModel
    """
    A scikit-learn Gaussian mixture, for continuous data.
    """

    STEPMIX = StepMixModel
    """
    A StepMix mixture, for data with a symbolic column.
    """

    JOINT_PROBABILITY_TREE = JointProbabilityTree
    """
    A joint probability tree.
    """


@dataclass
class Setting(ABC):
    """
    One configuration of a method.
    """

    @property
    @abstractmethod
    def method(self) -> Method:
        """
        :return: The method configured.
        """

    @abstractmethod
    def learning_method(self) -> LearningMethod:
        """
        :return: The configured method, not yet fitted.
        """


@dataclass
class MixtureSetting(Setting):
    """
    A Gaussian mixture with a number of components.
    """

    mixture: Method
    """
    The mixture, :attr:`Method.GAUSSIAN_MIXTURE` or :attr:`Method.STEPMIX`.
    """

    number_of_components: int
    """
    The number of components.
    """

    @property
    def method(self) -> Method:
        return self.mixture

    def learning_method(self) -> LearningMethod:
        method = self.mixture.value()
        method.model.set_params(n_components=self.number_of_components, random_state=0)
        return method

    def __str__(self) -> str:
        return f"{self.number_of_components} components"


@dataclass
class TreeSetting(Setting):
    """
    A joint probability tree with a minimum share of the rows in each leaf.
    """

    minimum_leaf_share: float
    """
    The minimum share of the rows in a leaf.
    """

    @property
    def method(self) -> Method:
        return Method.JOINT_PROBABILITY_TREE

    def learning_method(self) -> LearningMethod:
        return self.method.value(min_samples_per_leaf=self.minimum_leaf_share)

    def __str__(self) -> str:
        return f"leaves of at least {self.minimum_leaf_share:.0%} of the rows"


@dataclass
class GaussianMixtureBenchmarkResult(ExperimentResult):
    """
    One configuration of a method, fitted on one dataset.
    """

    dataset: Dataset
    """
    The dataset fitted and scored on.
    """

    method: Method
    """
    The method fitted.
    """

    setting: str
    """
    The configuration of the method.
    """

    average_log_likelihood: float
    """
    The mean log-likelihood of a row.
    """

    impossible_rows: int
    """
    The number of rows with zero density.
    """

    nodes: int
    """
    The number of units of the circuit.
    """

    duration: float
    """
    The time spent fitting, in seconds.
    """


def measure(
    dataset: Dataset, data: pd.DataFrame, setting: Setting
) -> GaussianMixtureBenchmarkResult:
    """
    :param dataset: The dataset the rows are from.
    :param data: The rows, as :meth:`Dataset.load` returns them.
    :param setting: The configuration to fit.
    :return: The configuration fitted and scored on the rows.
    """
    start = time.perf_counter()
    circuit = setting.learning_method().fit(data)
    duration = time.perf_counter() - start

    joint = log_likelihood(circuit, data)

    return GaussianMixtureBenchmarkResult(
        dataset=dataset,
        method=setting.method,
        setting=str(setting),
        average_log_likelihood=float(np.mean(joint)),
        impossible_rows=int(np.sum(~np.isfinite(joint))),
        nodes=len(circuit.nodes()),
        duration=duration,
    )


def log_likelihood(circuit: ProbabilisticCircuit, data: pd.DataFrame) -> np.ndarray:
    """
    :return: The log-likelihood of every row under the circuit.
    """
    columns = [variable.name for variable in circuit.variables]
    return circuit.log_likelihood(data[columns].to_numpy())


def benchmark(
    dataset: Dataset, settings: Optional[Sequence[Setting]] = None
) -> ExperimentsTable:
    """
    :param dataset: The dataset to fit on.
    :param settings: The configurations to fit, by default :meth:`Dataset.settings`.
    :return: One row per configuration.
    """
    data = dataset.load()
    return ExperimentsTable(
        [
            measure(dataset, data, setting)
            for setting in tqdm.tqdm(settings or dataset.settings(), desc=str(dataset))
        ]
    )


def main(datasets_to_run: Sequence[Dataset] = tuple(Dataset)):
    for dataset in datasets_to_run:
        print(
            TypstRenderer(benchmark(dataset)).render_figure(
                f"Gaussian mixtures and joint probability trees fitted and scored on "
                f"the whole {dataset} dataset. Higher log-likelihoods are better. Node "
                f"counts are not comparable across methods."
            )
        )
        print()


if __name__ == "__main__":
    main()
