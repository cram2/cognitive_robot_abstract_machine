"""
Rows of the circuit learning benchmark: one row type per table, each holding the
configuration it was measured in.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum

from experiments.circuit_learning.datasets import DatasetName
from experiments.circuit_learning.methods import LearningMethod
from experiments.circuit_learning.settings import (
    ArrayLibrary,
    ComputationSetting,
    Device,
    EdgeRepresentation,
    FloatingPointType,
)
from experiments.experiment_definitions import (
    ExperimentResult,
    MeanAndStandardDeviation,
)

# %% configuration values


class Query(StrEnum):
    """
    A query a circuit may answer exactly.
    """

    MARGINAL = "marginal"
    CONDITIONAL = "conditional"
    TRUNCATION = "truncation"
    MOST_LIKELY_ASSIGNMENT = "most likely assignment"
    CAUSAL = "causal"


# %% configuration


@dataclass(kw_only=True)
class BenchmarkConfiguration(ExperimentResult):
    """
    The method, dataset and computation setting one measurement was taken in.
    """

    learning_method: LearningMethod
    """
    The method that learned the circuit.
    """

    dataset: DatasetName
    """
    The dataset the circuit was learned from and evaluated on.
    """

    array_library: ArrayLibrary
    """
    The array library the measured computation ran with.
    """

    device: Device
    """
    The device the measured computation ran on.
    """

    floating_point_type: FloatingPointType
    """
    The floating point type of the measured computation.
    """

    edge_representation: EdgeRepresentation
    """
    How the circuit stored its edges during the measured computation.
    """

    seed: int
    """
    The seed of the structure, of the initial parameters and of the row order.
    """

    @classmethod
    def from_setting(
        cls,
        *,
        learning_method: LearningMethod,
        dataset: DatasetName,
        setting: ComputationSetting,
        seed: int,
    ) -> BenchmarkConfiguration:
        """
        :return: The configuration of a method and dataset in a setting.
        """
        return cls(
            learning_method=learning_method,
            dataset=dataset,
            array_library=setting.array_library,
            device=setting.device,
            floating_point_type=setting.floating_point_type,
            edge_representation=setting.edge_representation,
            seed=seed,
        )


# %% rows


@dataclass(kw_only=True)
class TrainingResult(ExperimentResult):
    """
    What learning a circuit cost, how large the circuit is and how well it fits.
    """

    configuration: BenchmarkConfiguration
    """
    The setting the circuit was learned in.
    """

    training_duration: timedelta
    """
    Wall-clock time of learning, structure and compilation included.
    """

    epochs_until_convergence: int | None
    """
    Number of passes over the training rows, ``None`` for methods without iterations.
    """

    average_epoch_duration: timedelta | None
    """
    Mean wall-clock time of an epoch after the first, which also compiles the training
    step, ``None`` for methods without iterations.
    """

    converged: bool | None
    """
    Whether training stopped by convergence rather than by the epoch or time limit,
    ``None`` for methods without iterations.
    """

    peak_memory_bytes: int
    """
    Largest memory use during learning, on the device it trained on.
    """

    number_of_parameters: int
    """
    Number of parameters of the learned circuit.
    """

    number_of_edges: int
    """
    Number of edges of the learned circuit.
    """

    average_train_log_likelihood: float
    """
    Mean log-likelihood per training row.
    """

    average_test_log_likelihood: float
    """
    Mean log-likelihood per test row.
    """

    fraction_of_test_rows_without_support: float
    """
    Fraction of the test rows the circuit gives likelihood zero, which make the mean
    test log-likelihood minus infinity.
    """

    average_test_log_likelihood_of_supported_rows: float
    """
    Mean log-likelihood of the test rows of positive likelihood, minus infinity if there
    are none.
    """


@dataclass(kw_only=True)
class ExactQueriesResult(ExperimentResult):
    """
    The queries a learned circuit answers exactly.
    """

    configuration: BenchmarkConfiguration
    """
    The setting the circuit was learned in.
    """

    exact_queries: list[Query]
    """
    The queries the learned circuit answers exactly.
    """


@dataclass(kw_only=True)
class LikelihoodSpeedResult(ExperimentResult):
    """
    How fast a learned circuit computes the likelihood of test rows.
    """

    configuration: BenchmarkConfiguration
    """
    The setting the likelihood was computed in.
    """

    first_call_duration: timedelta
    """
    Time of the first likelihood call, including any compilation.
    """

    rows_per_second: float
    """
    Test rows evaluated per second after the first call.
    """


@dataclass(kw_only=True)
class SamplingSpeedResult(ExperimentResult):
    """
    How fast a learned circuit draws samples.
    """

    configuration: BenchmarkConfiguration
    """
    The setting the samples were drawn in.
    """

    first_call_duration: timedelta
    """
    Time of the first sampling call.
    """

    samples_per_second: float
    """
    Samples drawn per second after the first call.
    """


@dataclass(kw_only=True)
class ConditioningSpeedResult(ExperimentResult):
    """
    How long a learned circuit takes to condition on observed values of some variables.
    """

    configuration: BenchmarkConfiguration
    """
    The setting the conditioning ran in.
    """

    number_of_conditioned_variables: int
    """
    Number of variables whose values are observed.
    """

    conditioning_time: MeanAndStandardDeviation
    """
    Time to build the conditional circuit, over the test rows the observations come
    from.
    """


@dataclass(kw_only=True)
class TruncationSpeedResult(ExperimentResult):
    """
    How long a learned circuit takes to truncate to an event.
    """

    configuration: BenchmarkConfiguration
    """
    The setting the truncation ran in.
    """

    number_of_simple_sets: int
    """
    Number of disjoint simple sets the event is made of.
    """

    truncation_time: MeanAndStandardDeviation
    """
    Time to build the truncated circuit, over the events of this size.
    """
