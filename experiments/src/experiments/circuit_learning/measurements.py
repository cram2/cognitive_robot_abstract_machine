"""
What the circuit learning benchmark measures: the cost of learning a circuit and how
well it fits, which queries it answers exactly, and how fast it answers likelihood,
sampling, conditioning and truncation queries.
"""

from __future__ import annotations

# %% imports
import time
from dataclasses import dataclass, field
from datetime import timedelta

import jax
import numpy as np
import torch
from array_api_compat import array_namespace, device
from random_events.interval import closed_open
from random_events.product_algebra import Event, SimpleEvent
from random_events.variable import Symbolic
from sortedcontainers import SortedSet
from krrood_compute_hardware.devices import Device
from krrood_compute_hardware.memory import HostMemory
from typing_extensions import Any, Callable

from experiments.circuit_learning.datasets import BenchmarkDataset
from experiments.circuit_learning.methods import (
    BenchmarkMethod,
    TrainingContext,
)
from experiments.circuit_learning.rows import (
    BenchmarkConfiguration,
    ConditioningSpeedResult,
    ExactQueriesResult,
    LikelihoodSpeedResult,
    Query,
    SamplingSpeedResult,
    TrainingResult,
    TruncationSpeedResult,
)
from experiments.circuit_learning.settings import ArrayLibrary, ComputationSetting
from experiments.experiment_definitions import (
    ExperimentResult,
    MeanAndStandardDeviation,
    Unit,
)
from probabilistic_model.probabilistic_circuit.tensorized.has_parameter_arrays import (
    HasParameterArrays,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import (
    InnerLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.sum_layer import (
    SumLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)

# %% evaluation


@dataclass(kw_only=True)
class ChunkedLogLikelihoodEvaluation:
    """
    Computes the log-likelihood of many rows in chunks, in the setting of the circuit.
    """

    circuit: LayeredProbabilisticCircuit
    """
    The circuit, of any backend.
    """

    chunk_rows: int = 4096
    """
    The number of rows evaluated at once.
    """

    def compute_log_likelihoods(self, *, rows: np.ndarray) -> np.ndarray:
        """
        :param rows: Rows whose symbolic values are hashes of domain elements.
        :return: The log-likelihood of every row, on the host.
        """
        return np.concatenate(
            [
                self.compute_chunk_log_likelihoods(
                    chunk=rows[start : start + self.chunk_rows]
                )
                for start in range(0, len(rows), self.chunk_rows)
            ]
        )

    def compute_chunk_log_likelihoods(self, *, chunk: np.ndarray) -> np.ndarray:
        """
        :param chunk: At most :attr:`chunk_rows` rows.
        :return: Their log-likelihoods, on the host.
        """
        parameter = self.circuit.parameters[0]
        data = array_namespace(parameter).asarray(
            self.circuit.encoded(chunk),
            dtype=parameter.dtype,
            device=device(parameter),
        )
        return np.asarray(
            HasParameterArrays.host_copy(
                self.circuit.root.log_likelihood_of_nodes(data)[:, 0]
            )
        )


# %% training


@dataclass(kw_only=True)
class MeasuredTraining:
    """
    The measurements of a training and the circuit it learned.
    """

    result: TrainingResult
    """
    The measurements.
    """

    circuit: LayeredProbabilisticCircuit
    """
    The learned circuit, its parameters stored as the setting it was learned in says.
    """


@dataclass(kw_only=True)
class TrainingMeasurement:
    """
    Learns a circuit with a method on a dataset in a setting and measures it.
    """

    dataset: BenchmarkDataset
    """
    The dataset.
    """

    method: BenchmarkMethod
    """
    The method.
    """

    setting: ComputationSetting
    """
    The setting to learn in.
    """

    context: TrainingContext
    """
    The limits of training.
    """

    def run(self) -> MeasuredTraining:
        """
        :return: The measurements and the learned circuit.
        """
        host_memory = HostMemory()
        host_memory.reset_peak()
        start = time.perf_counter()
        learned = self.method.learn(
            dataset=self.dataset, setting=self.setting, context=self.context
        )
        training_duration = timedelta(seconds=time.perf_counter() - start)
        peak_memory_bytes = (
            self.peak_device_memory_bytes()
            if self.setting.uses_graphics_processor
            else host_memory.peak_bytes()
        )
        circuit = learned.circuit
        log_likelihoods = ChunkedLogLikelihoodEvaluation(circuit=circuit)
        test = log_likelihoods.compute_log_likelihoods(
            rows=self.dataset.encoded(frame=self.dataset.test)
        )
        train = log_likelihoods.compute_log_likelihoods(
            rows=self.dataset.encoded(frame=self.dataset.train)
        )
        result = TrainingResult(
            configuration=BenchmarkConfiguration.from_setting(
                learning_method=self.method.name,
                dataset=self.dataset.name,
                setting=self.setting,
                seed=self.method.seed,
            ),
            training_duration=training_duration,
            epochs_until_convergence=(
                None if learned.history is None else learned.history.epochs
            ),
            converged=None if learned.history is None else learned.history.converged,
            peak_memory_bytes=peak_memory_bytes,
            number_of_parameters=int(circuit.number_of_parameters),
            number_of_edges=int(
                sum(
                    layer.edges.number_of_edges
                    for layer in circuit.root.all_layers()
                    if isinstance(layer, InnerLayer)
                )
            ),
            average_train_log_likelihood=float(train.mean()),
            average_test_log_likelihood=float(test.mean()),
            fraction_of_test_rows_without_support=float(np.isneginf(test).mean()),
            average_test_log_likelihood_of_supported_rows=self.supported_mean(
                log_likelihoods=test
            ),
        )
        return MeasuredTraining(result=result, circuit=circuit)

    @staticmethod
    def supported_mean(*, log_likelihoods: np.ndarray) -> float:
        """
        :param log_likelihoods: The log-likelihood of every row.
        :return: The mean over the rows of positive likelihood, minus infinity if there
            are none.
        """
        supported = log_likelihoods[~np.isneginf(log_likelihoods)]
        return float(supported.mean()) if len(supported) else -np.inf

    def peak_device_memory_bytes(self) -> int:
        """
        :return: The most GPU memory jax and torch held at once since the process
            started, summed.
        """
        statistics = jax.devices(Device.GRAPHICS_PROCESSOR)[0].memory_stats() or {}
        return int(
            statistics.get("peak_bytes_in_use", 0) + torch.cuda.max_memory_allocated()
        )


# %% queries


@dataclass(kw_only=True)
class QueryTiming:
    """
    How long a query took the first time and on average after that.
    """

    first_call: timedelta
    """
    The time of the first call, which includes any compilation.
    """

    later_call: timedelta
    """
    The mean time of a call after the first.
    """


@dataclass(kw_only=True)
class QueryMeasurement:
    """
    Measures the queries of a stored circuit in one setting: the likelihood in every
    setting, and sampling, conditioning, truncation and exactness on the host.
    """

    dataset: BenchmarkDataset
    """
    The dataset the circuit was learned from.
    """

    configuration: BenchmarkConfiguration
    """
    The method and dataset of the circuit, in the setting the queries run in.
    """

    setting: ComputationSetting
    """
    The setting the queries run in.
    """

    circuit: LayeredProbabilisticCircuit
    """
    The circuit, with host parameters.
    """

    likelihood_rows: int = 10000
    """
    The most test rows a likelihood call evaluates.
    """

    sampled_rows: int = 1000
    """
    The number of samples a sampling call draws.
    """

    minimum_timing: timedelta = timedelta(seconds=1)
    """
    How long a query is repeated after its first call to measure its speed.
    """

    query_sizes: tuple[int, ...] = (5, 10, 25, 50, 100)
    """
    The numbers of conditioned variables and of simple sets of truncation events.
    """

    repetitions: int = 10
    """
    The number of conditionings and truncations per size.
    """

    seed: int = 69
    """
    The seed of the observations and events.
    """

    random_number_generator: np.random.Generator = field(init=False, repr=False)
    """
    The generator of the observations and events.
    """

    def __post_init__(self) -> None:
        """
        Seed :attr:`random_number_generator`.
        """
        self.random_number_generator = np.random.default_rng(self.seed)

    def run(self) -> list[ExperimentResult]:
        """
        :return: The rows of every query measured in :attr:`setting`.
        """
        rows: list[ExperimentResult] = [self.likelihood_speed()]
        if self.setting.array_library is not ArrayLibrary.NUMPY:
            return rows
        circuit = self.setting.placed(circuit=self.circuit)
        rows.append(self.sampling_speed(circuit=circuit))
        rows.extend(self.conditioning_speeds(circuit=circuit))
        rows.extend(self.truncation_speeds(circuit=circuit))
        if self.setting == ComputationSetting.host():
            rows.append(self.exact_queries())
        return rows

    def timed_calls(self, *, query: Callable[[], Any]) -> QueryTiming:
        """
        :param query: A function without arguments that answers a query.
        :return: The time of its first call and the mean time of the calls after it,
            repeated for at least :attr:`minimum_timing`.
        """
        start = time.perf_counter()
        query()
        first_call = timedelta(seconds=time.perf_counter() - start)
        minimum_seconds = self.minimum_timing.total_seconds()
        calls = 0
        start = time.perf_counter()
        while calls < 3 or time.perf_counter() - start < minimum_seconds:
            query()
            calls += 1
        return QueryTiming(
            first_call=first_call,
            later_call=timedelta(seconds=(time.perf_counter() - start) / calls),
        )

    def likelihood_speed(self) -> LikelihoodSpeedResult:
        """
        :return: How fast the circuit computes the likelihood of test rows.
        """
        circuit = self.setting.placed(circuit=self.circuit)
        chunk_rows = min(self.likelihood_rows, len(self.dataset.test))
        rows = self.dataset.encoded(frame=self.dataset.test)[:chunk_rows]
        log_likelihoods = ChunkedLogLikelihoodEvaluation(
            circuit=circuit, chunk_rows=chunk_rows
        )
        timing = self.timed_calls(
            query=lambda: log_likelihoods.compute_chunk_log_likelihoods(chunk=rows)
        )
        return LikelihoodSpeedResult(
            configuration=self.configuration,
            first_call_duration=timing.first_call,
            rows_per_second=chunk_rows / timing.later_call.total_seconds(),
        )

    def sampling_speed(
        self, *, circuit: LayeredProbabilisticCircuit
    ) -> SamplingSpeedResult:
        """
        :param circuit: The circuit in :attr:`setting`.
        :return: How fast the circuit draws samples.
        """
        timing = self.timed_calls(query=lambda: circuit.sample(self.sampled_rows))
        return SamplingSpeedResult(
            configuration=self.configuration,
            first_call_duration=timing.first_call,
            samples_per_second=self.sampled_rows / timing.later_call.total_seconds(),
        )

    def conditioning_speeds(
        self, *, circuit: LayeredProbabilisticCircuit
    ) -> list[ConditioningSpeedResult]:
        """
        :param circuit: The circuit in :attr:`setting`.
        :return: How long conditioning on the values of a test row takes, for every
            number of observed variables up to the number of variables.
        """
        variables = list(circuit.variables)
        test = self.dataset.test
        results = []
        for size in self.query_sizes:
            if size > len(variables):
                break
            milliseconds = []
            for _ in range(self.repetitions):
                row = test.iloc[int(self.random_number_generator.integers(len(test)))]
                observed = self.random_number_generator.choice(
                    len(variables), size=size, replace=False
                )
                point = {
                    variables[index]: row[variables[index].name] for index in observed
                }
                start = time.perf_counter()
                circuit.log_conditional(point)
                milliseconds.append((time.perf_counter() - start) * 1000)
            results.append(
                ConditioningSpeedResult(
                    configuration=self.configuration,
                    number_of_conditioned_variables=size,
                    conditioning_time=MeanAndStandardDeviation.from_measurements(
                        milliseconds, Unit.MILLISECONDS
                    ),
                )
            )
        return results

    def truncation_speeds(
        self, *, circuit: LayeredProbabilisticCircuit
    ) -> list[TruncationSpeedResult]:
        """
        :param circuit: The circuit in :attr:`setting`.
        :return: How long truncating to an event of disjoint simple sets takes, for
            every number of simple sets.
        """
        results = []
        for size in self.query_sizes:
            milliseconds = []
            for _ in range(self.repetitions):
                event = self.cells_of_test_rows(circuit=circuit, number_of_cells=size)
                start = time.perf_counter()
                circuit.log_truncated(event)
                milliseconds.append((time.perf_counter() - start) * 1000)
            results.append(
                TruncationSpeedResult(
                    configuration=self.configuration,
                    number_of_simple_sets=size,
                    truncation_time=MeanAndStandardDeviation.from_measurements(
                        milliseconds, Unit.MILLISECONDS
                    ),
                )
            )
        return results

    def cells_of_test_rows(
        self,
        *,
        circuit: LayeredProbabilisticCircuit,
        number_of_cells: int,
        bins: int = 4,
    ) -> Event:
        """
        :param circuit: The circuit.
        :param number_of_cells: The number of simple sets of the event.
        :param bins: The number of quantile bins of a continuous variable.
        :return: The union of the cells of distinct test rows in a grid over randomly
            chosen variables, a symbolic variable split by its values and a continuous
            one by the quantiles of the training rows, with as few variables as give
            enough distinct cells.
        """
        test = self.dataset.test
        cells: dict[tuple, SimpleEvent] = {}
        chosen = []
        for index in self.random_number_generator.permutation(len(circuit.variables)):
            chosen.append(circuit.variables[int(index)])
            cells = self.distinct_cells(variables=chosen, bins=bins)
            if len(cells) >= number_of_cells:
                break
        keys = list(cells)
        picked = self.random_number_generator.choice(
            len(keys), size=min(number_of_cells, len(keys)), replace=False
        )
        return Event.from_simple_sets(*[cells[keys[int(index)]] for index in picked])

    def distinct_cells(self, *, variables: list, bins: int) -> dict[tuple, SimpleEvent]:
        """
        :param variables: The variables of the grid.
        :param bins: The number of quantile bins of a continuous variable.
        :return: The cell of every distinct test row in the grid, by its key.
        """
        assignments = []
        keys = []
        for variable in variables:
            values = self.dataset.test[variable.name].to_numpy()
            if isinstance(variable, Symbolic):
                keys.append(values)
                assignments.append({value: value for value in set(values)})
                continue
            edges = np.unique(
                np.quantile(
                    self.dataset.train[variable.name].to_numpy(),
                    np.linspace(0, 1, bins + 1)[1:-1],
                )
            )
            borders = np.concatenate([[-np.inf], edges, [np.inf]])
            positions = np.searchsorted(edges, values, side="right")
            keys.append(positions)
            assignments.append(
                {
                    position: closed_open(borders[position], borders[position + 1])
                    for position in set(positions)
                }
            )
        cells = {}
        for row in zip(*keys):
            if row in cells:
                continue
            cells[row] = SimpleEvent.from_data(
                {
                    variable: assignment[key]
                    for variable, assignment, key in zip(variables, assignments, row)
                }
            )
        return cells

    def exact_queries(self) -> ExactQueriesResult:
        """
        :return: The queries the circuit answers exactly.
        """
        circuit = self.circuit
        queries = []
        if circuit.root.is_decomposable():
            queries.extend([Query.MARGINAL, Query.CONDITIONAL, Query.TRUNCATION])
            if self.is_support_deterministic_over_first_variable(circuit=circuit):
                queries.append(Query.CAUSAL)
        if circuit.is_deterministic():
            queries.append(Query.MOST_LIKELY_ASSIGNMENT)
        return ExactQueriesResult(
            configuration=self.configuration, exact_queries=queries
        )

    @staticmethod
    def is_support_deterministic_over_first_variable(
        *, circuit: LayeredProbabilisticCircuit
    ) -> bool:
        """
        :param circuit: The circuit with host parameters.
        :return: Whether the children of every sum unit whose scope holds the first
            variable have disjoint supports over it, which a backdoor adjustment with
            that variable as the cause needs.
        """
        cause = SortedSet([circuit.variables[0]])
        for layer in circuit.layers:
            if not isinstance(layer, SumLayer) or 0 not in layer.variables:
                continue
            for node in layer.component_supports(circuit.variables):
                supports = [support.marginal(cause) for support in node.supports]
                if len(supports) > 1 and not SumLayer.are_pairwise_disjoint(supports):
                    return False
        return True
