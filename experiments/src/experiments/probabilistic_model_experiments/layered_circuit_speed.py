"""
Speed of the layered numpy probabilistic circuit implementation compared to the
rustworkx implementation it is converted from.

The circuit is a joint probability tree fitted to correlated normal samples. The most
demanding case is truncating it to an event with many simple sets, which is the
bottleneck of the application this implementation was built for: truncating once per
simple set and mixing the results spreads a circuit with a handful of layers over
hundreds of them, so every later query pays python overhead per layer instead of
running over arrays. This experiment measures the alternative, a single batched pass
that keeps the number of layers constant, against that baseline.

Conditioning on a partial point is measured too. It is inherently a single pass with
little to batch away, unlike truncating to an event with many simple sets, so the gain
over rustworkx there comes only from the layered layout removing per-node python
overhead, not from any batching.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from random_events.interval import SimpleInterval
from random_events.product_algebra import Event, SimpleEvent
from random_events.variable import Continuous

from experiments.experiment_definitions import (
    ExperimentResult,
    ExperimentsTable,
    TypstRenderer,
)
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit as RustworkxProbabilisticCircuit,
)

NUMBERS_OF_SIMPLE_SETS = (5, 10, 25, 50, 100)
LARGEST_NUMBER_OF_SIMPLE_SETS = NUMBERS_OF_SIMPLE_SETS[-1]


class BenchmarkStage(enum.Enum):
    """
    Whether a query benchmark ran on the original circuit or on the circuit truncated
    to :data:`LARGEST_NUMBER_OF_SIMPLE_SETS` simple sets.
    """

    BEFORE_TRUNCATION = "before truncation"
    AFTER_TRUNCATION = "after truncation"


@dataclass
class CorrelatedNormalTreeFactory:
    """
    Learns a joint probability tree fitted to correlated normal samples, shared by
    every measurement in this experiment.
    """

    number_of_variables: int = 4
    """
    Number of continuous variables of the fitted samples.
    """

    number_of_samples: int = 5000
    """
    Number of samples drawn to fit the tree.
    """

    min_samples_per_leaf: float = 0.02
    """
    Minimum fraction of samples a leaf of the tree must hold.
    """

    min_samples_per_quantile: int = 50
    """
    Minimum number of samples per quantile when inferring the variables' domains.
    """

    def learn_circuit(self) -> JointProbabilityTree:
        """
        :return: A joint probability tree fitted to correlated normal samples.
        """
        np.random.seed(69)
        covariance = np.random.uniform(
            0, 1, (self.number_of_variables, self.number_of_variables)
        )
        covariance = covariance @ covariance.T
        data = np.random.multivariate_normal(
            np.zeros(self.number_of_variables), covariance, self.number_of_samples
        )
        frame = pd.DataFrame(
            data, columns=[f"x_{index}" for index in range(self.number_of_variables)]
        )
        variables = infer_variables_from_dataframe(
            frame, min_samples_per_quantile=self.min_samples_per_quantile
        )
        return JointProbabilityTree(
            annotated_variables=variables, min_samples_per_leaf=self.min_samples_per_leaf
        ).fit(frame)


def staircase_of_boxes(
    first: Continuous,
    second: Continuous,
    first_interval: SimpleInterval,
    second_interval: SimpleInterval,
    number_of_boxes: int,
) -> Event:
    """
    Build an event out of ``number_of_boxes`` disjoint boxes.

    Every box gets its own window of both variables. Slicing only one variable would not
    work: a simple set may assign a composite interval to a variable, so boxes that agree
    on every other dimension are merged back together.

    :param first: The first variable to slice.
    :param second: The second variable to slice.
    :param first_interval: The range of the first variable.
    :param second_interval: The range of the second variable.
    :param number_of_boxes: The number of boxes.
    :return: The event.
    """
    first_edges = np.linspace(
        first_interval.lower, first_interval.upper, number_of_boxes + 1
    )
    second_edges = np.linspace(
        second_interval.lower, second_interval.upper, number_of_boxes + 1
    )

    result = None
    for index in range(number_of_boxes):
        box = SimpleEvent.from_data(
            {
                first: SimpleInterval.from_data(
                    first_edges[index], first_edges[index + 1]
                ).as_composite_set(),
                second: SimpleInterval.from_data(
                    second_edges[index], second_edges[index + 1]
                ).as_composite_set(),
            }
        ).as_composite_set()
        result = box if result is None else result | box
    return result


def fastest_duration(function, repeats: int = 3):
    """
    :param function: The function to time.
    :param repeats: How often to run it.
    :return: The shortest duration in seconds and the result of the last run.
    """
    best = float("inf")
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        best = min(best, time.perf_counter() - start)
    return best, result


@dataclass
class QueryDurationResult(ExperimentResult):
    """
    Wall-clock duration of one query, rustworkx vs the layered numpy circuit.
    """

    stage: BenchmarkStage
    """
    Whether this query ran on the original circuit or the one truncated to
    :data:`LARGEST_NUMBER_OF_SIMPLE_SETS` simple sets.
    """

    query: str
    """
    Name of the query, with its batch size where relevant.
    """

    rustworkx_duration: float
    """
    Fastest of several runs on the rustworkx circuit, in milliseconds.
    """

    layered_duration: float
    """
    Fastest of several runs on the layered circuit, in milliseconds.
    """

    speedup: float
    """
    How many times faster the layered circuit answered than rustworkx.
    """


def measure_query_durations(
    stage: BenchmarkStage,
    rustworkx_circuit: RustworkxProbabilisticCircuit,
    layered: LayeredProbabilisticCircuit,
) -> list[QueryDurationResult]:
    """
    Measure ``log_likelihood``, ``sample`` and ``probability_of_simple_event`` on both
    circuits.

    :param stage: Whether this is measuring the original or the truncated circuit.
    :param rustworkx_circuit: The rustworkx circuit.
    :param layered: The layered circuit it was converted from.
    :return: One result per query.
    """
    results = []

    for amount in (100, 1000, 10000):
        samples = rustworkx_circuit.sample(amount)
        rustworkx_duration, rustworkx_result = fastest_duration(
            lambda: rustworkx_circuit.log_likelihood(samples)
        )
        layered_duration, layered_result = fastest_duration(
            lambda: layered.log_likelihood(samples)
        )
        assert np.allclose(rustworkx_result, layered_result)
        results.append(
            _query_result(
                stage, f"log_likelihood, {amount} events",
                rustworkx_duration, layered_duration,
            )
        )

    for amount in (1000, 10000):
        rustworkx_duration, _ = fastest_duration(
            lambda: rustworkx_circuit.sample(amount), repeats=2
        )
        layered_duration, _ = fastest_duration(
            lambda: layered.sample(amount), repeats=2
        )
        results.append(
            _query_result(
                stage, f"sample, {amount}", rustworkx_duration, layered_duration
            )
        )

    bounding_box = rustworkx_circuit.support.bounding_box()
    rustworkx_duration, _ = fastest_duration(
        lambda: rustworkx_circuit.probability_of_simple_event(bounding_box)
    )
    layered_duration, _ = fastest_duration(
        lambda: layered.probability_of_simple_event(bounding_box)
    )
    results.append(
        _query_result(
            stage, "probability_of_simple_event", rustworkx_duration, layered_duration
        )
    )

    return results


def _query_result(
    stage: BenchmarkStage, query: str, rustworkx_duration: float, layered_duration: float
) -> QueryDurationResult:
    """
    :param stage: Whether this is measuring the original or the truncated circuit.
    :param query: Name of the query.
    :param rustworkx_duration: Fastest run on the rustworkx circuit, in seconds.
    :param layered_duration: Fastest run on the layered circuit, in seconds.
    :return: The result, with durations converted to milliseconds.
    """
    return QueryDurationResult(
        stage=stage,
        query=query,
        rustworkx_duration=round(rustworkx_duration * 1000, 3),
        layered_duration=round(layered_duration * 1000, 3),
        speedup=round(rustworkx_duration / layered_duration, 1),
    )


@dataclass
class TruncationScalingResult(ExperimentResult):
    """
    Cost of truncating a circuit to an event with a growing number of disjoint simple
    sets, and the size of the resulting layered circuit.
    """

    number_of_simple_sets: int
    """
    Number of disjoint simple sets the truncating event is composed of.
    """

    rustworkx_duration: float
    """
    Duration of truncating the rustworkx circuit once per simple set and mixing the
    results, in milliseconds.
    """

    layered_duration: float
    """
    Duration of truncating the layered circuit in a single batched pass, in
    milliseconds.
    """

    speedup: float
    """
    How many times faster the layered circuit's batched pass is than rustworkx.
    """

    result_number_of_nodes: int
    """
    Number of nodes of the layered result.
    """

    result_number_of_layers: int
    """
    Number of layers of the layered result. Stays constant regardless of
    :attr:`number_of_simple_sets`, which is the point of the batched truncation pass:
    truncating once per simple set instead would spread the result over one set of
    layers per simple set.
    """


def measure_truncation_scaling(
    rustworkx_circuit: RustworkxProbabilisticCircuit, layered: LayeredProbabilisticCircuit
) -> tuple[list[TruncationScalingResult], RustworkxProbabilisticCircuit, LayeredProbabilisticCircuit]:
    """
    Measure truncating both circuits to a staircase of disjoint boxes of growing size.

    :param rustworkx_circuit: The rustworkx circuit.
    :param layered: The layered circuit it was converted from.
    :return: One result per number of simple sets, and the two circuits truncated to
        :data:`LARGEST_NUMBER_OF_SIMPLE_SETS` simple sets.
    """
    bounding_box = rustworkx_circuit.support.bounding_box()
    first, second = layered.variables[0], layered.variables[1]

    results = []
    largest_rustworkx_truncated = None
    largest_layered_truncated = None

    for number_of_boxes in NUMBERS_OF_SIMPLE_SETS:
        event = staircase_of_boxes(
            first,
            second,
            bounding_box[first].simple_sets[0],
            bounding_box[second].simple_sets[0],
            number_of_boxes,
        )

        rustworkx_duration, (rustworkx_truncated, rustworkx_probability) = (
            fastest_duration(
                lambda: rustworkx_circuit.truncated(event.__deepcopy__()), repeats=1
            )
        )
        layered_duration, (layered_truncated, layered_probability) = fastest_duration(
            lambda: layered.truncated(event.__deepcopy__()), repeats=1
        )
        assert np.isclose(rustworkx_probability, layered_probability)

        if number_of_boxes == LARGEST_NUMBER_OF_SIMPLE_SETS:
            largest_rustworkx_truncated = rustworkx_truncated
            largest_layered_truncated = layered_truncated

        results.append(
            TruncationScalingResult(
                number_of_simple_sets=len(event.simple_sets),
                rustworkx_duration=round(rustworkx_duration * 1000, 3),
                layered_duration=round(layered_duration * 1000, 3),
                speedup=round(rustworkx_duration / layered_duration, 1),
                result_number_of_nodes=layered_truncated.number_of_nodes,
                result_number_of_layers=len(layered_truncated.layers),
            )
        )

    return results, largest_rustworkx_truncated, largest_layered_truncated


@dataclass
class ConditioningResult(ExperimentResult):
    """
    Wall-clock duration of conditioning on a partial point, rustworkx vs the layered
    numpy circuit.

    rustworkx's ``log_conditional_in_place`` reports the wrong probability on circuits
    with shared subcircuits (this joint probability tree shares leaves across branches),
    so correctness is cross-checked against the layered circuit's own
    ``marginal().log_likelihood()`` instead of trusting rustworkx's return value; only
    the timing is compared between the two.
    """

    number_of_conditioned_variables: int
    """
    Number of variables the point assigns a value to.
    """

    rustworkx_duration: float
    """
    Fastest of several runs on the rustworkx circuit, in milliseconds.
    """

    layered_duration: float
    """
    Fastest of several runs on the layered circuit, in milliseconds.
    """

    speedup: float
    """
    How many times faster the layered circuit answered than rustworkx.
    """


def measure_conditioning(
    rustworkx_circuit: JointProbabilityTree, layered: LayeredProbabilisticCircuit
) -> list[ConditioningResult]:
    """
    Measure conditioning on points that fix a growing number of variables.

    :param rustworkx_circuit: The rustworkx circuit.
    :param layered: The layered circuit it was converted from.
    :return: One result per number of conditioned variables.
    """
    sample = rustworkx_circuit.sample(1)[0]
    variables = layered.variables
    results = []

    for number_of_conditioned in range(1, len(variables) + 1):
        point = {variables[i]: sample[i] for i in range(number_of_conditioned)}
        conditioned_variables = list(point.keys())

        marginal_check = layered.marginal(conditioned_variables)
        row = np.array([[point[v] for v in conditioned_variables]])
        expected_log_probability = float(marginal_check.log_likelihood(row)[0])

        rustworkx_duration, _ = fastest_duration(
            lambda: rustworkx_circuit.__deepcopy__().log_conditional_in_place(point),
            repeats=5,
        )
        layered_duration, (_, layered_log_probability) = fastest_duration(
            lambda: layered.__deepcopy__().log_conditional_in_place(point), repeats=5
        )
        assert np.isclose(expected_log_probability, layered_log_probability)

        results.append(
            ConditioningResult(
                number_of_conditioned_variables=number_of_conditioned,
                rustworkx_duration=round(rustworkx_duration * 1000, 3),
                layered_duration=round(layered_duration * 1000, 3),
                speedup=round(rustworkx_duration / layered_duration, 1),
            )
        )

    return results


def main():
    rustworkx_circuit = CorrelatedNormalTreeFactory().learn_circuit()
    layered = LayeredProbabilisticCircuit.from_rustworkx(rustworkx_circuit)

    before_table = ExperimentsTable(
        measure_query_durations(
            BenchmarkStage.BEFORE_TRUNCATION, rustworkx_circuit, layered
        )
    )
    print(
        TypstRenderer(before_table).render_figure(
            f"Query durations (ms) on a joint probability tree with "
            f"{layered.number_of_nodes} nodes, rustworkx vs the layered numpy circuit."
        )
    )
    print()

    scaling_results, rustworkx_truncated, layered_truncated = measure_truncation_scaling(
        rustworkx_circuit, layered
    )
    print(
        TypstRenderer(ExperimentsTable(scaling_results)).render_figure(
            "Truncating the circuit to a staircase of disjoint boxes with a growing "
            "number of simple sets. The layered result stays at a constant number of "
            "layers regardless of how many simple sets it was truncated to."
        )
    )
    print()

    after_table = ExperimentsTable(
        measure_query_durations(
            BenchmarkStage.AFTER_TRUNCATION, rustworkx_truncated, layered_truncated
        )
    )
    print(
        TypstRenderer(after_table).render_figure(
            f"Query durations (ms) on the circuit truncated to "
            f"{LARGEST_NUMBER_OF_SIMPLE_SETS} simple sets "
            f"({layered_truncated.number_of_nodes} nodes)."
        )
    )
    print()

    conditioning_results = measure_conditioning(rustworkx_circuit, layered)
    print(
        TypstRenderer(ExperimentsTable(conditioning_results)).render_figure(
            f"Conditioning durations (ms) on a joint probability tree with "
            f"{layered.number_of_nodes} nodes, as the number of conditioned "
            f"variables grows, rustworkx vs the layered numpy circuit."
        )
    )


if __name__ == "__main__":
    main()
