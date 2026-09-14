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
from probabilistic_model.probabilistic_circuit.tensorized.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit as RustworkxProbabilisticCircuit,
)

NUMBER_OF_VARIABLES = 4
NUMBER_OF_SAMPLES = 5000
MIN_SAMPLES_PER_LEAF = 0.02
MIN_SAMPLES_PER_QUANTILE = 50
NUMBERS_OF_SIMPLE_SETS = (5, 10, 25, 50, 100)
LARGEST_NUMBER_OF_SIMPLE_SETS = NUMBERS_OF_SIMPLE_SETS[-1]


class BenchmarkStage(enum.Enum):
    """
    Whether a query benchmark ran on the original circuit or on the circuit truncated
    to :data:`LARGEST_NUMBER_OF_SIMPLE_SETS` simple sets.
    """

    BEFORE_TRUNCATION = "before truncation"
    AFTER_TRUNCATION = "after truncation"


def learn_circuit() -> JointProbabilityTree:
    """
    :return: A joint probability tree fitted to correlated normal samples.
    """
    np.random.seed(69)
    covariance = np.random.uniform(0, 1, (NUMBER_OF_VARIABLES, NUMBER_OF_VARIABLES))
    covariance = covariance @ covariance.T
    data = np.random.multivariate_normal(
        np.zeros(NUMBER_OF_VARIABLES), covariance, NUMBER_OF_SAMPLES
    )
    frame = pd.DataFrame(
        data, columns=[f"x_{index}" for index in range(NUMBER_OF_VARIABLES)]
    )
    variables = infer_variables_from_dataframe(
        frame, min_samples_per_quantile=MIN_SAMPLES_PER_QUANTILE
    )
    return JointProbabilityTree(
        annotated_variables=variables, min_samples_per_leaf=MIN_SAMPLES_PER_LEAF
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
    layered: ProbabilisticCircuit,
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
    rustworkx_circuit: RustworkxProbabilisticCircuit, layered: ProbabilisticCircuit
) -> tuple[list[TruncationScalingResult], RustworkxProbabilisticCircuit, ProbabilisticCircuit]:
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


def main():
    rustworkx_circuit = learn_circuit()
    layered = ProbabilisticCircuit.from_rustworkx(rustworkx_circuit)

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


if __name__ == "__main__":
    main()
