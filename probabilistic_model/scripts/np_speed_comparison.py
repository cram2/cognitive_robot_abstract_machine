"""
Compare the layered numpy circuits against the rustworkx circuits they are converted from.

Reproduces the numbers in ``doc/layered_circuit.md``. The interesting case is the last
one: truncating to an event with many simple sets, which is the most demanding query of
the package.
"""

import time

import numpy as np
import pandas as pd
from random_events.interval import SimpleInterval, closed
from random_events.product_algebra import Event, SimpleEvent

from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.probabilistic_circuit.np.probabilistic_circuit import (
    ProbabilisticCircuit,
)

np.random.seed(69)

NUMBER_OF_VARIABLES = 4
NUMBER_OF_SAMPLES = 5000
MIN_SAMPLES_PER_LEAF = 0.02
MIN_SAMPLES_PER_QUANTILE = 50
NUMBERS_OF_SIMPLE_SETS = (5, 10, 25, 50, 100)


def learn_circuit() -> JointProbabilityTree:
    """
    :return: A joint probability tree fitted to correlated normal samples.
    """
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
    first, second, first_interval, second_interval, number_of_boxes: int
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


def fastest(function, repeats: int = 3):
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


def report(label: str, rustworkx_time: float, layered_time: float):
    print(
        f"{label:>32}: rustworkx {rustworkx_time * 1000:>9.1f}ms  "
        f"layered {layered_time * 1000:>9.1f}ms  "
        f"{rustworkx_time / layered_time:>7.1f}x"
    )


def main():
    rustworkx_circuit = learn_circuit()
    layered = ProbabilisticCircuit.from_rustworkx(rustworkx_circuit)

    print(
        f"circuit: {len(rustworkx_circuit.nodes())} nodes, "
        f"{len(layered.layers)} layers"
    )

    print("\nqueries")
    for amount in (100, 1000, 10000):
        samples = rustworkx_circuit.sample(amount)
        rustworkx_time, rustworkx_result = fastest(
            lambda: rustworkx_circuit.log_likelihood(samples)
        )
        layered_time, layered_result = fastest(
            lambda: layered.log_likelihood(samples)
        )
        assert np.allclose(rustworkx_result, layered_result)
        report(f"log_likelihood, {amount} events", rustworkx_time, layered_time)

    for amount in (1000, 10000):
        rustworkx_time, _ = fastest(
            lambda: rustworkx_circuit.sample(amount), repeats=2
        )
        layered_time, _ = fastest(lambda: layered.sample(amount), repeats=2)
        report(f"sample, {amount}", rustworkx_time, layered_time)

    bounding_box = rustworkx_circuit.support.bounding_box()
    rustworkx_time, _ = fastest(
        lambda: rustworkx_circuit.probability_of_simple_event(bounding_box)
    )
    layered_time, _ = fastest(
        lambda: layered.probability_of_simple_event(bounding_box)
    )
    report("probability_of_simple_event", rustworkx_time, layered_time)

    print("\ntruncation to an event with many simple sets")
    first, second = layered.variables[0], layered.variables[1]
    truncated_circuits = {}

    for number_of_boxes in NUMBERS_OF_SIMPLE_SETS:
        event = staircase_of_boxes(
            first,
            second,
            bounding_box[first].simple_sets[0],
            bounding_box[second].simple_sets[0],
            number_of_boxes,
        )

        rustworkx_time, (rustworkx_truncated, rustworkx_probability) = fastest(
            lambda: rustworkx_circuit.truncated(event.__deepcopy__()), repeats=1
        )
        layered_time, (layered_truncated, layered_probability) = fastest(
            lambda: layered.truncated(event.__deepcopy__()), repeats=1
        )
        assert np.isclose(rustworkx_probability, layered_probability)
        truncated_circuits[number_of_boxes] = (
            rustworkx_truncated,
            layered_truncated,
        )

        report(
            f"{len(event.simple_sets)} simple sets", rustworkx_time, layered_time
        )
        print(
            f"{'':>32}  the result has {layered_truncated.number_of_nodes} nodes in "
            f"{len(layered_truncated.layers)} layers"
        )

    print("\nqueries on the circuit truncated to the largest event")
    rustworkx_truncated, layered_truncated = truncated_circuits[
        NUMBERS_OF_SIMPLE_SETS[-1]
    ]
    samples = layered_truncated.sample(1000)

    rustworkx_time, rustworkx_result = fastest(
        lambda: rustworkx_truncated.log_likelihood(samples), repeats=2
    )
    layered_time, layered_result = fastest(
        lambda: layered_truncated.log_likelihood(samples), repeats=2
    )
    assert np.allclose(rustworkx_result, layered_result)
    report("log_likelihood, 1000 events", rustworkx_time, layered_time)

    rustworkx_time, _ = fastest(
        lambda: rustworkx_truncated.sample(1000), repeats=2
    )
    layered_time, _ = fastest(lambda: layered_truncated.sample(1000), repeats=2)
    report("sample, 1000", rustworkx_time, layered_time)

    rustworkx_time, _ = fastest(
        lambda: rustworkx_truncated.probability_of_simple_event(bounding_box),
        repeats=2,
    )
    layered_time, _ = fastest(
        lambda: layered_truncated.probability_of_simple_event(bounding_box),
        repeats=2,
    )
    report("probability_of_simple_event", rustworkx_time, layered_time)


if __name__ == "__main__":
    main()
