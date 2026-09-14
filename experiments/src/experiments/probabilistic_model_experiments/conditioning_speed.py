"""
Speed of conditioning the layered numpy probabilistic circuit on a partial point,
compared to the rustworkx implementation it is converted from.

Conditioning on a point is inherently a single pass with little to batch away, unlike
truncating to an event with many simple sets (see :mod:`layered_circuit_speed`), so the
gain over rustworkx here comes only from the layered layout removing per-node python
overhead, not from any batching.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from experiments.experiment_definitions import (
    ExperimentResult,
    ExperimentsTable,
    TypstRenderer,
)
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.probabilistic_circuit.np.probabilistic_circuit import (
    ProbabilisticCircuit,
)

NUMBER_OF_VARIABLES = 4
NUMBER_OF_SAMPLES = 5000
MIN_SAMPLES_PER_LEAF = 0.02
MIN_SAMPLES_PER_QUANTILE = 50


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


def fastest_duration(function, repeats: int = 5):
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
    Number of variables the point assigns a value to, out of :data:`NUMBER_OF_VARIABLES`.
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
    rustworkx_circuit: JointProbabilityTree, layered: ProbabilisticCircuit
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

    for number_of_conditioned in range(1, NUMBER_OF_VARIABLES + 1):
        point = {variables[i]: sample[i] for i in range(number_of_conditioned)}
        conditioned_variables = list(point.keys())

        marginal_check = layered.marginal(conditioned_variables)
        row = np.array([[point[v] for v in conditioned_variables]])
        expected_log_probability = float(marginal_check.log_likelihood(row)[0])

        rustworkx_duration, _ = fastest_duration(
            lambda: rustworkx_circuit.__deepcopy__().log_conditional_in_place(point)
        )
        layered_duration, (_, layered_log_probability) = fastest_duration(
            lambda: layered.__deepcopy__().log_conditional_in_place(point)
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
    rustworkx_circuit = learn_circuit()
    layered = ProbabilisticCircuit.from_rustworkx(rustworkx_circuit)

    results = measure_conditioning(rustworkx_circuit, layered)
    print(
        TypstRenderer(ExperimentsTable(results)).render_figure(
            f"Conditioning durations (ms) on a joint probability tree with "
            f"{layered.number_of_nodes} nodes, as the number of conditioned "
            f"variables grows, rustworkx vs the layered numpy circuit."
        )
    )


if __name__ == "__main__":
    main()
