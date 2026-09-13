"""
Benchmark conditioning on a partial point, rustworkx vs the numpy layered circuit.

Mirrors scripts/np_speed_comparison.py's setup (same JPT, same seed) but measures
``conditional`` instead of truncation.
"""

import time

import numpy as np
import pandas as pd

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


def learn_circuit() -> JointProbabilityTree:
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


def fastest(function, repeats: int = 5):
    best = float("inf")
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        best = min(best, time.perf_counter() - start)
    return best, result


def report(label: str, rustworkx_time: float, layered_time: float):
    print(
        f"{label:>40}: rustworkx {rustworkx_time * 1000:>9.3f}ms  "
        f"layered {layered_time * 1000:>9.3f}ms  "
        f"{rustworkx_time / layered_time:>7.1f}x"
    )


def main():
    rustworkx_circuit = learn_circuit()
    layered = ProbabilisticCircuit.from_rustworkx(rustworkx_circuit)
    print(
        f"circuit: {len(rustworkx_circuit.nodes())} nodes, "
        f"{len(layered.layers)} layers"
    )

    sample = rustworkx_circuit.sample(1)[0]
    variables = layered.variables

    # rustworkx's log_conditional_in_place returns a wrong probability on circuits with
    # shared subcircuits (see memory rx-circuit-simplify-mass-loss), and JPTs share
    # leaves across branches -- so this only cross-checks the numpy result against its
    # own marginal + log_likelihood (independently validated in test_np), not against
    # rustworkx's return value. Timing is still measured on both.
    for number_of_conditioned in (1, 2, 3, 4):
        point = {variables[i]: sample[i] for i in range(number_of_conditioned)}
        conditioned_variables = list(point.keys())

        marginal_check = layered.marginal(conditioned_variables)
        row = np.array([[point[v] for v in conditioned_variables]])
        expected_log_probability = float(marginal_check.log_likelihood(row)[0])

        rustworkx_time, _ = fastest(
            lambda: rustworkx_circuit.__deepcopy__().log_conditional_in_place(point)
        )
        layered_time, (_, layered_log_probability) = fastest(
            lambda: layered.__deepcopy__().log_conditional_in_place(point)
        )
        assert np.isclose(expected_log_probability, layered_log_probability), (
            expected_log_probability,
            layered_log_probability,
        )
        report(
            f"conditional on {number_of_conditioned} variable(s)",
            rustworkx_time,
            layered_time,
        )

    print("\nrepeated conditioning on the same single-variable point (100x)")
    point = {variables[0]: sample[0]}
    rustworkx_time, _ = fastest(
        lambda: [
            rustworkx_circuit.__deepcopy__().log_conditional_in_place(point)
            for _ in range(100)
        ],
        repeats=2,
    )
    layered_time, _ = fastest(
        lambda: [
            layered.__deepcopy__().log_conditional_in_place(point) for _ in range(100)
        ],
        repeats=2,
    )
    report("100 conditionings", rustworkx_time, layered_time)


if __name__ == "__main__":
    main()
