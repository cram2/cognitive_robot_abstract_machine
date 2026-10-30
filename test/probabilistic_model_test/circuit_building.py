"""
Building blocks the tests of :mod:`probabilistic_model` assemble circuits from.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from random_events.interval import closed
from random_events.variable import Continuous

from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    LeafUnit,
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.uniform_layer import (
    UniformLayer,
)

# %% leaves


def uniform_leaf(
    variable: Continuous, lower: float, upper: float, circuit: ProbabilisticCircuit
) -> LeafUnit:
    """
    :param variable: The variable of the distribution.
    :param lower: The lower bound of the closed interval the distribution is uniform on.
    :param upper: The upper bound of that interval.
    :param circuit: The circuit to add the leaf to.
    :return: A leaf of a uniform distribution over the closed interval.
    """
    return leaf(
        UniformDistribution(
            variable=variable, interval=closed(lower, upper).simple_sets[0]
        ),
        circuit,
    )


def uniform_layer_of(
    variable_index: int, intervals: list[tuple[float, float]]
) -> UniformLayer:
    """
    :param variable_index: The index of the variable of the layer.
    :param intervals: The bounds of the closed interval of every node.
    :return: A layer of one uniform distribution per interval.
    """
    return UniformLayer.from_distributions(
        variable_index,
        [
            UniformDistribution(
                variable=Continuous("placeholder"),
                interval=closed(lower, upper).simple_sets[0],
            )
            for lower, upper in intervals
        ],
    )


# %% inner units


def mixture_of_intervals(
    circuit: ProbabilisticCircuit,
    variable: Continuous,
    weighted_intervals: list[tuple[float, float, float]],
) -> SumUnit:
    """
    :param circuit: The circuit to build the units in.
    :param variable: The variable of the distributions.
    :param weighted_intervals: The weight and the bounds of every component.
    :return: A sum unit of one uniform distribution per closed interval.
    """
    mixture = SumUnit(probabilistic_circuit=circuit)
    for weight, lower, upper in weighted_intervals:
        mixture.add_subcircuit(
            uniform_leaf(variable, lower, upper, circuit), math.log(weight)
        )
    return mixture


def mixture_of_products_of_mixtures(
    circuit: ProbabilisticCircuit,
    first_leaves: tuple[LeafUnit, LeafUnit],
    second_leaves: tuple[LeafUnit, LeafUnit],
    weights_per_product: list[tuple[tuple[float, float], tuple[float, float]]],
) -> ProbabilisticCircuit:
    """
    Build an equal mixture of products, each of a mixture over the first leaves and a
    mixture over the second leaves, so that every product shares the same leaves.

    :param circuit: The circuit that holds the leaves and receives the inner units.
    :param first_leaves: The leaves over the first variable.
    :param second_leaves: The leaves over the second variable.
    :param weights_per_product: The weights of the mixture over the first leaves and of
        the mixture over the second leaves, for every product.
    :return: The circuit.
    """
    root = SumUnit(probabilistic_circuit=circuit)
    for weights_of_mixtures in weights_per_product:
        product = ProductUnit(probabilistic_circuit=circuit)
        root.add_subcircuit(product, np.log(1 / len(weights_per_product)))
        for leaves, weights in zip((first_leaves, second_leaves), weights_of_mixtures):
            mixture = SumUnit(probabilistic_circuit=circuit)
            product.add_subcircuit(mixture)
            for mixed_leaf, weight in zip(leaves, weights):
                mixture.add_subcircuit(mixed_leaf, np.log(weight))
    return circuit


# %% learned circuits


def correlated_gaussian_samples(
    number_of_variables: int, amount: int = 1000
) -> pd.DataFrame:
    """
    :param number_of_variables: The number of variables, named ``x_0``, ``x_1``, ...
    :param amount: The number of samples.
    :return: Samples of a centered Gaussian with a random covariance, drawn from the
        global random state of :mod:`numpy`.
    """
    covariance = np.random.uniform(0, 1, (number_of_variables, number_of_variables))
    covariance = covariance @ covariance.T
    samples = np.random.multivariate_normal(
        np.zeros(number_of_variables), covariance, amount
    )
    return pd.DataFrame(
        samples, columns=[f"x_{index}" for index in range(number_of_variables)]
    )


def fit_joint_probability_tree(
    data: pd.DataFrame, min_samples_per_quantile: int
) -> ProbabilisticCircuit:
    """
    :param data: The data to fit the tree to.
    :param min_samples_per_quantile: The minimal number of samples per quantile of the
        variables inferred from the data.
    :return: A joint probability tree fitted to the data.
    """
    variables = infer_variables_from_dataframe(
        data, min_samples_per_quantile=min_samples_per_quantile
    )
    return JointProbabilityTree(
        annotated_variables=variables, min_samples_per_leaf=0.1
    ).fit(data)
