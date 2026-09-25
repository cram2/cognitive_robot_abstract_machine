from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from random_events.variable import Variable
from sortedcontainers import SortedSet
from typing_extensions import List, Self

from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    NodeValues,
    SampleColumn,
    SampleNodeValues,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.continuous_layer_with_density import (
    ContinuousLayerWithFiniteSupport,
)


@dataclass(eq=False, repr=False)
class UniformLayer(ContinuousLayerWithFiniteSupport):
    """
    A layer of uniform distributions over one continuous variable.
    """

    @property
    def number_of_own_parameters(self) -> int:
        return 2 * self.number_of_nodes

    def log_probability_density_function_value(self) -> NodeValues:
        """
        :return: The log-density of every node.
        """
        with np.errstate(divide="ignore"):
            return -np.log(self.upper - self.lower)

    def log_likelihood_of_nodes_from_column(
        self, values: SampleColumn
    ) -> SampleNodeValues:
        return np.where(
            self.included_condition(values),
            self.log_probability_density_function_value(),
            -np.inf,
        )

    def cumulative_distribution_of_nodes_from_column(
        self, values: SampleColumn
    ) -> SampleNodeValues:
        column = np.asarray(values, dtype=float).reshape(-1, 1)
        result = (column - self.lower) / (self.upper - self.lower)
        return np.clip(result, 0.0, 1.0)

    def moment_of_nodes_own(
        self, order: int, center: float, variable: Variable
    ) -> NodeValues:
        return self.antiderivative_of_moment_at(
            self.upper, order, center
        ) - self.antiderivative_of_moment_at(self.lower, order, center)

    def antiderivative_of_moment_at(
        self, bound: NodeValues, order: int, center: float
    ) -> NodeValues:
        """
        :param bound: One value per node.
        :param order: The order of the moment.
        :param center: The center of the moment.
        :return: The antiderivative of ``density * (value - center) ** order`` of every
            node at its bound.
        """
        density = np.exp(self.log_probability_density_function_value())
        return density * (bound - center) ** (order + 1) / (order + 1)

    def node_distribution(self, index: int, variable: Variable) -> UniformDistribution:
        return UniformDistribution(
            variable=variable, interval=self.simple_interval_of(index)
        )

    @classmethod
    def from_distributions(
        cls, variable_index: int, distributions: List[UniformDistribution]
    ) -> Self:
        interval = np.array(
            [
                [distribution.interval.lower, distribution.interval.upper]
                for distribution in distributions
            ],
            dtype=float,
        )
        bounds = np.array(
            [
                [int(distribution.interval.left), int(distribution.interval.right)]
                for distribution in distributions
            ],
            dtype=np.int64,
        )
        return cls(variable_index, interval, bounds)

    def sample_of_node(
        self, node: int, amount: int, variables: SortedSet
    ) -> SampleColumn:
        return np.random.uniform(self.lower[node], self.upper[node], amount)
