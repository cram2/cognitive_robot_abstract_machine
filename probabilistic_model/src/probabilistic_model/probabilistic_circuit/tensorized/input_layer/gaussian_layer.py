from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from random_events.interval import SimpleInterval, reals
from random_events.variable import Variable
from scipy.stats import norm
from sortedcontainers import SortedSet
from typing_extensions import List, Self, Type

from probabilistic_model.distributions.gaussian import (
    GaussianDistribution,
    TruncatedGaussianDistribution,
)
from probabilistic_model.exceptions import ShapeMismatchError
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    NodeMask,
    NodeValues,
    SampleColumn,
    SampleNodeValues,
    VariableValues,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import Layer
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.continuous_layer_with_density import (
    ContinuousLayerWithDensity,
    ContinuousLayerWithFiniteSupport,
)
from probabilistic_model.probabilistic_circuit.tensorized.structural_query import (
    LayerWithLogProbabilities,
)


@dataclass(eq=False, repr=False)
class GaussianLayer(ContinuousLayerWithDensity):
    """
    A layer of Gaussian distributions over one continuous variable.
    """

    location: NodeValues
    """
    The mean of every node.
    """

    scale: NodeValues
    """
    The standard deviation of every node.
    """

    @property
    def number_of_nodes(self) -> int:
        return len(self.location)

    @property
    def number_of_own_parameters(self) -> int:
        return 2 * self.number_of_nodes

    def validate_own(self):
        if self.location.shape != self.scale.shape:
            raise ShapeMismatchError(self.location.shape, self.scale.shape)

    def log_likelihood_of_nodes_from_column(
        self, values: SampleColumn
    ) -> SampleNodeValues:
        column = np.asarray(values, dtype=float).reshape(-1, 1)
        return norm.logpdf(column, loc=self.location, scale=self.scale)

    def cumulative_distribution_of_nodes_from_column(
        self, values: SampleColumn
    ) -> SampleNodeValues:
        column = np.asarray(values, dtype=float).reshape(-1, 1)
        return norm.cdf(column, loc=self.location, scale=self.scale)

    def raw_moments(self, order: int) -> List[NodeValues]:
        r"""
        The raw moments of order ``0`` to ``order`` of every node.

        .. math::

            E(X^n) = \sum_{j=0}^{\lfloor \frac{n}{2}\rfloor}
            \binom{n}{2j}\dfrac{\mu^{n-2j}\sigma^{2j}(2j)!}{j!2^j}

        :param order: The highest order to calculate.
        :return: The raw moments of every node, one entry per order.
        """
        result = []
        for current_order in range(order + 1):
            raw_moment = np.zeros(self.number_of_nodes)
            for j in range(math.floor(current_order / 2) + 1):
                raw_moment += (
                    math.comb(current_order, 2 * j)
                    * self.location ** (current_order - 2 * j)
                    * self.scale ** (2 * j)
                    * math.factorial(2 * j)
                    / (math.factorial(j) * (2**j))
                )
            result.append(raw_moment)
        return result

    def moment_of_nodes_own(
        self, order: int, center: float, variable: Variable
    ) -> NodeValues:
        raw_moments = self.raw_moments(order)
        result = np.zeros(self.number_of_nodes)
        for current_order in range(order + 1):
            result += (
                math.comb(order, current_order)
                * raw_moments[current_order]
                * (-center) ** (order - current_order)
            )
        return result

    def node_distribution(self, index: int, variable: Variable) -> GaussianDistribution:
        return GaussianDistribution(
            variable=variable,
            location=float(self.location[index]),
            scale=float(self.scale[index]),
        )

    @classmethod
    def from_distributions(
        cls, variable_index: int, distributions: List[GaussianDistribution]
    ) -> Self:
        return cls(
            variable_index,
            np.array([distribution.location for distribution in distributions]),
            np.array([distribution.scale for distribution in distributions]),
        )

    def select_nodes(self, mask: NodeMask) -> Self:
        return self.__class__(self.variable, self.location[mask], self.scale[mask])

    @classmethod
    def concatenate(cls, layers: List[Self]) -> Self:
        return cls(
            layers[0].variable,
            np.concatenate([layer.location for layer in layers]),
            np.concatenate([layer.scale for layer in layers]),
        )

    def sample_of_node(
        self, node: int, amount: int, variables: SortedSet
    ) -> SampleColumn:
        return norm.rvs(loc=self.location[node], scale=self.scale[node], size=amount)

    def apply_translation_own(self, translation: VariableValues):
        self.location = self.location + translation[self.variable]

    def apply_scaling_own(self, scaling: VariableValues):
        self.location = self.location * scaling[self.variable]
        self.scale = self.scale * scaling[self.variable]

    def type_of_layer_truncated_to_interval(
        self, interval: SimpleInterval
    ) -> Type[Layer]:
        if interval.as_composite_set() == reals():
            return GaussianLayer
        return TruncatedGaussianLayer

    def log_truncated_of_non_singleton_interval(
        self, interval: SimpleInterval
    ) -> LayerWithLogProbabilities:
        """
        Truncate every node to a simple interval.

        Truncating to the whole real line leaves every node a Gaussian. Any other
        interval turns the layer into a :class:`TruncatedGaussianLayer` bounded by it.

        :param interval: The simple interval, which is not a singleton.
        :return: The truncated layer and the log-probability of the interval under every
            node.
        """
        lower, upper = float(interval.lower), float(interval.upper)
        cumulative = self.cumulative_distribution_of_nodes_from_column(
            np.array([lower, upper])
        )
        probability = cumulative[1] - cumulative[0]
        alive = probability > 0
        log_probabilities = np.where(
            alive, np.log(np.where(alive, probability, 1.0)), -np.inf
        )

        if self.type_of_layer_truncated_to_interval(interval) is GaussianLayer:
            return LayerWithLogProbabilities(self.__deepcopy__(), log_probabilities)

        interval_of_nodes = np.tile([lower, upper], (self.number_of_nodes, 1))
        bounds_of_nodes = np.tile(
            np.array([int(interval.left), int(interval.right)], dtype=np.int64),
            (self.number_of_nodes, 1),
        )
        return LayerWithLogProbabilities(
            TruncatedGaussianLayer(
                self.variable,
                interval_of_nodes,
                bounds_of_nodes,
                self.location.copy(),
                self.scale.copy(),
            ),
            log_probabilities,
        )

    def __deepcopy__(self, memo=None) -> GaussianLayer:
        if memo is None:
            memo = {}
        if id(self) in memo:
            return memo[id(self)]
        result = self.__class__(self.variable, self.location.copy(), self.scale.copy())
        memo[id(self)] = result
        return result


@dataclass(eq=False, repr=False)
class TruncatedGaussianLayer(ContinuousLayerWithFiniteSupport):
    """
    A layer of truncated Gaussian distributions over one continuous variable.

    This is the layer that truncating a :class:`GaussianLayer` to a bounded interval
    produces.
    """

    location: NodeValues
    """
    The mean of the untruncated Gaussian of every node.
    """

    scale: NodeValues
    """
    The standard deviation of the untruncated Gaussian of every node.
    """

    @property
    def number_of_own_parameters(self) -> int:
        return 4 * self.number_of_nodes

    @property
    def cumulative_distribution_to_lower(self) -> NodeValues:
        """
        :return: The untruncated cumulative distribution at the lower bound of every
            node.
        """
        return norm.cdf(self.lower, loc=self.location, scale=self.scale)

    @property
    def normalizing_constant(self) -> NodeValues:
        """
        :return: The probability of the support of every node under its untruncated
            Gaussian.
        """
        return (
            norm.cdf(self.upper, loc=self.location, scale=self.scale)
            - self.cumulative_distribution_to_lower
        )

    def log_likelihood_of_nodes_from_column(
        self, values: SampleColumn
    ) -> SampleNodeValues:
        column = np.asarray(values, dtype=float).reshape(-1, 1)
        with np.errstate(divide="ignore"):
            density = norm.logpdf(column, loc=self.location, scale=self.scale) - np.log(
                self.normalizing_constant
            )
        return np.where(self.included_condition(values), density, -np.inf)

    def cumulative_distribution_of_nodes_from_column(
        self, values: SampleColumn
    ) -> SampleNodeValues:
        column = np.asarray(values, dtype=float).reshape(-1, 1)
        left_included = np.where(
            self.left_closed, self.lower <= column, self.lower < column
        )
        untruncated = norm.cdf(column, loc=self.location, scale=self.scale)
        result = (
            untruncated - self.cumulative_distribution_to_lower
        ) / self.normalizing_constant
        return np.minimum(1.0, np.where(left_included, result, 0.0))

    def node_distribution(
        self, index: int, variable: Variable
    ) -> TruncatedGaussianDistribution:
        return TruncatedGaussianDistribution(
            variable=variable,
            interval=self.simple_interval_of(index),
            location=float(self.location[index]),
            scale=float(self.scale[index]),
        )

    @classmethod
    def from_distributions(
        cls, variable_index: int, distributions: List[TruncatedGaussianDistribution]
    ) -> Self:
        return cls(
            variable_index,
            np.array(
                [
                    [distribution.interval.lower, distribution.interval.upper]
                    for distribution in distributions
                ],
                dtype=float,
            ),
            np.array(
                [
                    [int(distribution.interval.left), int(distribution.interval.right)]
                    for distribution in distributions
                ],
                dtype=np.int64,
            ),
            np.array([distribution.location for distribution in distributions]),
            np.array([distribution.scale for distribution in distributions]),
        )

    def select_nodes(self, mask: NodeMask) -> Self:
        return self.__class__(
            self.variable,
            self.interval[mask],
            self.bounds[mask],
            self.location[mask],
            self.scale[mask],
        )

    @classmethod
    def concatenate(cls, layers: List[Self]) -> Self:
        return cls(
            layers[0].variable,
            np.concatenate([layer.interval for layer in layers]),
            np.concatenate([layer.bounds for layer in layers]),
            np.concatenate([layer.location for layer in layers]),
            np.concatenate([layer.scale for layer in layers]),
        )

    def apply_translation_own(self, translation: VariableValues):
        super().apply_translation_own(translation)
        self.location = self.location + translation[self.variable]

    def apply_scaling_own(self, scaling: VariableValues):
        super().apply_scaling_own(scaling)
        self.location = self.location * scaling[self.variable]
        self.scale = self.scale * scaling[self.variable]

    def __deepcopy__(self, memo=None) -> TruncatedGaussianLayer:
        if memo is None:
            memo = {}
        if id(self) in memo:
            return memo[id(self)]
        result = self.__class__(
            self.variable,
            self.interval.copy(),
            self.bounds.copy(),
            self.location.copy(),
            self.scale.copy(),
        )
        memo[id(self)] = result
        return result
