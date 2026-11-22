"""
Tensorized input layers whose leaves are Gaussian and truncated Gaussian distributions.
"""

from __future__ import annotations

import dataclasses
import copy
import math
from dataclasses import dataclass

import numpy as np
from random_events.interval import SimpleInterval, reals
from random_events.variable import Variable
from krrood.patterns.classproperty import classproperty
from scipy.stats import norm, truncnorm
from sortedcontainers import SortedSet
from typing_extensions import Self

from probabilistic_model.distributions.gaussian import (
    GaussianDistribution,
    TruncatedGaussianDistribution,
)
from probabilistic_model.exceptions import ShapeMismatchError
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    BackendArray,
    NodeIndices,
    NodeMask,
    NodeValues,
    SampleColumn,
    SampleNodeValues,
    VariableValues,
)
from probabilistic_model.probabilistic_circuit.tensorized.exceptions import (
    NonPositiveScaleError,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import Layer
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.continuous_layer_with_density import (
    ContinuousLayerWithDensity,
    ContinuousLayerWithConstrainedSupport,
)
from probabilistic_model.probabilistic_circuit.tensorized.normal_distribution_functions import (
    NormalDistributionFunctions,
)
from probabilistic_model.probabilistic_circuit.tensorized.structural_query import (
    LayerWithLogProbabilities,
)


@dataclass(eq=False, repr=False)
class HasLocationAndScale:
    """
    Base class for the layers whose nodes are described by a Gaussian with a location
    and a scale.
    """

    location: NodeValues
    """
    The mean of the Gaussian of every node.
    """

    log_scale: NodeValues
    """
    The natural logarithm of the standard deviation of the Gaussian of every node.
    """

    @property
    def scale(self) -> NodeValues:
        """
        :return: The standard deviation of the Gaussian of every node.
        """
        return self.namespace.exp(self.log_scale)

    @property
    def parameters(self) -> list[BackendArray]:
        """
        :return: The location and the logarithm of the scale of every node.
        """
        return [self.location, self.log_scale]

    def copy_with_parameters(self, *, parameters: list[BackendArray]) -> Self:
        """
        Replace the locations and the logarithms of the scales, see
        :meth:`Layer.copy_with_parameters`.
        """
        location, log_scale = parameters
        return dataclasses.replace(self, location=location, log_scale=log_scale)

    @classproperty
    def minimum_scale(cls) -> float:
        """
        The smallest scale :meth:`projected_parameters` keeps, so that no density
        becomes infinitely narrow.
        """
        return 1e-3

    def projected_parameters(
        self, *, parameters: list[BackendArray]
    ) -> list[BackendArray]:
        """
        Keep the logarithm of every scale at least the logarithm of
        :attr:`minimum_scale`, see :meth:`Layer.projected_parameters`.
        """
        *others, log_scale = parameters
        return others + [
            self.namespace.clip(log_scale, min=math.log(self.minimum_scale))
        ]

    def validate_own(self):
        """
        :raises ShapeMismatchError: If there is not one scale per location.
        :raises NonPositiveScaleError: If a scale is not positive.
        """
        super().validate_own()
        if self.location.shape != self.log_scale.shape:
            raise ShapeMismatchError(self.location.shape, self.log_scale.shape)
        # the negated comparison also catches a scale that is nan
        non_positive = ~(self.scale > 0)
        if non_positive.any():
            raise NonPositiveScaleError(self.scale[non_positive])

    def normal_log_density(self, *, column: BackendArray) -> SampleNodeValues:
        """
        :param column: The values with shape (#events, 1).
        :return: The log-density of the Gaussian of every node at every value, shape
            (#events, #nodes).
        """
        return NormalDistributionFunctions.from_arrays(
            self.location, self.scale
        ).log_density(column, self.location, self.scale)

    def normal_cumulative_distribution(self, values: NodeValues) -> NodeValues:
        """
        :param values: Values that broadcast against the nodes, such as one per node or
            a column of shape (#events, 1).
        :return: The cumulative distribution function of the Gaussian of every node at
            the values.
        """
        return NormalDistributionFunctions.from_arrays(
            self.location, self.scale
        ).cumulative_distribution(values, self.location, self.scale)

    def has_equal_parameters(self, *, other: Self) -> bool:
        """
        :param other: A layer that :meth:`LeafLayer.concatenate` may join with this one.
        :return: Whether every node of both layers has the same location and scale.
        """
        return np.array_equal(self.location, other.location) and np.array_equal(
            self.log_scale, other.log_scale
        )


@dataclass(eq=False, repr=False)
class GaussianLayer(HasLocationAndScale, ContinuousLayerWithDensity):
    """
    A layer of Gaussian distributions over one continuous variable.
    """

    @property
    def number_of_nodes(self) -> int:
        return len(self.location)

    @property
    def number_of_own_parameters(self) -> int:
        return 2 * self.number_of_nodes

    def log_likelihood_of_nodes_from_column(
        self, values: SampleColumn
    ) -> SampleNodeValues:
        return self.normal_log_density(column=self.float_column(values=values))

    def cumulative_distribution_of_nodes_from_column(
        self, values: SampleColumn
    ) -> SampleNodeValues:
        return self.normal_cumulative_distribution(self.float_column(values=values))

    def raw_moments(self, order: int) -> list[NodeValues]:
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
            raw_moment = self.namespace.zeros_like(self.location)
            for j in range(math.floor(current_order / 2) + 1):
                raw_moment = raw_moment + (
                    math.comb(current_order, 2 * j)
                    * self.location ** (current_order - 2 * j)
                    * self.scale ** (2 * j)
                    * math.factorial(2 * j)
                    / (math.factorial(j) * (2**j))
                )
            result.append(raw_moment)
        return result

    def moment_of_nodes_own(
        self, *, order: int, center: float, variable: Variable
    ) -> NodeValues:
        """
        Like :meth:`~probabilistic_model.probabilistic_circuit.tensorized.input_layer.ba
        se.UnivariateInputLayer.moment_of_nodes_own`, in closed form from the raw
        moments of :meth:`raw_moments`.
        """
        raw_moments = self.raw_moments(order)
        result = self.namespace.zeros_like(self.location)
        for current_order in range(order + 1):
            result = result + (
                math.comb(order, current_order)
                * raw_moments[current_order]
                * (-center) ** (order - current_order)
            )
        return result

    def node_distribution(
        self, *, index: int, variable: Variable
    ) -> GaussianDistribution:
        return GaussianDistribution(
            variable=variable,
            location=float(self.location[index]),
            scale=float(self.scale[index]),
        )

    @classmethod
    def from_distributions(
        cls, *, variable_index: int, distributions: list[GaussianDistribution]
    ) -> Self:
        return cls(
            np.full(len(distributions), variable_index, dtype=np.int64),
            np.array([distribution.location for distribution in distributions]),
            np.log([distribution.scale for distribution in distributions]),
        )

    def select_nodes(self, *, mask: NodeMask) -> Self:
        return self.__class__(
            self.node_variables[mask], self.location[mask], self.log_scale[mask]
        )

    @classmethod
    def concatenate(cls, *, layers: list[Self]) -> Self:
        return cls(
            np.concatenate([layer.node_variables for layer in layers]),
            np.concatenate([layer.location for layer in layers]),
            np.concatenate([layer.log_scale for layer in layers]),
        )

    def sample_of_nodes(
        self, *, nodes: NodeIndices, variables: SortedSet
    ) -> SampleColumn:
        """
        Like :meth:`UnivariateInputLayer.sample_of_nodes`, drawing from the Gaussians of
        all nodes at once.
        """
        return norm.rvs(loc=self.location[nodes], scale=self.scale[nodes])

    def apply_translation_own(self, *, translation: VariableValues) -> None:
        """
        Translate the locations of the nodes, see :meth:`Layer.apply_translation_own`.
        """
        self.location = self.location + self.node_variable_values(values=translation)

    def apply_scaling_own(self, *, scaling: VariableValues) -> None:
        """
        Scale the locations and the scales of the nodes, see
        :meth:`Layer.apply_scaling_own`.
        """
        node_scaling = self.node_variable_values(values=scaling)
        self.location = self.location * node_scaling
        self.log_scale = self.log_scale + self.namespace.log(node_scaling)

    def type_of_layer_truncated_to_interval(
        self, interval: SimpleInterval
    ) -> type[Layer]:
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
            return LayerWithLogProbabilities(copy.deepcopy(self), log_probabilities)

        interval_of_nodes = np.tile([lower, upper], (self.number_of_nodes, 1))
        bounds_of_nodes = np.tile(
            np.array([int(interval.left), int(interval.right)], dtype=np.int64),
            (self.number_of_nodes, 1),
        )
        return LayerWithLogProbabilities(
            TruncatedGaussianLayer(
                self.node_variables.copy(),
                interval_of_nodes,
                bounds_of_nodes,
                self.location.copy(),
                self.log_scale.copy(),
            ),
            log_probabilities,
        )


@dataclass(eq=False, repr=False)
class TruncatedGaussianLayer(
    HasLocationAndScale, ContinuousLayerWithConstrainedSupport
):
    """
    A layer of truncated Gaussian distributions over one continuous variable.

    The location and scale are those of the untruncated Gaussian of every node.

    This is the layer that truncating a :class:`GaussianLayer` to a bounded interval
    produces.
    """

    @property
    def parameters(self) -> list[BackendArray]:
        """
        :return: The supports, the locations and the logarithms of the scales of the
            nodes.
        """
        return [self.interval, self.location, self.log_scale]

    def copy_with_parameters(self, *, parameters: list[BackendArray]) -> Self:
        """
        Replace the supports, the locations and the logarithms of the scales, see
        :meth:`Layer.copy_with_parameters`.
        """
        interval, location, log_scale = parameters
        return dataclasses.replace(
            self, interval=interval, location=location, log_scale=log_scale
        )

    def validate_own(self):
        HasLocationAndScale.validate_own(self)
        ContinuousLayerWithConstrainedSupport.validate_own(self)

    def has_equal_parameters(self, *, other: Self) -> bool:
        """
        Like :meth:`HasLocationAndScale.has_equal_parameters`, comparing the supports of
        the nodes too.
        """
        return HasLocationAndScale.has_equal_parameters(
            self, other=other
        ) and ContinuousLayerWithConstrainedSupport.has_equal_parameters(
            self, other=other
        )

    @property
    def number_of_own_parameters(self) -> int:
        return 4 * self.number_of_nodes

    @property
    def cumulative_distribution_to_lower(self) -> NodeValues:
        """
        :return: The untruncated cumulative distribution at the lower bound of every
            node.
        """
        return self.normal_cumulative_distribution(self.lower)

    @property
    def normalizing_constant(self) -> NodeValues:
        """
        :return: The probability of the support of every node under its untruncated
            Gaussian.
        """
        return (
            self.normal_cumulative_distribution(self.upper)
            - self.cumulative_distribution_to_lower
        )

    def log_likelihood_of_nodes_from_column(
        self, values: SampleColumn
    ) -> SampleNodeValues:
        with np.errstate(divide="ignore"):
            density = self.normal_log_density(
                column=self.float_column(values=values)
            ) - self.namespace.log(self.normalizing_constant)
        return self.namespace.where(self.included_condition(values), density, -np.inf)

    def cumulative_distribution_of_nodes_from_column(
        self, values: SampleColumn
    ) -> SampleNodeValues:
        column = self.float_column(values=values)
        left_included = self.namespace.where(
            self.namespace.asarray(self.left_closed, device=self.device),
            self.lower <= column,
            self.lower < column,
        )
        untruncated = self.normal_cumulative_distribution(column)
        result = (
            untruncated - self.cumulative_distribution_to_lower
        ) / self.normalizing_constant
        return self.namespace.where(
            left_included,
            self.namespace.minimum(self.namespace.maximum(result, 0.0), 1.0),
            0.0,
        )

    def sample_of_nodes(
        self, *, nodes: NodeIndices, variables: SortedSet
    ) -> SampleColumn:
        """
        Like :meth:`UnivariateInputLayer.sample_of_nodes`, drawing from the truncated
        Gaussians of all nodes at once.
        """
        location, scale = self.location[nodes], self.scale[nodes]
        return truncnorm.rvs(
            a=(self.lower[nodes] - location) / scale,
            b=(self.upper[nodes] - location) / scale,
            loc=location,
            scale=scale,
        )

    def node_distribution(
        self, *, index: int, variable: Variable
    ) -> TruncatedGaussianDistribution:
        return TruncatedGaussianDistribution(
            variable=variable,
            interval=self.simple_interval_of(index),
            location=float(self.location[index]),
            scale=float(self.scale[index]),
        )

    @classmethod
    def from_distributions(
        cls, *, variable_index: int, distributions: list[TruncatedGaussianDistribution]
    ) -> Self:
        return cls(
            np.full(len(distributions), variable_index, dtype=np.int64),
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
            np.log([distribution.scale for distribution in distributions]),
        )

    def select_nodes(self, *, mask: NodeMask) -> Self:
        """
        Keep the supports, locations and scales of the selected nodes, see
        :meth:`UnivariateInputLayer.select_nodes`.
        """
        return self.__class__(
            self.node_variables[mask],
            self.interval[mask],
            self.bounds[mask],
            self.location[mask],
            self.log_scale[mask],
        )

    @classmethod
    def concatenate(cls, *, layers: list[Self]) -> Self:
        """
        Concatenate the supports, locations and scales of the layers, see
        :meth:`LeafLayer.concatenate`.
        """
        return cls(
            np.concatenate([layer.node_variables for layer in layers]),
            np.concatenate([layer.interval for layer in layers]),
            np.concatenate([layer.bounds for layer in layers]),
            np.concatenate([layer.location for layer in layers]),
            np.concatenate([layer.log_scale for layer in layers]),
        )

    def apply_translation_own(self, *, translation: VariableValues) -> None:
        """
        Translate the supports and the locations of the nodes, see
        :meth:`Layer.apply_translation_own`.
        """
        super().apply_translation_own(translation=translation)
        self.location = self.location + self.node_variable_values(values=translation)

    def apply_scaling_own(self, *, scaling: VariableValues) -> None:
        """
        Scale the supports, the locations and the scales of the nodes, see
        :meth:`Layer.apply_scaling_own`.
        """
        super().apply_scaling_own(scaling=scaling)
        node_scaling = self.node_variable_values(values=scaling)
        self.location = self.location * node_scaling
        self.log_scale = self.log_scale + self.namespace.log(node_scaling)
