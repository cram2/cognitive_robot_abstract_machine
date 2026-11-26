"""
The rules by which tensorized expectation maximization learns the parameters of every
type of layer: the sufficient statistics a batch gives a layer, read off the gradient of
the total log-likelihood, and the maximization step that updates the layer from them.
"""

from __future__ import annotations

# %% imports
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood.utils import recursive_subclasses
from typing_extensions import Generic, Self, TypeVar

from probabilistic_model.learning.parameters.expectation_maximization.exceptions import (
    UnsupportedLayerError,
)
from probabilistic_model.learning.parameters.expectation_maximization.expectation_maximization import (
    MaximizationRegularization,
)
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    BackendArray,
    NodeStateValues,
    NodeValues,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import Layer
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.edge_matrix import (
    RowReduction,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.sum_layer import (
    SumLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.discrete_layer import (
    DiscreteLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.gaussian_layer import (
    GaussianLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.probability_table import (
    DenseProbabilityTable,
)

LearnedLayerType = TypeVar("LearnedLayerType", bound=Layer)
"""
The type of layer whose parameters a :class:`LayerMaximization` learns.
"""

# %% sufficient statistics


@dataclass(kw_only=True)
class SufficientStatistics(ABC):
    """
    What the maximization step of a layer needs to know about the rows of the batches
    since the last update, accumulated by adding the statistics of every batch.
    """

    @property
    @abstractmethod
    def arrays(self) -> list[BackendArray]:
        """
        :return: Every array of the statistics, in the order :meth:`from_arrays` takes
            them.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_arrays(cls, *, arrays: list[BackendArray]) -> Self:
        """
        :param arrays: Arrays in the order and with the shapes of :attr:`arrays`.
        :return: The statistics made of these arrays.
        """
        raise NotImplementedError

    def added(self, *, other: Self) -> Self:
        """
        :param other: Statistics of the same layer on other rows.
        :return: The statistics of the rows of both.
        """
        return self.from_arrays(
            arrays=[own + others for own, others in zip(self.arrays, other.arrays)]
        )


@dataclass(kw_only=True)
class EdgeCounts(SufficientStatistics):
    """
    The expected number of rows that pass through every edge of a sum layer.
    """

    counts: BackendArray
    """
    The expected count of every edge, laid out as the edges of the layer store values.
    """

    @property
    def arrays(self) -> list[BackendArray]:
        return [self.counts]

    @classmethod
    def from_arrays(cls, *, arrays: list[BackendArray]) -> Self:
        [counts] = arrays
        return cls(counts=counts)


@dataclass(kw_only=True)
class GaussianStatistics(SufficientStatistics):
    """
    The expected number of rows of every Gaussian of a layer, and their first two
    moments about the location the statistics were computed with.
    """

    counts: NodeValues
    """
    The node count of every node.
    """

    centered_sums: NodeValues
    """
    The sum of the deviations of the rows from the location, each weighted by the node's
    share of it, for every node.
    """

    centered_squared_sums: NodeValues
    """
    The weighted sum of the squared deviations of the rows from the location, for every
    node.
    """

    @property
    def arrays(self) -> list[BackendArray]:
        return [self.counts, self.centered_sums, self.centered_squared_sums]

    @classmethod
    def from_arrays(cls, *, arrays: list[BackendArray]) -> Self:
        counts, centered_sums, centered_squared_sums = arrays
        return cls(
            counts=counts,
            centered_sums=centered_sums,
            centered_squared_sums=centered_squared_sums,
        )


@dataclass(kw_only=True)
class StateCounts(SufficientStatistics):
    """
    The expected number of rows with every state that every node of a probability table
    explains.
    """

    counts: NodeStateValues
    """
    The expected count of every state for every node.
    """

    @property
    def arrays(self) -> list[BackendArray]:
        return [self.counts]

    @classmethod
    def from_arrays(cls, *, arrays: list[BackendArray]) -> Self:
        [counts] = arrays
        return cls(counts=counts)


# %% rules


@dataclass(kw_only=True)
class LayerMaximization(Generic[LearnedLayerType], SubClassSafeGeneric, ABC):
    """
    How expectation maximization learns the parameters of one layer.

    The node count of a node is the derivative of the total log-likelihood of a batch by
    the log-value of the node, the expected number of rows the node explains.
    """

    layer: LearnedLayerType
    """
    The layer whose parameters are learned.
    """

    @classmethod
    def from_layer(cls, *, layer: Layer) -> LayerMaximization:
        """
        :param layer: A layer with parameters.
        :return: The rule of the closest type of the layer that a rule is declared for.
        :raises UnsupportedLayerError: If no rule is declared for any type of the layer.
        """
        rules = [
            rule
            for rule in recursive_subclasses(LayerMaximization)
            if not inspect.isabstract(rule)
        ]
        for layer_type in type(layer).__mro__:
            for rule in rules:
                if (
                    rule.get_type_of_generic_parameter(parameter=LearnedLayerType)
                    is layer_type
                ):
                    return rule(layer=layer)
        raise UnsupportedLayerError(layer_type=type(layer))

    @abstractmethod
    def expected_statistics(
        self, *, parameter_gradients: list[BackendArray], node_counts: NodeValues
    ) -> SufficientStatistics:
        """
        :param parameter_gradients: The derivative of the total log-likelihood of a
            batch by every parameter array, in the order of the layer's parameters.
        :param node_counts: The node count of every node of the layer on the batch.
        :return: The sufficient statistics of the batch, zero for zero gradients and
            counts.
        """
        raise NotImplementedError

    @abstractmethod
    def maximized_parameters(
        self,
        *,
        statistics: SufficientStatistics,
        regularization: MaximizationRegularization,
        step_size: float,
    ) -> list[BackendArray]:
        """
        :param statistics: The sufficient statistics accumulated since the last update,
            computed with the layer's parameters.
        :param regularization: How the learned weights and variances stay positive.
        :param step_size: The share of the way from the parameters to those that
            maximize the expected log-likelihood, one for a full update.
        :return: The updated parameter arrays; nodes no row reached keep theirs.
        """
        raise NotImplementedError


@dataclass(kw_only=True)
class SumLayerMaximization(LayerMaximization[SumLayer]):
    """
    Expectation maximization of the weights of a sum layer.
    """

    def expected_statistics(
        self, *, parameter_gradients: list[BackendArray], node_counts: NodeValues
    ) -> EdgeCounts:
        [log_weight_gradient] = parameter_gradients
        return EdgeCounts(
            counts=log_weight_gradient
            + self.layer.normalized_weights
            * self.layer.edges.values_of_rows(values=node_counts)
        )

    def maximized_parameters(
        self,
        *,
        statistics: EdgeCounts,
        regularization: MaximizationRegularization,
        step_size: float,
    ) -> list[BackendArray]:
        layer = self.layer
        namespace = layer.namespace
        is_edge = namespace.isfinite(layer.log_weights)
        counts = namespace.where(
            is_edge, statistics.counts + regularization.smoothing, 0.0
        )
        proportions = counts / layer.edges.values_of_rows(
            values=layer.edges.reduce_by_row(values=counts, reduction=RowReduction.SUM)
        )
        weights = (1.0 - step_size) * layer.normalized_weights + step_size * proportions
        log_weights = namespace.where(
            is_edge,
            namespace.log(
                namespace.clip(weights, min=regularization.minimum_mixture_proportion)
            ),
            -np.inf,
        )
        return [
            layer.copy_with_parameters(parameters=[log_weights]).normalized_log_weights
        ]


@dataclass(kw_only=True)
class GaussianLayerMaximization(LayerMaximization[GaussianLayer]):
    """
    Expectation maximization of the locations and scales of a Gaussian layer.
    """

    def expected_statistics(
        self, *, parameter_gradients: list[BackendArray], node_counts: NodeValues
    ) -> GaussianStatistics:
        location_gradient, log_scale_gradient = parameter_gradients
        variance = self.layer.scale**2
        return GaussianStatistics(
            counts=node_counts,
            centered_sums=variance * location_gradient,
            centered_squared_sums=variance * (log_scale_gradient + node_counts),
        )

    def maximized_parameters(
        self,
        *,
        statistics: GaussianStatistics,
        regularization: MaximizationRegularization,
        step_size: float,
    ) -> list[BackendArray]:
        layer = self.layer
        namespace = layer.namespace
        is_reached = statistics.counts > 0
        counts = namespace.where(is_reached, statistics.counts, 1.0)
        shift = statistics.centered_sums / counts
        variance = namespace.clip(
            statistics.centered_squared_sums / counts - shift**2,
            min=regularization.minimum_gaussian_variance,
        )
        blended_variance = (
            (1.0 - step_size) * layer.scale**2
            + step_size * variance
            + step_size * (1.0 - step_size) * shift**2
        )
        return [
            namespace.where(
                is_reached, layer.location + step_size * shift, layer.location
            ),
            namespace.where(
                is_reached, 0.5 * namespace.log(blended_variance), layer.log_scale
            ),
        ]


@dataclass(kw_only=True)
class DiscreteLayerMaximization(LayerMaximization[DiscreteLayer]):
    """
    Expectation maximization of the probabilities of a discrete layer whose table is a
    :class:`DenseProbabilityTable`.
    """

    def __post_init__(self) -> None:
        """
        :raises UnsupportedLayerError: If the table of the layer is not dense.
        """
        if not isinstance(self.layer.table, DenseProbabilityTable):
            raise UnsupportedLayerError(layer_type=type(self.layer.table))

    def expected_statistics(
        self, *, parameter_gradients: list[BackendArray], node_counts: NodeValues
    ) -> StateCounts:
        [log_probability_gradient] = parameter_gradients
        table = self.layer.table
        return StateCounts(
            counts=log_probability_gradient
            + table.normalized_probabilities
            * table.namespace.expand_dims(node_counts, axis=1)
        )

    def maximized_parameters(
        self,
        *,
        statistics: StateCounts,
        regularization: MaximizationRegularization,
        step_size: float,
    ) -> list[BackendArray]:
        table = self.layer.table
        namespace = table.namespace
        counts = namespace.clip(statistics.counts, min=0.0)
        totals = namespace.sum(counts, axis=1, keepdims=True)
        is_reached = totals > 0
        probabilities = (1.0 - step_size) * table.normalized_probabilities + (
            step_size * counts / namespace.where(is_reached, totals, 1.0)
        )
        is_possible = probabilities > 0
        log_probabilities = namespace.where(
            is_possible,
            namespace.log(namespace.where(is_possible, probabilities, 1.0)),
            -np.inf,
        )
        return [namespace.where(is_reached, log_probabilities, table.log_probabilities)]
