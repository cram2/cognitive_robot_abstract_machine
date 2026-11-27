"""
Expectation maximization of the parameters of a layered circuit on any array library
that differentiates automatically: the expectation step is the gradient of the total
log-likelihood of a batch, the maximization step the update of every layer.
"""

from __future__ import annotations

# %% imports
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from array_api_compat import array_namespace
from typing_extensions import Any

from probabilistic_model.learning.parameters.circuit_parameter_learning import (
    CircuitParameterLearning,
    CircuitParameterTraining,
    LayerType,
)
from probabilistic_model.learning.parameters.expectation_maximization.expectation_maximization import (
    MaximizationRegularization,
)
from probabilistic_model.learning.parameters.expectation_maximization.layer_maximization import (
    LayerMaximization,
    SufficientStatistics,
)
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    BackendArray,
    SampleArray,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import Layer
from probabilistic_model.probabilistic_circuit.tensorized.query_cache import (
    QueryCache,
    QueryCacheKey,
)

# %% node counts


@dataclass(kw_only=True)
class NodeCountQueryCache(QueryCache):
    """
    A pass that adds an offset to the log-likelihood of every node of some layers, so
    that the derivative by the offset of a node is its node count.
    """

    node_offsets: dict[QueryCacheKey, BackendArray]
    """
    The offset of every node, one array per layer, by the key of the log-likelihood
    query of the layer.
    """

    def stored_result(self, *, key: QueryCacheKey, result: Any) -> Any:
        """
        The result shifted by the offsets of its layer, if it is the log-likelihood of a
        layer with offsets, see :meth:`QueryCache.stored_result`.
        """
        if key not in self.node_offsets:
            return result
        return result + self.node_offsets[key]


# %% expectation maximization


@dataclass(kw_only=True)
class TensorizedExpectationMaximization(CircuitParameterLearning[LayerType], ABC):
    """
    Expectation maximization of every parameter of a circuit, accumulating the
    sufficient statistics of every batch and updating after :attr:`batches_per_update`
    batches, with the array library of its subclass.
    """

    batches_per_update: int | None = None
    """
    The batches after which the parameters are updated, once at the end of every epoch
    if ``None``, which is full-batch expectation maximization.
    """

    step_size: float = 1.0
    """
    The share of the way from the parameters to the maximizing ones that an update
    moves, one for full updates and less for online expectation maximization.
    """

    regularization: MaximizationRegularization = field(
        default_factory=MaximizationRegularization
    )
    """
    How the maximization step keeps the learned weights and variances away from zero.
    """

    layer_maximization_types: list[type[LayerMaximization]] = field(init=False)
    """
    The rule of every learned layer, in the order of :meth:`learned_layers`.
    """

    def __post_init__(self) -> None:
        """
        Find the rule of every learned layer.

        :raises UnsupportedLayerError: If a layer of the circuit has parameters that
            expectation maximization cannot learn.
        """
        super().__post_init__()
        self.layer_maximization_types = [
            type(LayerMaximization.from_layer(layer=layer))
            for layer in self.learned_layers(root=self.layer)
        ]

    @staticmethod
    def learned_layers(*, root: Layer) -> list[Layer]:
        """
        :param root: The root of the circuit or of a copy of it.
        :return: The layers with parameters, in the order of ``all_layers``.
        """
        return [layer for layer in root.all_layers() if layer.parameters]

    def layer_maximizations(self, *, root: Layer) -> list[LayerMaximization]:
        """
        :param root: The root of a copy of the circuit.
        :return: The rule of every learned layer of the copy.
        """
        return [
            layer_maximization_type(layer=layer)
            for layer_maximization_type, layer in zip(
                self.layer_maximization_types, self.learned_layers(root=root)
            )
        ]

    @abstractmethod
    def zero_node_counts(self) -> list[BackendArray]:
        """
        :return: A zero for every node of every learned layer, on the device and in the
            floating point type of the parameters.
        """
        raise NotImplementedError

    def zero_statistics(
        self, *, parameters: list[BackendArray]
    ) -> list[SufficientStatistics]:
        """
        :param parameters: Arrays in the order of the circuit's parameters.
        :return: The statistics of no rows, for every learned layer.
        """
        namespace = array_namespace(*parameters)
        return self.statistics_from_gradients(
            parameters=parameters,
            parameter_gradients=[
                namespace.zeros_like(parameter) for parameter in parameters
            ],
            node_counts=self.zero_node_counts(),
        )

    def total_log_likelihood(
        self,
        *,
        parameters: list[BackendArray],
        node_offsets: list[BackendArray],
        data: SampleArray,
    ) -> BackendArray:
        """
        :param parameters: Arrays in the order of the circuit's parameters.
        :param node_offsets: An offset for every node of every learned layer, zero to
            compute node counts.
        :param data: The rows of a batch.
        :return: The sum of the log-likelihoods of the rows under the circuit with these
            parameters, each learned node's log-likelihood shifted by its offset.
        """
        root = self.layer.copy_with_circuit_parameters(parameters=parameters)
        cache = NodeCountQueryCache(
            node_offsets={
                QueryCacheKey.from_query(
                    query=type(layer).log_likelihood_of_nodes, layer=layer
                ): offset
                for layer, offset in zip(self.learned_layers(root=root), node_offsets)
            }
        )
        log_likelihoods = root.log_likelihood_of_nodes(events=data, cache=cache)[:, 0]
        return log_likelihoods.sum()

    def statistics_from_gradients(
        self,
        *,
        parameters: list[BackendArray],
        parameter_gradients: list[BackendArray],
        node_counts: list[BackendArray],
    ) -> list[SufficientStatistics]:
        """
        :param parameters: Arrays in the order of the circuit's parameters.
        :param parameter_gradients: The derivative of :meth:`total_log_likelihood` by
            every parameter array.
        :param node_counts: Its derivative by the offsets of every learned layer.
        :return: The sufficient statistics of every learned layer.
        """
        root = self.layer.copy_with_circuit_parameters(parameters=parameters)
        gradients_by_layer = [
            gradients
            for gradients in root.parameters_by_layer(parameters=parameter_gradients)
            if gradients
        ]
        return [
            layer_maximization.expected_statistics(
                parameter_gradients=gradients, node_counts=counts
            )
            for layer_maximization, gradients, counts in zip(
                self.layer_maximizations(root=root), gradients_by_layer, node_counts
            )
        ]

    def maximized_circuit_parameters(
        self,
        *,
        parameters: list[BackendArray],
        statistics: list[SufficientStatistics],
    ) -> list[BackendArray]:
        """
        :param parameters: Arrays in the order of the circuit's parameters, those the
            statistics were computed with.
        :param statistics: The sufficient statistics of every learned layer since the
            last update.
        :return: The updated parameter arrays.
        """
        root = self.layer.copy_with_circuit_parameters(parameters=parameters)
        return [
            parameter
            for layer_maximization, layer_statistics in zip(
                self.layer_maximizations(root=root), statistics
            )
            for parameter in layer_maximization.maximized_parameters(
                statistics=layer_statistics,
                regularization=self.regularization,
                step_size=self.step_size,
            )
        ]


# %% training on mini-batches


@dataclass(kw_only=True)
class TensorizedExpectationMaximizationTraining(
    CircuitParameterTraining[LayerType], ABC
):
    """
    Mini-batch training by a :class:`TensorizedExpectationMaximization`: every step
    accumulates the statistics of its batch, and the parameters are updated after
    :attr:`TensorizedExpectationMaximization.batches_per_update` batches.
    """

    batches_per_update: int | None
    """
    The batches after which the parameters are updated, at the end of every epoch if
    ``None``.
    """

    batches_since_update: int = field(default=0, init=False)
    """
    The batches whose statistics were accumulated since the last update.
    """

    @abstractmethod
    def accumulate_statistics(self, *, batch: tuple[BackendArray, ...]) -> BackendArray:
        """
        Add the sufficient statistics of a batch to those since the last update.

        :param batch: The same rows of every training array.
        :return: The negative average log-likelihood of the rows of the batch, a scalar
            array of the backend.
        """
        raise NotImplementedError

    @abstractmethod
    def accumulate_statistics_and_update(
        self, *, batch: tuple[BackendArray, ...]
    ) -> BackendArray:
        """
        Add the sufficient statistics of a batch and update the parameters from all
        statistics since the last update, starting to accumulate anew.

        :param batch: The same rows of every training array.
        :return: The negative average log-likelihood of the rows of the batch before the
            update, a scalar array of the backend.
        """
        raise NotImplementedError

    @abstractmethod
    def update_parameters(self) -> None:
        """
        Update the parameters from the statistics since the last update and start
        accumulating anew.
        """
        raise NotImplementedError

    def count_batch_toward_update(self) -> bool:
        """
        Count one more batch since the last update.

        :return: Whether the parameters are updated after this batch, from which the
            count starts anew.
        """
        self.batches_since_update += 1
        if self.batches_since_update != self.batches_per_update:
            return False
        self.batches_since_update = 0
        return True

    def step_on_batch(self, *, batch: tuple[BackendArray, ...]) -> BackendArray:
        if self.count_batch_toward_update():
            return self.accumulate_statistics_and_update(batch=batch)
        return self.accumulate_statistics(batch=batch)

    def finish_epoch(self, *, average_loss: float) -> None:
        """
        Update the parameters at the end of the epoch for full-batch expectation
        maximization, see :meth:`MiniBatchTraining.finish_epoch`.
        """
        if self.batches_per_update is None:
            self.update_parameters()
            self.batches_since_update = 0
