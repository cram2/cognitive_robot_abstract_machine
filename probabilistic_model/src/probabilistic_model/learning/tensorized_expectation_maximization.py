"""
Expectation maximization of the parameters of a layered circuit on any array library
that differentiates automatically: the expectation step is the gradient of the total
log-likelihood of a batch, the maximization step the update of every layer.
"""

from __future__ import annotations

# %% imports
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import Any

from probabilistic_model.learning.circuit_parameter_learning import (
    CircuitParameterLearning,
    CircuitParameterTraining,
    LayerType,
)
from probabilistic_model.learning.exceptions import UnsupportedLayerError
from probabilistic_model.learning.expectation_maximization import (
    MaximizationRegularization,
)
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    BackendArray,
    SampleArray,
)
from probabilistic_model.probabilistic_circuit.tensorized.expectation_maximization_layer import (
    ExpectationMaximizationLayer,
    SufficientStatistics,
)
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
class TensorizedExpectationMaximization(
    CircuitParameterLearning[LayerType], MaximizationRegularization, ABC
):
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

    def __post_init__(self) -> None:
        """
        :raises UnsupportedLayerError: If a layer of the circuit has parameters that
            expectation maximization cannot learn.
        """
        super().__post_init__()
        for layer in self.layer.all_layers():
            if layer.parameters and not isinstance(layer, ExpectationMaximizationLayer):
                raise UnsupportedLayerError(layer_type=type(layer))

    @staticmethod
    def learned_layers(*, root: Any) -> list[ExpectationMaximizationLayer]:
        """
        :param root: The root of the circuit or of a copy of it.
        :return: The layers with parameters, in the order of ``all_layers``.
        """
        return [layer for layer in root.all_layers() if layer.parameters]

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
            layer.expected_statistics(parameter_gradients=gradients, node_counts=counts)
            for layer, gradients, counts in zip(
                self.learned_layers(root=root), gradients_by_layer, node_counts
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
            for layer, layer_statistics in zip(
                self.learned_layers(root=root), statistics
            )
            for parameter in layer.maximized_parameters(
                statistics=layer_statistics,
                regularization=self,
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
    def update_parameters(self) -> None:
        """
        Update the parameters from the statistics since the last update and start
        accumulating anew.
        """
        raise NotImplementedError

    def step_on_batch(self, *, batch: tuple[BackendArray, ...]) -> BackendArray:
        loss = self.accumulate_statistics(batch=batch)
        self.batches_since_update += 1
        if self.batches_since_update == self.batches_per_update:
            self.update_parameters()
            self.batches_since_update = 0
        return loss

    def finish_epoch(self, *, average_loss: float) -> None:
        """
        Update the parameters at the end of the epoch for full-batch expectation
        maximization, see :meth:`MiniBatchTraining.finish_epoch`.
        """
        if self.batches_per_update is None:
            self.update_parameters()
            self.batches_since_update = 0
