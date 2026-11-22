"""
Expectation maximization of the parameters of a layered circuit with torch: autograd
gives the statistics of a batch, which accumulate in place so that a step can be
recorded as a CUDA graph.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass, field

import torch
from krrood_compute_resources.torch_memory import TorchDeviceMemory

from probabilistic_model.learning.circuit_parameter_learning import LayerType
from probabilistic_model.learning.exceptions import UnmeasurableHostStepMemoryError
from probabilistic_model.learning.tensorized_expectation_maximization import (
    TensorizedExpectationMaximization,
    TensorizedExpectationMaximizationTraining,
)
from probabilistic_model.learning.torch_circuit_parameter_learning import (
    TorchCircuitParameterLearning,
    TorchCircuitParameterTraining,
    TorchDeviceType,
)
from probabilistic_model.learning.training_schedule import (
    LearningRatePlateau,
    MiniBatches,
)
from probabilistic_model.probabilistic_circuit.tensorized.expectation_maximization_parameters import (
    SufficientStatistics,
)

# %% statistics of a batch


@dataclass(kw_only=True)
class BatchStatistics:
    """
    What one autograd pass learns about the rows of a batch.
    """

    total_log_likelihood: torch.Tensor
    """
    The sum of the log-likelihoods of the rows, without a gradient.
    """

    statistics: list[SufficientStatistics]
    """
    The sufficient statistics of every learned layer on the rows.
    """


# %% expectation maximization


@dataclass(kw_only=True)
class TorchExpectationMaximization(
    TensorizedExpectationMaximization[LayerType],
    TorchCircuitParameterLearning[LayerType],
):
    """
    Expectation maximization with torch, one autograd pass per batch.
    """

    def zero_node_counts(self) -> list[torch.Tensor]:
        """
        :return: A zero for every node of every learned layer, on the device and in the
            floating point type of the parameters, differentiable.
        """
        return [
            torch.zeros(
                layer.number_of_nodes,
                dtype=self.parameters[0].dtype,
                device=self.parameters[0].device,
                requires_grad=True,
            )
            for layer in self.learned_layers(root=self.layer)
        ]

    def batch_statistics(self, *, data: torch.Tensor) -> BatchStatistics:
        """
        :param data: The rows of a batch.
        :return: What the circuit with the current parameters learns about the rows.
        """
        node_offsets = self.zero_node_counts()
        total = self.total_log_likelihood(
            parameters=self.parameters, node_offsets=node_offsets, data=data
        )
        gradients = torch.autograd.grad(total, self.parameters + node_offsets)
        with torch.no_grad():
            return BatchStatistics(
                total_log_likelihood=total.detach(),
                statistics=self.statistics_from_gradients(
                    parameters=[parameter.detach() for parameter in self.parameters],
                    parameter_gradients=list(gradients[: len(self.parameters)]),
                    node_counts=list(gradients[len(self.parameters) :]),
                ),
            )

    def probe_step_bytes(self, *, batch: tuple[torch.Tensor, ...]) -> int:
        """
        :return: The most bytes the statistics of the batch had in use beyond what was
            in use before.
        :raises UnmeasurableHostStepMemoryError: If the parameters are on the CPU.
        """
        device = self.parameters[0].device
        if device.type != TorchDeviceType.COMPUTE_UNIFIED_DEVICE_ARCHITECTURE:
            raise UnmeasurableHostStepMemoryError()
        memory = TorchDeviceMemory(device=device)
        memory.reset_peak()
        before = memory.in_use_bytes()
        [data] = batch
        self.batch_statistics(data=data)
        return memory.peak_bytes() - before

    def start_training(
        self,
        *,
        arguments: tuple[torch.Tensor, ...],
        mini_batches: MiniBatches,
        learning_rate_plateau: LearningRatePlateau | None = None,
    ) -> TorchExpectationMaximizationTraining[LayerType]:
        return TorchExpectationMaximizationTraining(
            arguments=arguments,
            mini_batches=mini_batches,
            batches_per_update=self.batches_per_update,
            replays_recorded_steps=self.replays_recorded_steps,
            expectation_maximization=self,
        )


# %% training on mini-batches


@dataclass(kw_only=True)
class TorchExpectationMaximizationTraining(
    TensorizedExpectationMaximizationTraining[LayerType],
    TorchCircuitParameterTraining[LayerType],
):
    """
    Mini-batch training by :class:`TorchExpectationMaximization`, its statistics and
    parameters updated in place.
    """

    expectation_maximization: TorchExpectationMaximization[LayerType]
    """
    The expectation maximization whose parameters are updated.
    """

    statistics: list[SufficientStatistics] = field(init=False)
    """
    The sufficient statistics of every learned layer since the last update.
    """

    def __post_init__(self) -> None:
        """
        Start without statistics.
        """
        with torch.no_grad():
            parameters = [
                parameter.detach()
                for parameter in self.expectation_maximization.parameters
            ]
            self.statistics = self.expectation_maximization.statistics_from_gradients(
                parameters=parameters,
                parameter_gradients=[
                    torch.zeros_like(parameter) for parameter in parameters
                ],
                node_counts=[
                    counts.detach()
                    for counts in self.expectation_maximization.zero_node_counts()
                ],
            )

    def take_step(self, *, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        [data] = batch
        batch_statistics = self.expectation_maximization.batch_statistics(data=data)
        with torch.no_grad():
            for accumulated, added in zip(self.statistics, batch_statistics.statistics):
                for accumulated_array, added_array in zip(
                    accumulated.arrays, added.arrays
                ):
                    accumulated_array.add_(added_array)
        return -batch_statistics.total_log_likelihood / data.shape[0]

    def accumulate_statistics(self, *, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self.take_step_or_replay(batch=batch)

    def update_parameters(self) -> None:
        parameters = self.expectation_maximization.parameters
        with torch.no_grad():
            maximized = self.expectation_maximization.maximized_circuit_parameters(
                parameters=[parameter.detach() for parameter in parameters],
                statistics=self.statistics,
            )
            for parameter, updated in zip(parameters, maximized):
                parameter.copy_(updated)
            for statistics in self.statistics:
                for array in statistics.arrays:
                    array.zero_()

    def trained_root(self) -> LayerType:
        return self.expectation_maximization.layer.copy_with_circuit_parameters(
            parameters=[
                parameter.detach()
                for parameter in self.expectation_maximization.parameters
            ]
        )
