"""
Training the parameters of a tensorized circuit with torch: the parameter arrays are
leaf tensors of a torch optimizer and the loss is differentiated by autograd.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import torch
from krrood_compute_hardware.memory import HostMemory, PeakTrackingMemory
from krrood_compute_hardware.torch_memory import TorchDeviceMemory

from probabilistic_model.learning.gradient_descent import (
    GradientDescent,
    GradientDescentTraining,
    LayerType,
)
from probabilistic_model.learning.exceptions import UnmeasurableHostStepMemoryError
from probabilistic_model.learning.training_schedule import (
    LearningRatePlateau,
    MiniBatches,
)


# %% gradient descent


@dataclass
class TorchGradientDescent(GradientDescent[LayerType]):
    """
    Gradient descent with torch, one optimizer step per batch, after which the
    parameters are projected onto valid parameters with
    ``projected_circuit_parameters``.
    """

    learning_rate: float = 1e-3
    """
    The learning rate of the optimizer.
    """

    optimizer_type: type[torch.optim.Optimizer] = torch.optim.Adam
    """
    The torch optimizer to use.
    """

    parameters: list[torch.Tensor] = field(init=False)
    """
    The leaf tensors the optimizer updates, copies of the parameters of the circuit.
    """

    def __post_init__(self) -> None:
        """
        Copy the parameters of the circuit into the trained leaf tensors.
        """
        super().__post_init__()
        self.parameters = [
            parameter.detach().clone().requires_grad_(True)
            for parameter in self.layer.circuit_parameters
        ]

    def start_training(
        self,
        *,
        arguments: tuple[torch.Tensor, ...],
        mini_batches: MiniBatches,
        learning_rate_plateau: LearningRatePlateau | None = None,
    ) -> TorchGradientDescentTraining[LayerType]:
        return TorchGradientDescentTraining(
            arguments=arguments,
            mini_batches=mini_batches,
            gradient_descent=self,
            optimizer=self.optimizer_type(self.parameters, lr=self.learning_rate),
            learning_rate_plateau=learning_rate_plateau,
        )

    def device_memory(self) -> PeakTrackingMemory:
        device = self.parameters[0].device
        if device.type == TorchDeviceType.CUDA:
            return TorchDeviceMemory(device=device)
        return HostMemory()

    def probe_step_bytes(self, *, batch: tuple[torch.Tensor, ...]) -> int:
        """
        :return: The most bytes the loss and its gradient on the batch had in use
            beyond what was in use before; the gradient is discarded afterwards.
        :raises UnmeasurableHostStepMemoryError: If the parameters are on the CPU.
        """
        device = self.parameters[0].device
        if device.type != TorchDeviceType.CUDA:
            raise UnmeasurableHostStepMemoryError()
        memory = TorchDeviceMemory(device=device)
        memory.reset_peak()
        before = memory.in_use_bytes()
        self.loss.evaluate(
            self.layer.copy_with_circuit_parameters(self.parameters), *batch
        ).backward()
        peak = memory.peak_bytes()
        for parameter in self.parameters:
            parameter.grad = None
        return peak - before

    def step(
        self, *, optimizer: torch.optim.Optimizer, arguments: tuple[torch.Tensor, ...]
    ) -> float:
        """
        Update the parameters by one optimizer step and project them onto valid
        parameters.

        :param optimizer: The optimizer of :attr:`parameters`.
        :param arguments: The tensors the :attr:`loss` takes after the circuit.
        :return: The loss of the parameters before the step.
        """
        optimizer.zero_grad()
        loss = self.loss.evaluate(
            self.layer.copy_with_circuit_parameters(self.parameters), *arguments
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for parameter, zero_probability in zip(
                self.parameters, self.zero_probability_entries
            ):
                parameter.masked_fill_(zero_probability, -torch.inf)
            projected = self.layer.projected_circuit_parameters(self.parameters)
            for parameter, valid in zip(self.parameters, projected):
                parameter.copy_(valid)
        return loss.item()


# %% training on mini-batches


@dataclass(kw_only=True)
class TorchGradientDescentTraining(GradientDescentTraining[LayerType]):
    """
    Mini-batch training by :class:`TorchGradientDescent`, one optimizer step per batch.
    """

    gradient_descent: TorchGradientDescent[LayerType]
    """
    The gradient descent whose parameters are updated.
    """

    optimizer: torch.optim.Optimizer
    """
    The optimizer of the parameters of :attr:`gradient_descent`.
    """

    def step_on_batch(self, *, batch: tuple[torch.Tensor, ...]) -> float:
        return self.gradient_descent.step(optimizer=self.optimizer, arguments=batch)

    def scale_learning_rate(self, *, scale: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = self.gradient_descent.learning_rate * scale

    def trained_root(self) -> LayerType:
        return self.gradient_descent.layer.copy_with_circuit_parameters(
            [parameter.detach() for parameter in self.gradient_descent.parameters]
        )


# %% device memory


class TorchDeviceType(StrEnum):
    """
    The types of torch devices that are told apart.
    """

    CUDA = "cuda"
