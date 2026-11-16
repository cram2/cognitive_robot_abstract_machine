"""
Training the parameters of a tensorized circuit with torch: the parameter arrays are
leaf tensors of a torch optimizer and the loss is differentiated by autograd.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import torch
import tqdm
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from typing_extensions import Generic

from probabilistic_model.learning.gradient_descent import (
    LayerType,
    Loss,
    TrainingRun,
)
from probabilistic_model.learning.exceptions import UnmeasurableHostStepMemoryError
from probabilistic_model.learning.device_memory import (
    HostMemory,
    PeakTrackingMemory,
    ProbeMeasurement,
    StepMemory,
)
from probabilistic_model.learning.training_schedule import (
    Convergence,
    MiniBatches,
    MiniBatchTraining,
)


# %% gradient descent


@dataclass
class TorchGradientDescent(Generic[LayerType], SubClassSafeGeneric):
    """
    Minimizes a loss of the circuit rooted at a tensorized layer whose parameters are
    torch tensors, one optimizer step per epoch.

    After every step the parameters are projected onto valid parameters with
    ``projected_circuit_parameters``.
    """

    layer: LayerType
    """
    The root of the circuit to train, which is left unchanged.
    """

    loss: Loss[LayerType]
    """
    The loss to minimize.
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

    zero_probability_entries: list[torch.Tensor] = field(init=False)
    """
    Where every parameter is ``-inf`` before training, a log-probability or log-weight
    of zero probability, which training keeps.
    """

    def __post_init__(self) -> None:
        """
        Copy the parameters of the circuit into the trained leaf tensors and record
        their :attr:`zero_probability_entries`.
        """
        self.parameters = [
            parameter.detach().clone().requires_grad_(True)
            for parameter in self.layer.circuit_parameters
        ]
        self.zero_probability_entries = [
            torch.isneginf(parameter) for parameter in self.parameters
        ]

    def minimize(self, epochs: int, *arguments: torch.Tensor) -> LayerType:
        """
        :param epochs: The number of epochs.
        :param arguments: The tensors the :attr:`loss` takes after the circuit.
        :return: The root of the trained circuit.
        """
        optimizer = self.optimizer_type(self.parameters, lr=self.learning_rate)
        progress_bar = tqdm.tqdm(range(epochs), desc="Fitting")
        for _ in progress_bar:
            loss = self.step(optimizer=optimizer, arguments=arguments)
            progress_bar.set_postfix_str(f"Loss: {loss}")
        return self.trained_root()

    def minimize_until_converged(
        self,
        *arguments: torch.Tensor,
        convergence: Convergence | None = None,
        mini_batches: MiniBatches | None = None,
    ) -> TrainingRun[LayerType]:
        """
        Minimize the :attr:`loss` on shuffled mini-batches, one optimizer step per
        batch, until the average loss of an epoch converges.

        :param arguments: The tensors the :attr:`loss` takes after the circuit, one row
            per training example each.
        :param convergence: When training stops, :class:`Convergence` with its defaults
            if ``None``.
        :param mini_batches: How the rows are split into the batches of every epoch,
            :class:`MiniBatches` with its defaults, one batch of all rows, if ``None``.
        :return: The trained circuit and the losses of its epochs.
        """
        training = TorchGradientDescentTraining(
            arguments=arguments,
            mini_batches=MiniBatches() if mini_batches is None else mini_batches,
            gradient_descent=self,
            optimizer=self.optimizer_type(self.parameters, lr=self.learning_rate),
        )
        history = (
            Convergence() if convergence is None else convergence
        ).history_until_converged(training=training)
        return TrainingRun(trained_root=self.trained_root(), history=history)

    def device_memory(self) -> PeakTrackingMemory:
        """
        :return: The memory of the device the parameters of the circuit are stored on.
        """
        device = self.parameters[0].device
        if device.type == TorchDeviceType.CUDA:
            return TorchDeviceMemory(device=device)
        return HostMemory()

    def step_memory(self, *arguments: torch.Tensor, probe_rows: int = 64) -> StepMemory:
        """
        Measure the memory of the loss and its gradient on two small batches, leaving
        the parameters unchanged.

        :param arguments: The tensors the :attr:`loss` takes after the circuit, one row
            per training example each.
        :param probe_rows: The rows of the smaller of the two batches; the larger has
            twice as many.
        :return: The memory one step allocates.
        :raises UnmeasurableHostStepMemoryError: If the parameters are on the CPU.
        """
        device = self.parameters[0].device
        if device.type != TorchDeviceType.CUDA:
            raise UnmeasurableHostStepMemoryError()
        memory = TorchDeviceMemory(device=device)
        smaller_rows = max(1, min(probe_rows, arguments[0].shape[0] // 2))
        return StepMemory.from_two_probes(
            smaller=ProbeMeasurement(
                rows=smaller_rows,
                step_bytes=self.probe_step_bytes(
                    batch=tuple(argument[:smaller_rows] for argument in arguments),
                    memory=memory,
                ),
            ),
            larger=ProbeMeasurement(
                rows=2 * smaller_rows,
                step_bytes=self.probe_step_bytes(
                    batch=tuple(argument[: 2 * smaller_rows] for argument in arguments),
                    memory=memory,
                ),
            ),
        )

    def probe_step_bytes(
        self, *, batch: tuple[torch.Tensor, ...], memory: TorchDeviceMemory
    ) -> int:
        """
        :param batch: The same rows of every training array.
        :param memory: The memory of the CUDA device the parameters are stored on.
        :return: The most bytes the loss and its gradient on the batch had in use beyond
            what was in use before; the gradient is discarded afterwards.
        """
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

    def trained_root(self) -> LayerType:
        """
        :return: The root of a copy of the circuit with the current parameters.
        """
        return self.layer.copy_with_circuit_parameters(
            [parameter.detach() for parameter in self.parameters]
        )


# %% training on mini-batches


@dataclass(kw_only=True)
class TorchGradientDescentTraining(MiniBatchTraining):
    """
    Mini-batch training by :class:`TorchGradientDescent`, one optimizer step per batch.
    """

    gradient_descent: TorchGradientDescent
    """
    The gradient descent whose parameters are updated.
    """

    optimizer: torch.optim.Optimizer
    """
    The optimizer of the parameters of :attr:`gradient_descent`.
    """

    def step_on_batch(self, *, batch: tuple[torch.Tensor, ...]) -> float:
        return self.gradient_descent.step(optimizer=self.optimizer, arguments=batch)


# %% device memory


class TorchDeviceType(StrEnum):
    """
    The types of torch devices that are told apart.
    """

    CUDA = "cuda"


@dataclass(kw_only=True)
class TorchDeviceMemory(PeakTrackingMemory):
    """
    The memory of a CUDA device torch stores tensors on.
    """

    device: torch.device
    """
    The CUDA device.
    """

    def free_bytes(self) -> int:
        free, _ = torch.cuda.mem_get_info(self.device)
        return free

    def in_use_bytes(self) -> int:
        return torch.cuda.memory_allocated(self.device)

    def peak_bytes(self) -> int:
        return torch.cuda.max_memory_allocated(self.device)

    def reset_peak(self) -> None:
        torch.cuda.reset_peak_memory_stats(self.device)
