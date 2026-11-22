"""
Training the parameters of a tensorized circuit with torch: the parameter arrays are
leaf tensors of a torch optimizer and the loss is differentiated by autograd.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import torch
from krrood_compute_resources.memory import HostMemory, PeakTrackingMemory
from krrood_compute_resources.torch_memory import TorchDeviceMemory

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

    records_cuda_graphs: bool = field(default=False, kw_only=True)
    """
    Whether training on a CUDA device records the step on batches of every shape as a
    CUDA graph and replays it.

    The optimizer has to accept ``capturable``, and the numeric queries of the circuit
    must not copy from the host.
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
            optimizer=(
                self.optimizer_type(
                    self.parameters,
                    lr=torch.tensor(
                        self.learning_rate,
                        dtype=self.parameters[0].dtype,
                        device=self.parameters[0].device,
                    ),
                    capturable=True,
                )
                if self.replays_recorded_steps
                else self.optimizer_type(self.parameters, lr=self.learning_rate)
            ),
            learning_rate_plateau=learning_rate_plateau,
        )

    @property
    def replays_recorded_steps(self) -> bool:
        """
        :return: Whether steps are recorded as CUDA graphs and replayed, which needs
            :attr:`records_cuda_graphs` and parameters on a CUDA device.
        """
        return (
            self.records_cuda_graphs
            and self.parameters[0].device.type
            == TorchDeviceType.COMPUTE_UNIFIED_DEVICE_ARCHITECTURE
        )

    def device_memory(self) -> PeakTrackingMemory:
        device = self.parameters[0].device
        if device.type == TorchDeviceType.COMPUTE_UNIFIED_DEVICE_ARCHITECTURE:
            return TorchDeviceMemory(device=device)
        return HostMemory()

    def probe_step_bytes(self, *, batch: tuple[torch.Tensor, ...]) -> int:
        """
        :return: The most bytes the loss and its gradient on the batch had in use
            beyond what was in use before; the gradient is discarded afterwards.
        :raises UnmeasurableHostStepMemoryError: If the parameters are on the CPU.
        """
        device = self.parameters[0].device
        if device.type != TorchDeviceType.COMPUTE_UNIFIED_DEVICE_ARCHITECTURE:
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
    ) -> torch.Tensor:
        """
        Update the parameters by one optimizer step and project them onto valid
        parameters.

        :param optimizer: The optimizer of :attr:`parameters`.
        :param arguments: The tensors the :attr:`loss` takes after the circuit.
        :return: The loss of the parameters before the step, a scalar tensor without a
            gradient on the device of the parameters.
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
        return loss.detach()


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

    recorded_steps: dict[BatchShape, RecordedTrainingStep] = field(
        default_factory=dict, init=False
    )
    """
    The step on batches of every shape, if
    :attr:`TorchGradientDescent.replays_recorded_steps`, by their :class:`BatchShape`.
    """

    def step_on_batch(self, *, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if not self.gradient_descent.replays_recorded_steps:
            return self.gradient_descent.step(optimizer=self.optimizer, arguments=batch)
        shape = BatchShape.from_batch(batch=batch)
        if shape not in self.recorded_steps:
            self.recorded_steps[shape] = RecordedTrainingStep()
        return self.recorded_steps[shape].take(
            gradient_descent=self.gradient_descent,
            optimizer=self.optimizer,
            batch=batch,
        )

    def scale_learning_rate(self, *, scale: float) -> None:
        learning_rate = self.gradient_descent.learning_rate * scale
        for group in self.optimizer.param_groups:
            if isinstance(group["lr"], torch.Tensor):
                # a recorded step reads the learning rate from this tensor
                group["lr"].fill_(learning_rate)
            else:
                group["lr"] = learning_rate

    def trained_root(self) -> LayerType:
        return self.gradient_descent.layer.copy_with_circuit_parameters(
            [parameter.detach() for parameter in self.gradient_descent.parameters]
        )


# %% steps recorded as CUDA graphs


@dataclass(kw_only=True, frozen=True)
class BatchShape:
    """
    The shapes of the arrays of a batch, which a recorded step only takes batches of.
    """

    array_shapes: tuple[torch.Size, ...]
    """
    The shape of every array of the batch, in the order of the batch.
    """

    @classmethod
    def from_batch(cls, *, batch: tuple[torch.Tensor, ...]) -> BatchShape:
        """
        :param batch: The same rows of every training array.
        :return: The shape of ``batch``.
        """
        return cls(array_shapes=tuple(array.shape for array in batch))


@dataclass(kw_only=True)
class RecordedTrainingStep:
    """
    The training step on batches of one shape, taken operation by operation for
    :attr:`warm_up_steps` steps, then recorded as a CUDA graph and replayed.
    """

    warm_up_steps: int = 3
    """
    The steps taken operation by operation before the step is recorded, which lets torch
    and the optimizer create their lazily created state.
    """

    steps_taken: int = 0
    """
    The steps taken operation by operation so far.
    """

    graph: torch.cuda.CUDAGraph | None = None
    """
    The recorded step, ``None`` before it is recorded.
    """

    batch: tuple[torch.Tensor, ...] = ()
    """
    The arrays the recorded step reads its batch from.
    """

    loss: torch.Tensor | None = None
    """
    The tensor the recorded step writes the loss of its batch to.
    """

    def take(
        self,
        *,
        gradient_descent: TorchGradientDescent,
        optimizer: torch.optim.Optimizer,
        batch: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """
        :param gradient_descent: The gradient descent whose step is taken.
        :param optimizer: The capturable optimizer of its parameters.
        :param batch: The same rows of every training array, of the shape of this step.
        :return: The loss of the batch before the step.
        """
        if self.steps_taken < self.warm_up_steps:
            self.steps_taken += 1
            return self.take_on_side_stream(
                gradient_descent=gradient_descent, optimizer=optimizer, batch=batch
            )
        if self.graph is None:
            self.record(
                gradient_descent=gradient_descent, optimizer=optimizer, batch=batch
            )
        for recorded, array in zip(self.batch, batch):
            recorded.copy_(array)
        self.graph.replay()
        return self.loss.clone()

    @staticmethod
    def take_on_side_stream(
        *,
        gradient_descent: TorchGradientDescent,
        optimizer: torch.optim.Optimizer,
        batch: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """
        Take the step operation by operation on a stream of its own, as recording a CUDA
        graph needs of the steps before it.

        :return: The loss of the batch before the step.
        """
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            loss = gradient_descent.step(optimizer=optimizer, arguments=batch)
        torch.cuda.current_stream().wait_stream(side_stream)
        return loss

    def record(
        self,
        *,
        gradient_descent: TorchGradientDescent,
        optimizer: torch.optim.Optimizer,
        batch: tuple[torch.Tensor, ...],
    ) -> None:
        """
        Record the step on copies of the arrays of a batch as :attr:`graph`, without
        taking it.
        """
        self.batch = tuple(array.clone() for array in batch)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.loss = gradient_descent.step(optimizer=optimizer, arguments=self.batch)


# %% device memory


class TorchDeviceType(StrEnum):
    """
    The types of torch devices that are told apart.
    """

    COMPUTE_UNIFIED_DEVICE_ARCHITECTURE = "cuda"
