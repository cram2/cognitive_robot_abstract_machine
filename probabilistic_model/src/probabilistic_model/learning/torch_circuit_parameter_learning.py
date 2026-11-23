"""Learning the parameters of a layered circuit with torch: the parameter
arrays are leaf tensors that every step updates in place, and a step on a CUDA
device may be recorded as a CUDA graph and replayed."""

from __future__ import annotations

# %% imports
import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from types import MethodType, TracebackType

import torch
from krrood_compute_resources.memory import HostMemory, PeakTrackingMemory
from krrood_compute_resources.torch_memory import TorchDeviceMemory
from typing_extensions import Callable

from probabilistic_model.learning.circuit_parameter_learning import (
    CircuitParameterLearning,
    CircuitParameterTraining,
    LayerType,
)
from probabilistic_model.learning.exceptions import UnmeasurableHostStepMemoryError

# %% learning


@dataclass(kw_only=True)
class TorchCircuitParameterLearning(CircuitParameterLearning[LayerType], ABC):
    """Learns the parameters of a circuit with torch parameters, updating
    copies of them in place."""

    records_cuda_graphs: bool = False
    """Whether training on a CUDA device records the step on batches of every
    shape as a CUDA graph and replays it.

    A circuit with a layer that does not train without host copies takes
    every step one by one.
    """

    parameters: list[torch.Tensor] = field(init=False)
    """The leaf tensors that training updates, copies of the parameters of the
    circuit."""

    def __post_init__(self) -> None:
        """Copy the parameters of the circuit into the trained leaf tensors."""
        super().__post_init__()
        self.parameters = [
            parameter.detach().clone().requires_grad_(True)
            for parameter in self.layer.circuit_parameters
        ]

    @property
    def replays_recorded_steps(self) -> bool:
        """
        :return: Whether steps are recorded as CUDA graphs and replayed, which needs
            :attr:`records_cuda_graphs`, parameters on a CUDA device and a circuit whose
            every layer trains without host copies.
        """
        return (
            self.records_cuda_graphs
            and self.parameters[0].device.type
            == TorchDeviceType.COMPUTE_UNIFIED_DEVICE_ARCHITECTURE
            and all(
                layer.trains_without_host_copies for layer in self.layer.all_layers()
            )
        )

    def device_memory(self) -> PeakTrackingMemory:
        device = self.parameters[0].device
        if device.type == TorchDeviceType.COMPUTE_UNIFIED_DEVICE_ARCHITECTURE:
            return TorchDeviceMemory(device=device)
        return HostMemory()

    @abstractmethod
    def compute_step_without_update(self, *, batch: tuple[torch.Tensor, ...]) -> None:
        """
        Compute everything a step on the batch computes, leaving the parameters and the
        training state unchanged.

        :param batch: The same rows of every training array.
        """
        raise NotImplementedError

    def probe_step_bytes(self, *, batch: tuple[torch.Tensor, ...]) -> int:
        """
        :return: The most bytes :meth:`compute_step_without_update` had in use beyond
            what was in use before.
        :raises UnmeasurableHostStepMemoryError: If the parameters are on the CPU.
        """
        device = self.parameters[0].device
        if device.type != TorchDeviceType.COMPUTE_UNIFIED_DEVICE_ARCHITECTURE:
            raise UnmeasurableHostStepMemoryError()
        memory = TorchDeviceMemory(device=device)
        memory.reset_peak()
        before = memory.in_use_bytes()
        self.compute_step_without_update(batch=batch)
        return memory.peak_bytes() - before


# %% training on mini-batches


@dataclass(kw_only=True)
class TorchCircuitParameterTraining(CircuitParameterTraining[LayerType], ABC):
    """Mini-batch training by a :class:`TorchCircuitParameterLearning`, its
    step on batches of every shape replayed from a CUDA graph if it records
    them."""

    learning: TorchCircuitParameterLearning[LayerType]
    """The learning whose parameters are updated in place."""

    replays_recorded_steps: bool = False
    """Whether steps are recorded as CUDA graphs and replayed, see
    :attr:`TorchCircuitParameterLearning.replays_recorded_steps`."""

    recorded_steps: dict[RecordedStepKey, RecordedTrainingStep] = field(
        default_factory=dict, init=False
    )
    """Every step on batches of every shape, if
    :attr:`replays_recorded_steps`."""

    def take_step_or_replay(
        self,
        *,
        step: MethodType,
        batch: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Take a step on a batch, replayed from its recording for batches of
        its shape if :attr:`replays_recorded_steps`.

        :param step: The step operation by operation, a method of this
            training called with ``batch=`` and returning the loss of
            the batch before the step, a scalar tensor without a
            gradient on the device of the parameters.
        :param batch: The same rows of every training array.
        :return: The loss of the batch before the step.
        """
        if not self.replays_recorded_steps:
            return step(batch=batch)
        # Keyed by the unbound function, so that no reference cycle keeps a finished
        # training and its recorded graphs alive until the next collection.
        key = RecordedStepKey(
            step_function=step.__func__, batch_shape=BatchShape.from_batch(batch=batch)
        )
        if key not in self.recorded_steps:
            self.recorded_steps[key] = RecordedTrainingStep()
        return self.recorded_steps[key].take(step=step, batch=batch)

    def trained_root(self) -> LayerType:
        return self.learning.layer.copy_with_circuit_parameters(
            parameters=[parameter.detach() for parameter in self.learning.parameters]
        )


# %% steps recorded as CUDA graphs


@dataclass(kw_only=True, frozen=True)
class BatchShape:
    """The shapes of the arrays of a batch, which a recorded step only takes
    batches of."""

    array_shapes: tuple[torch.Size, ...]
    """The shape of every array of the batch, in the order of the batch."""

    @classmethod
    def from_batch(cls, *, batch: tuple[torch.Tensor, ...]) -> BatchShape:
        """
        :param batch: The same rows of every training array.
        :return: The shape of ``batch``.
        """
        return cls(array_shapes=tuple(array.shape for array in batch))


@dataclass(kw_only=True, frozen=True)
class RecordedStepKey:
    """Which step, on batches of which shape, a recording replays."""

    step_function: Callable[..., torch.Tensor]
    """The function of the method that takes the step operation by operation,
    unbound from the training."""

    batch_shape: BatchShape
    """The shape of the batches it takes."""


@dataclass(kw_only=True)
class RecordedTrainingStep:
    """The training step on batches of one shape, taken operation by operation
    for :attr:`warm_up_steps` steps, then recorded as a CUDA graph and
    replayed."""

    warm_up_steps: int = 3
    """The steps taken operation by operation before the step is recorded,
    which lets torch and the optimizer create their lazily created state."""

    steps_taken: int = 0
    """The steps taken operation by operation so far."""

    graph: torch.cuda.CUDAGraph | None = None
    """The recorded step, ``None`` before it is recorded."""

    batch: tuple[torch.Tensor, ...] = ()
    """The arrays the recorded step reads its batch from."""

    loss: torch.Tensor | None = None
    """The tensor the recorded step writes the loss of its batch to."""

    def take(
        self,
        *,
        step: Callable[..., torch.Tensor],
        batch: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """
        :param step: The step operation by operation, called with ``batch=`` and
            returning the loss of the batch before the step.
        :param batch: The same rows of every training array, of the shape of this step.
        :return: The loss of the batch before the step.
        """
        if self.steps_taken < self.warm_up_steps:
            self.steps_taken += 1
            return self.take_on_side_stream(step=step, batch=batch)
        if self.graph is None:
            self.record(step=step, batch=batch)
        for recorded, array in zip(self.batch, batch):
            recorded.copy_(array)
        self.graph.replay()
        return self.loss.clone()

    @staticmethod
    def take_on_side_stream(
        *,
        step: Callable[..., torch.Tensor],
        batch: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Take the step operation by operation on a stream of its own, as
        recording a CUDA graph needs of the steps before it.

        :return: The loss of the batch before the step.
        """
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            loss = step(batch=batch)
        torch.cuda.current_stream().wait_stream(side_stream)
        return loss

    def record(
        self,
        *,
        step: Callable[..., torch.Tensor],
        batch: tuple[torch.Tensor, ...],
    ) -> None:
        """Record the step on copies of the arrays of a batch as :attr:`graph`,
        without taking it."""
        self.batch = tuple(array.clone() for array in batch)
        self.graph = torch.cuda.CUDAGraph()
        # A collection that frees another recorded graph during the recording would
        # invalidate it.
        with PausedGarbageCollection(), torch.cuda.graph(self.graph):
            self.loss = step(batch=self.batch)


@dataclass(kw_only=True)
class PausedGarbageCollection:
    """Keeps the garbage collector from running automatically inside the
    context, and lets it run afterwards if it did before."""

    was_enabled: bool = field(default=False, init=False)
    """Whether the garbage collector ran automatically before the context was
    entered."""

    def __enter__(self) -> PausedGarbageCollection:
        """Stop the garbage collector from running automatically."""
        self.was_enabled = gc.isenabled()
        gc.disable()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Let the garbage collector run automatically again if it did
        before."""
        if self.was_enabled:
            gc.enable()


# %% device memory


class TorchDeviceType(StrEnum):
    """The types of torch devices that are told apart."""

    COMPUTE_UNIFIED_DEVICE_ARCHITECTURE = "cuda"
