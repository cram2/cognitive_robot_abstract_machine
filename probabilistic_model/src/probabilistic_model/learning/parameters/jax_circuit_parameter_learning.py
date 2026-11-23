"""
Learning the parameters of a layered circuit with jax: one compiled step per batch,
whose rows are split across the devices of the parameters' platform.
"""

from __future__ import annotations

# %% imports
import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from krrood_compute_resources.jax_memory import JaxDeviceMemory
from krrood_compute_resources.memory import DeviceMemory, HostMemory
from typing_extensions import Any, Callable, Self, TypeVar

from probabilistic_model.learning.parameters.circuit_parameter_learning import (
    CircuitParameterLearning,
    CircuitParameterTraining,
    LayerType,
)

TrainingStateType = TypeVar("TrainingStateType")
"""
The type of the pytree that one compiled step reads and returns.
"""

# %% training state


@dataclass(kw_only=True)
class JaxTrainingState:
    """
    The values every compiled step of a jax training reads and returns; a subclass adds
    what its learning carries from step to step and registers itself as a pytree.
    """

    parameters: list[jax.Array]
    """
    The parameter arrays of the circuit, in the order of its ``circuit_parameters``.
    """

    loss: jax.Array
    """
    The loss of the last batch before its step.
    """


# %% learning


@dataclass(kw_only=True)
class JaxCircuitParameterLearning(CircuitParameterLearning[LayerType], ABC):
    """
    Learns the parameters of a circuit with jax parameters by one compiled step per
    batch on every device of the parameters' platform.
    """

    initial_parameters: list[jax.Array] = field(init=False)
    """
    The parameter arrays of the circuit before training.
    """

    def __post_init__(self) -> None:
        """
        Record the parameters before training.
        """
        super().__post_init__()
        self.initial_parameters = list(self.layer.circuit_parameters)

    @abstractmethod
    def step(
        self, *, state: JaxTrainingState, batch: tuple[jax.Array, ...]
    ) -> JaxTrainingState:
        """
        :param state: The training state before the step.
        :param batch: The same rows of every training array.
        :return: The training state after one step on the batch, with the loss of the
            batch before the step.
        """
        raise NotImplementedError

    @abstractmethod
    def initial_state(self) -> JaxTrainingState:
        """
        :return: The training state before the first step.
        """
        raise NotImplementedError

    def loss_before_any_step(self) -> jax.Array:
        """
        :return: The loss of a state no step has reported a loss to, not a number in
            the floating point type of the parameters.
        """
        return jnp.full((), jnp.nan, jnp.result_type(*self.initial_parameters))

    @property
    def parameter_device(self) -> jax.Device:
        """
        :return: The device the parameters of the circuit are stored on.
        """
        [device] = self.initial_parameters[0].devices()
        return device

    @property
    def training_devices(self) -> list[jax.Device]:
        """
        :return: The devices of the platform of the parameters, which every step
            computes on.
        """
        return jax.devices(self.parameter_device.platform)

    def largest_even_batch_size(self, *, rows: int) -> int:
        """
        A multiple of the number of :attr:`training_devices`, see
        :meth:`CircuitParameterLearning.largest_even_batch_size`.
        """
        number_of_devices = len(self.training_devices)
        if rows < number_of_devices:
            return rows
        return rows - rows % number_of_devices

    def device_memory(self) -> DeviceMemory:
        device = self.parameter_device
        if device.platform == JaxPlatform.CPU:
            return HostMemory()
        return JaxDeviceMemory(device=device)

    def probe_step_bytes(self, *, batch: tuple[jax.Array, ...]) -> int:
        """
        :return: The bytes the step compiled for this batch allocates, as XLA reports
            them: its batch, its temporaries and its new state.
        """
        analysis = (
            jax.jit(self.step)
            .lower(state=self.initial_state(), batch=batch)
            .compile()
            .memory_analysis()
        )
        return (
            sum(argument.nbytes for argument in batch)
            + analysis.temp_size_in_bytes
            + analysis.output_size_in_bytes
            - analysis.alias_size_in_bytes
        )


# %% compiled steps


@dataclass(kw_only=True)
class CompiledStep:
    """
    A step of a jax training, compiled once for batches of every shape and once for
    every number of rows it gathers from the training arrays itself.
    """

    on_batch: Callable[..., JaxTrainingState]
    """
    The step on a batch, called with ``state=`` and ``batch=``.
    """

    on_rows: Callable[..., JaxTrainingState]
    """
    The step on the rows the indices pick out of the training arrays, called with
    ``state=``, ``rows=`` and ``arguments=``.
    """

    @classmethod
    def from_step(cls, *, step: Callable[..., JaxTrainingState]) -> Self:
        """
        :param step: A function of the training state and a batch, such as
            :meth:`JaxCircuitParameterLearning.step`.
        :return: That step compiled.
        """
        return cls(
            on_batch=jax.jit(step),
            on_rows=jax.jit(functools.partial(cls.step_on_gathered_rows, step=step)),
        )

    @staticmethod
    def step_on_gathered_rows(
        *,
        step: Callable[..., JaxTrainingState],
        state: JaxTrainingState,
        rows: jax.Array,
        arguments: tuple[jax.Array, ...],
    ) -> JaxTrainingState:
        """
        :param step: A function of the training state and a batch.
        :param state: The training state before the step.
        :param rows: The indices of the rows of the batch.
        :param arguments: The training arrays, one row per training example each.
        :return: The training state after the step on those rows.
        """
        return step(
            state=state,
            batch=tuple(jnp.take(argument, rows, axis=0) for argument in arguments),
        )


# %% training on mini-batches


@dataclass(kw_only=True)
class JaxCircuitParameterTraining(CircuitParameterTraining[LayerType], ABC):
    """
    Mini-batch training by a :class:`JaxCircuitParameterLearning`, whose state is copied
    to every training device and whose batches are split across them.
    """

    learning: JaxCircuitParameterLearning[LayerType]
    """
    The learning whose compiled steps are taken.
    """

    state: JaxTrainingState = field(init=False)
    """
    The state after the last step.
    """

    batch_split_devices: BatchSplitDevices | None = field(init=False, default=None)
    """
    The :attr:`JaxCircuitParameterLearning.training_devices` that the rows of a batch
    are split across and the state is copied to, ``None`` if there is only one.
    """

    def __post_init__(self) -> None:
        """
        Start from the state before training, copied to every training device.
        """
        self.state = self.learning.initial_state()
        self.batch_split_devices = BatchSplitDevices.from_devices(
            devices=self.learning.training_devices
        )
        if self.batch_split_devices is not None:
            self.state = self.batch_split_devices.replicated(state=self.state)

    def take_compiled_step(
        self, *, step: CompiledStep, batch: tuple[jax.Array, ...]
    ) -> jax.Array:
        """
        :param step: The compiled step to take.
        :param batch: The same rows of every training array.
        :return: The loss of the batch before the step.
        """
        if self.batch_split_devices is not None:
            batch = self.batch_split_devices.placed_batch(batch=batch)
        self.state = step.on_batch(state=self.state, batch=batch)
        return self.state.loss

    def take_compiled_step_on_rows(
        self, *, step: CompiledStep, rows: np.ndarray
    ) -> jax.Array:
        """
        :param step: The compiled step to take.
        :param rows: The indices of the rows of the batch, gathered by the compiled
            step itself on a single training device.
        :return: The loss of the batch before the step.
        """
        if self.batch_split_devices is not None:
            return self.take_compiled_step(
                step=step,
                batch=self.mini_batches.selected_rows(
                    arguments=self.arguments, rows=rows
                ),
            )
        self.state = step.on_rows(state=self.state, rows=rows, arguments=self.arguments)
        return self.state.loss

    def trained_root(self) -> LayerType:
        return self.learning.layer.copy_with_circuit_parameters(
            parameters=jax.device_put(
                self.state.parameters, self.learning.parameter_device
            )
        )


# %% batches split across devices


@dataclass(kw_only=True)
class BatchSplitDevices:
    """
    The devices that the rows of every batch are split across and the training state is
    copied to.
    """

    mesh: Mesh
    """
    The devices, arranged along the axis :attr:`BatchSplitAxis.ROWS`.
    """

    @classmethod
    def from_devices(cls, *, devices: list[jax.Device]) -> Self | None:
        """
        :param devices: The devices every step computes on.
        :return: Those devices, ``None`` if there is only one, which needs no split.
        """
        if len(devices) < 2:
            return None
        return cls(mesh=Mesh(np.array(devices), (BatchSplitAxis.ROWS,)))

    def replicated(self, *, state: TrainingStateType) -> TrainingStateType:
        """
        :param state: A pytree of arrays.
        :return: The pytree with a copy of every array on every device.
        """
        return jax.device_put(state, NamedSharding(self.mesh, PartitionSpec()))

    def placed_batch(self, *, batch: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
        """
        :param batch: The same rows of every training array.
        :return: The batch split by rows across the devices if they split the rows
            evenly, whole on every one of them otherwise.
        """
        rows = batch[0].shape[0]
        partition = (
            PartitionSpec(BatchSplitAxis.ROWS)
            if rows % self.mesh.size == 0
            else PartitionSpec()
        )
        return jax.device_put(batch, NamedSharding(self.mesh, partition))


class BatchSplitAxis(StrEnum):
    """
    The named axes of the arrangement of devices that training splits batches across.
    """

    ROWS = "rows"
    """
    The axis the rows of a batch are split along.
    """


# %% compiler options


class CompilerVariable(StrEnum):
    """
    The environment variables that configure the compilers of a process.
    """

    ACCELERATED_LINEAR_ALGEBRA_OPTIONS = "XLA_FLAGS"
    """
    The options of the XLA compiler.
    """


class CompilerOption(StrEnum):
    """
    The options of the XLA compiler a process is given.
    """

    GRAPHICS_PROCESSOR_AUTOTUNE_LEVEL = "--xla_gpu_autotune_level"
    """
    How much XLA tunes the kernels it compiles for the GPU.
    """

    HOST_PLATFORM_DEVICE_COUNT = "--xla_force_host_platform_device_count"
    """
    The number of devices jax divides the processor cores into.
    """

    def to_flag(self, *, value: int) -> str:
        """
        :param value: The value of this option.
        :return: This option set to ``value``, as one of the options in
            :attr:`CompilerVariable.ACCELERATED_LINEAR_ALGEBRA_OPTIONS`.
        """
        return f"{self}={value}"


# %% device memory


class JaxPlatform(StrEnum):
    """
    The platforms of jax devices that are told apart.
    """

    CPU = "cpu"
