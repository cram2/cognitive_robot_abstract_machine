"""
Learning the parameters of a layered circuit with jax: one compiled step per batch,
whose rows are split across the devices of the parameters' platform.
"""

from __future__ import annotations

# %% imports
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from krrood.patterns.called_positionally import called_positionally
from krrood_compute_resources.jax_memory import JaxDeviceMemory
from krrood_compute_resources.memory import DeviceMemory, HostMemory
from typing_extensions import Any, Self, TypeVar

from probabilistic_model.learning.circuit_parameter_learning import (
    CircuitParameterLearning,
    LayerType,
)

TrainingStateType = TypeVar("TrainingStateType")
"""
The type of the pytree that one compiled step reads and returns.
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
    @called_positionally
    def step(self, state: Any, *batch: jax.Array) -> Any:
        """
        :param state: The training state before the step.
        :param batch: The same rows of every training array.
        :return: The training state after one step on the batch, with the loss of the
            batch before the step.
        """
        raise NotImplementedError

    @abstractmethod
    def initial_state(self) -> Any:
        """
        :return: The training state before the first step.
        """
        raise NotImplementedError

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
            jax.jit(self.step).lower(self.initial_state(), *batch).compile()
        ).memory_analysis()
        return (
            sum(argument.nbytes for argument in batch)
            + analysis.temp_size_in_bytes
            + analysis.output_size_in_bytes
            - analysis.alias_size_in_bytes
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


# %% device memory


class JaxPlatform(StrEnum):
    """
    The platforms of jax devices that are told apart.
    """

    CPU = "cpu"
