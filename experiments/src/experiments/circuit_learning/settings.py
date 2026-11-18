"""
The settings a circuit is trained and queried in: the array library, the device, the
floating point type and the storage of the edges.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass
from enum import StrEnum
from types import ModuleType

import array_api_compat.torch as torch_namespace
import jax
import jax.numpy as jnp
import numpy as np
import torch
from krrood.entity_query_language.backends import (
    EntityQueryLanguageGenerativeBackend,
)
from krrood.entity_query_language.factories import a, variable_from
from typing_extensions import Any, Self

from probabilistic_model.learning.device_memory import DeviceMemory
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.edge_matrix import (
    DenseEdgeMatrix,
    EdgeMatrix,
    SparseEdgeMatrix,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)

# %% values


class ArrayLibrary(StrEnum):
    """
    The array library a circuit computes with.
    """

    NUMPY = "numpy"
    JAX = "jax"
    TORCH = "torch"


class Device(StrEnum):
    """
    The device a circuit computes on.
    """

    CPU = "cpu"
    GPU = "gpu"


class FloatingPointType(StrEnum):
    """
    The floating point type of a circuit's parameters and computations.
    """

    FLOAT32 = "float32"
    FLOAT64 = "float64"


class EdgeRepresentation(StrEnum):
    """
    How the edges of a layered circuit's sum and product layers are stored.
    """

    SPARSE = "sparse"
    DENSE = "dense"


# %% setting


@dataclass(kw_only=True, frozen=True)
class ComputationSetting:
    """
    One combination of array library, device, floating point type and edge storage.
    """

    array_library: ArrayLibrary
    """
    The array library.
    """

    device: Device
    """
    The device.
    """

    floating_point_type: FloatingPointType
    """
    The floating point type.
    """

    edge_representation: EdgeRepresentation
    """
    The storage of the edges.
    """

    @classmethod
    def reference(cls) -> Self:
        """
        :return: The setting every method that trains on the GPU is measured in first,
            with dense edges since jax takes minutes to compile sparse ones.
        """
        return cls(
            array_library=ArrayLibrary.JAX,
            device=Device.GPU,
            floating_point_type=FloatingPointType.FLOAT32,
            edge_representation=EdgeRepresentation.DENSE,
        )

    @classmethod
    def host(cls) -> Self:
        """
        :return: The setting of methods that learn on the host with numpy alone.
        """
        return cls(
            array_library=ArrayLibrary.NUMPY,
            device=Device.CPU,
            floating_point_type=FloatingPointType.FLOAT64,
            edge_representation=EdgeRepresentation.SPARSE,
        )

    @classmethod
    def grid(
        cls,
        *,
        array_libraries: tuple[ArrayLibrary, ...],
        devices: tuple[Device, ...] = tuple(Device),
    ) -> list[Self]:
        """
        :param array_libraries: The array libraries of the grid.
        :param devices: The devices of the grid.
        :return: Every setting of these libraries and devices, the reference first.
        """
        settings = a(cls)(
            array_library=variable_from(array_libraries),
            device=variable_from(devices),
            floating_point_type=variable_from(tuple(FloatingPointType)),
            edge_representation=variable_from(tuple(EdgeRepresentation)),
        )
        return sorted(
            settings.evaluate(backend=EntityQueryLanguageGenerativeBackend()),
            key=lambda setting: setting != cls.reference(),
        )

    @property
    def key(self) -> str:
        """
        :return: A name of the setting usable in file names.
        """
        return "-".join(
            value.value
            for value in (
                self.array_library,
                self.device,
                self.floating_point_type,
                self.edge_representation,
            )
        )

    @property
    def uses_graphics_processor(self) -> bool:
        """
        :return: Whether the setting computes on the GPU.
        """
        return self.device is Device.GPU

    @property
    def namespace(self) -> ModuleType:
        """
        :return: The Array API namespace of the array library.
        """
        if self.array_library is ArrayLibrary.JAX:
            return jnp
        if self.array_library is ArrayLibrary.TORCH:
            return torch_namespace
        return np

    @property
    def device_object(self) -> Any:
        """
        :return: The device as the array library names it.
        """
        if self.array_library is ArrayLibrary.JAX:
            return jax.devices(self.device)[0]
        if self.array_library is ArrayLibrary.TORCH:
            return torch.device("cuda" if self.device is Device.GPU else "cpu")
        return None

    @property
    def dtype(self) -> Any:
        """
        :return: The floating point type as the array library names it.
        """
        if self.floating_point_type is FloatingPointType.FLOAT32:
            return self.namespace.float32
        return self.namespace.float64

    @property
    def edge_matrix_type(self) -> type[EdgeMatrix]:
        """
        :return: The type of edge matrix of the edge storage.
        """
        if self.edge_representation is EdgeRepresentation.DENSE:
            return DenseEdgeMatrix
        return SparseEdgeMatrix

    def placed(
        self, *, circuit: LayeredProbabilisticCircuit
    ) -> LayeredProbabilisticCircuit:
        """
        :param circuit: A circuit with host parameters.
        :return: A copy of the circuit with its edges and parameters stored as this
            setting says.
        """
        host = circuit.to_namespace(np)
        host.store_edges_as(self.edge_matrix_type)
        return host.to_namespace(
            self.namespace, device=self.device_object, floating_point_type=self.dtype
        )

    def asarray(self, *, rows: np.ndarray) -> Any:
        """
        :param rows: Rows on the host.
        :return: The rows as an array of this setting.
        """
        return self.namespace.asarray(rows, dtype=self.dtype, device=self.device_object)


# %% memory budget


@dataclass(kw_only=True)
class BudgetedMemory(DeviceMemory):
    """
    The memory of a device of which a run may use at most a fixed amount, since other
    runs share the device.
    """

    memory: DeviceMemory
    """
    The memory of the device.
    """

    budget_bytes: int
    """
    The most bytes the run may allocate.
    """

    def free_bytes(self) -> int:
        return min(self.memory.free_bytes(), self.budget_bytes)
