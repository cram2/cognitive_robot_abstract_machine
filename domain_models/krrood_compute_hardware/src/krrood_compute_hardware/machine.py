"""
The machine a process runs on, and its division into shares that processes running side
by side each keep to.
"""

from __future__ import annotations

# %% imports
import os
from dataclasses import dataclass
from enum import StrEnum

from krrood_compute_hardware.devices import Device
from krrood_compute_hardware.exceptions import (
    EmptyMachineDivisionError,
    MissingGraphicsProcessorError,
)
from krrood_compute_hardware.memory import DeviceMemory, HostMemory

# %% environment of a process


class ThreadVariable(StrEnum):
    """
    The environment variables that cap the threads of the numeric libraries.
    """

    OPEN_MULTI_PROCESSING = "OMP_NUM_THREADS"
    OPEN_BASIC_LINEAR_ALGEBRA_SUBPROGRAMS = "OPENBLAS_NUM_THREADS"
    MATH_KERNEL_LIBRARY = "MKL_NUM_THREADS"


class GraphicsProcessorVariable(StrEnum):
    """
    The environment variables that configure how a process uses the GPU.
    """

    JAX_MEMORY_PREALLOCATION = "XLA_PYTHON_CLIENT_PREALLOCATE"
    """
    Whether jax claims most of the GPU memory when it starts.
    """

    VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
    """
    The GPUs a process sees.
    """


# %% machine


@dataclass(kw_only=True)
class Machine:
    """
    The processor cores and memory of the machine a process runs on.
    """

    processor_cores: set[int]
    """
    The processor cores the process may run on.
    """

    host_memory: DeviceMemory
    """
    The main memory of the machine.
    """

    graphics_processor_memory: DeviceMemory | None = None
    """
    The memory of the GPU, ``None`` on a machine without one.
    """

    @classmethod
    def from_current_process(
        cls, *, graphics_processor_memory: DeviceMemory | None = None
    ) -> Machine:
        """
        :param graphics_processor_memory: The memory of the GPU, ``None`` without one.
        :return: The machine as the current process sees it: the cores it may run on
            and the memory of the host within its control group.
        """
        return cls(
            processor_cores=os.sched_getaffinity(0),
            host_memory=HostMemory(),
            graphics_processor_memory=graphics_processor_memory,
        )


# %% shares of a machine


@dataclass(kw_only=True)
class MachineShare:
    """
    A share of a machine that one process uses at a time.
    """

    device: Device
    """
    The device the process computes on.
    """

    processor_cores: set[int]
    """
    The processor cores of the share.
    """

    memory_budget_bytes: int
    """
    The memory of its device the process may allocate.
    """

    host_memory_limit_bytes: int
    """
    The resident host memory above which the process is stopped.
    """

    def process_environment(self, *, environment: dict[str, str]) -> dict[str, str]:
        """
        :param environment: The environment the process would otherwise get.
        :return: That environment, with the threads of the numeric libraries capped at
            the cores of the share, and with the GPU hidden from a share on the CPU and
            shared without preallocation on a share of the GPU.
        """
        confined = dict(environment)
        for variable in ThreadVariable:
            confined[variable] = str(len(self.processor_cores))
        if self.device is Device.GRAPHICS_PROCESSOR:
            confined[GraphicsProcessorVariable.JAX_MEMORY_PREALLOCATION] = "false"
        else:
            confined[GraphicsProcessorVariable.VISIBLE_DEVICES] = ""
        return confined

    def confine_current_process(self) -> None:
        """
        Restrict the current process to the processor cores of the share.
        """
        os.sched_setaffinity(0, self.processor_cores)


@dataclass(kw_only=True)
class MachineDivision:
    """
    The division of a machine into shares of the GPU and shares of the processor cores
    alone.
    """

    graphics_processor_shares: int = 3
    """
    The number of shares that compute on the GPU.
    """

    processor_core_shares: int = 7
    """
    The number of shares that compute on the processor cores alone.
    """

    host_memory_fraction: float = 0.8
    """
    The fraction of the free host memory that is divided among the shares.
    """

    host_memory_limit_shares: float = 2.0
    """
    The number of host memory shares a process may hold before it is stopped.
    """

    graphics_processor_memory_fraction: float = 0.9
    """
    The fraction of the free GPU memory that is divided among the GPU shares, the rest
    kept free since the memory a process needs is often only estimated.
    """

    def __post_init__(self) -> None:
        """
        :raises EmptyMachineDivisionError: If the division has no shares.
        """
        if self.graphics_processor_shares + self.processor_core_shares == 0:
            raise EmptyMachineDivisionError()

    def shares_device(self, *, device: Device) -> bool:
        """
        :param device: A kind of device.
        :return: Whether the division has a share that computes on it.
        """
        if device is Device.GRAPHICS_PROCESSOR:
            return self.graphics_processor_shares > 0
        return self.processor_core_shares > 0

    def shares(self, *, machine: Machine) -> list[MachineShare]:
        """
        :param machine: The machine to divide.
        :return: The shares of the GPU, then the shares of the processor cores, each
            with its own equal part of the cores and of the free memory of its device.
        :raises MissingGraphicsProcessorError: If shares of the GPU are asked of a
            machine without one.
        """
        if self.graphics_processor_shares and machine.graphics_processor_memory is None:
            raise MissingGraphicsProcessorError(
                graphics_processor_shares=self.graphics_processor_shares
            )
        count = self.graphics_processor_shares + self.processor_core_shares
        cores = sorted(machine.processor_cores)
        cores_per_share = len(cores) // count
        host_share = int(
            self.host_memory_fraction * machine.host_memory.free_bytes() / count
        )
        graphics_processor_share = (
            int(
                self.graphics_processor_memory_fraction
                * machine.graphics_processor_memory.free_bytes()
                / self.graphics_processor_shares
            )
            if self.graphics_processor_shares
            else 0
        )
        return [
            MachineShare(
                device=(
                    Device.GRAPHICS_PROCESSOR
                    if index < self.graphics_processor_shares
                    else Device.CENTRAL_PROCESSOR
                ),
                processor_cores=set(
                    cores[index * cores_per_share : (index + 1) * cores_per_share]
                ),
                memory_budget_bytes=(
                    graphics_processor_share
                    if index < self.graphics_processor_shares
                    else host_share
                ),
                host_memory_limit_bytes=int(self.host_memory_limit_shares * host_share),
            )
            for index in range(count)
        ]
