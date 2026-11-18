"""
The memory of the accelerators jax stores arrays on.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass
from enum import StrEnum

import jax

from krrood_compute_hardware.memory import DeviceMemory

# %% memory of a jax device


class JaxMemoryStatistic(StrEnum):
    """
    The entries of :meth:`jax.Device.memory_stats` that are read, each in bytes.
    """

    LIMIT = "bytes_limit"
    IN_USE = "bytes_in_use"


@dataclass(kw_only=True)
class JaxDeviceMemory(DeviceMemory):
    """
    The memory of an accelerator jax stores arrays on.
    """

    device: jax.Device
    """
    The accelerator.
    """

    def free_bytes(self) -> int:
        statistics = self.device.memory_stats()
        return (
            statistics[JaxMemoryStatistic.LIMIT] - statistics[JaxMemoryStatistic.IN_USE]
        )
