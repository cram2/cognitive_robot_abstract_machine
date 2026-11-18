"""
Devices whose memory tests fix.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass

from krrood_compute_hardware.memory import DeviceMemory

# %% devices


@dataclass(kw_only=True)
class FixedFreeMemory(DeviceMemory):
    """
    A device with a fixed amount of free memory.
    """

    free: int
    """
    The bytes the device reports free.
    """

    def free_bytes(self) -> int:
        return self.free
