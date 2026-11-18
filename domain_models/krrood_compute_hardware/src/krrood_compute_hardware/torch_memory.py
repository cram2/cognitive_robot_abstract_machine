"""
The memory of the CUDA devices torch stores tensors on.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass

import torch

from krrood_compute_hardware.memory import PeakTrackingMemory

# %% memory of a torch device


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
