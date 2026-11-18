"""
The kinds of device a computation runs on.
"""

from __future__ import annotations

# %% imports
from enum import StrEnum

# %% devices


class Device(StrEnum):
    """
    The kind of device a computation runs on, valued as jax names its platform.
    """

    CENTRAL_PROCESSOR = "cpu"
    GRAPHICS_PROCESSOR = "gpu"
