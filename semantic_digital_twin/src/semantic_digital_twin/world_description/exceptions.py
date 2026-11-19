from __future__ import annotations

# %% Imports

from dataclasses import dataclass

from krrood.exceptions import DataclassException

# %% Color channels


@dataclass
class InvalidColorChannelCount(DataclassException, ValueError):
    """
    A channel sequence cannot describe an RGB or RGBA color.
    """

    channel_count: int
    """Number of supplied color channels."""

    def error_message(self) -> str:
        return f"A color cannot be constructed from {self.channel_count} channels."

    def suggest_correction(self) -> str:
        return "Supply three RGB channels or four RGBA channels."
