"""
Exceptions raised while describing the physics models of the semantic digital twin.
"""

from __future__ import annotations

from dataclasses import dataclass

from semantic_digital_twin.exceptions import UsageError

# %% container geometry


@dataclass(kw_only=True)
class NonPositiveContainerHeightError(UsageError):
    """
    Raised when a pouring container is described with a non-positive height.
    """

    container_height: float
    """
    The inner height that was given, in metres.
    """

    def error_message(self) -> str:
        return (
            f"Container height must be strictly positive, got {self.container_height}."
        )

    def suggest_correction(self) -> str:
        return "describe the container with its positive inner height in metres."
