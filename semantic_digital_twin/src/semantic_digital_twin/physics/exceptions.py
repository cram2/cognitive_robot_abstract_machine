"""
Exceptions raised while describing the physics models of the semantic digital twin.
"""

from __future__ import annotations

from dataclasses import dataclass

from semantic_digital_twin.exceptions import UsageError

# %% container geometry


@dataclass(kw_only=True)
class NonPositiveContainerGeometryError(UsageError):
    """
    Raised when a pouring-domain container is described with a non-positive height or
    width.
    """

    container_height: float
    """
    The container height that was given, in metres.
    """

    container_width: float
    """
    The container width that was given, in metres.
    """

    def error_message(self) -> str:
        return (
            f"Container geometry must be strictly positive, got height="
            f"{self.container_height} and width={self.container_width}."
        )

    def suggest_correction(self) -> str:
        return (
            "describe the container with its positive inner height and width in metres."
        )
