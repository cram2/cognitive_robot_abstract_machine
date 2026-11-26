"""
Exceptions raised while describing the physics models of the semantic digital twin.
"""

from __future__ import annotations

from dataclasses import dataclass

from semantic_digital_twin.exceptions import UsageError
from semantic_digital_twin.world_description.geometry import Scale

# %% container geometry


@dataclass(kw_only=True)
class NonPositiveContainerGeometryError(UsageError):
    """
    Raised when a pouring container is described with a non-positive width across its
    lip or a non-positive height.
    """

    container_scale: Scale
    """
    The inner extents that were given, in metres.
    """

    def error_message(self) -> str:
        return (
            f"Container width (y) and height (z) must be strictly positive, got "
            f"{self.container_scale}."
        )

    def suggest_correction(self) -> str:
        return "describe the container with its positive inner extents in metres."
