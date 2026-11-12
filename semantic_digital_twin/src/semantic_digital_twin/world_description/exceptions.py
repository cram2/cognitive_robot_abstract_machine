from __future__ import annotations

# %% Imports

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

from krrood.exceptions import DataclassException

if TYPE_CHECKING:
    from semantic_digital_twin.world_description.world_entity import (
        KinematicStructureEntity,
    )

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


# %% Footprints


@dataclass(kw_only=True)
class NoShapesToProject(DataclassException):
    """
    Raised when the footprint of a shape collection that holds no shapes is asked for.
    """

    reference_frame: KinematicStructureEntity | None
    """
    The entity the empty collection belongs to, if any.
    """

    def error_message(self) -> str:
        return (
            f"The shape collection of '{self.reference_frame}' holds no shapes, so it "
            f"covers nothing to project."
        )

    def suggest_correction(self) -> str:
        return "Add the shapes of the body before asking for its footprint."
