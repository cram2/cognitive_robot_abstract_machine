from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException
from semantic_digital_twin.adapters.artvip_dataset.schema import ArtVipCategory


@dataclass
class ArtVipObjectNotFoundError(DataclassException, LookupError):
    """
    Raised when no ArtVIP dataset entry matches a requested category and object name.
    """

    category: ArtVipCategory
    """The category that was searched."""

    name: str
    """
    The object name that could not be found.
    """

    def error_message(self) -> str:
        return (
            f"No ArtVIP object named '{self.name}' found in category "
            f"'{self.category.value}'."
        )

    def suggest_correction(self) -> str:
        return "Call ArtVipDatasetLoader.available_objects(category) for the names available in that category."


@dataclass
class ArtVipMainStageFileAmbiguousError(DataclassException, LookupError):
    """
    Raised when an ArtVIP object's directory does not contain exactly one top-level USD
    file to open as its main stage - either none (a folder present without its stage,
    or fully nested under a further subcategory) or more than one (an object whose main
    file cannot be picked out unambiguously).
    """

    category: ArtVipCategory
    """The object's category."""

    name: str
    """
    The object name whose directory was searched.
    """

    candidates: tuple[str, ...]
    """The top-level ``.usd`` file paths found directly in the object's directory."""

    def error_message(self) -> str:
        return (
            f"ArtVIP object '{self.name}' in category '{self.category.value}' has "
            f"{len(self.candidates)} top-level USD files, not exactly one: "
            f"{self.candidates}."
        )

    def suggest_correction(self) -> str:
        return (
            "Inspect the object's directory on the Hugging Face repository to "
            "identify its actual main stage file."
        )
