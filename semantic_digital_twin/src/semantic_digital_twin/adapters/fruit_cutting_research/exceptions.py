"""
The exceptions raised while parsing a sample of the fruit-cutting generator.
"""

from __future__ import annotations

# %% Imports

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

from krrood.exceptions import DataclassException

if TYPE_CHECKING:
    from semantic_digital_twin.adapters.fruit_cutting_research.schema import (
        FruitCuttingSample,
        FruitCuttingTissue,
    )

# %% Exceptions


@dataclass(kw_only=True)
class UnknownTissueId(DataclassException):
    """
    Raised when a mesh of a sample's model gives its faces a tissue id the sample's
    record has no tissue for.
    """

    sample: FruitCuttingSample
    """
    The sample whose record lacks the tissue.
    """

    tissue_id: int
    """
    The tissue id, counted from 1.
    """

    def error_message(self) -> str:
        return (
            f"The model of sample {self.sample.name} uses tissue id {self.tissue_id}, "
            f"but its record lists only {len(self.sample.materials)} tissues."
        )

    def suggest_correction(self) -> str:
        return (
            "Check that the record sample.json belongs to the model fruit.usdz beside "
            "it."
        )


@dataclass(kw_only=True)
class PrimWithoutSinglePart(DataclassException):
    """
    Raised when the tissues a prim of a sample's model is made of do not make up exactly
    one part of the fruit, so the prim cannot become a part.
    """

    sample: FruitCuttingSample
    """
    The sample the prim belongs to.
    """

    prim_path: str
    """
    The path of the prim.
    """

    tissues: list[FruitCuttingTissue]
    """
    The tissues the prim is made of.
    """

    def error_message(self) -> str:
        return (
            f"The prim {self.prim_path} of sample {self.sample.name} is made of "
            f"{[str(tissue) for tissue in self.tissues]}, which do not make up exactly "
            f"one part."
        )

    def suggest_correction(self) -> str:
        return (
            "Give the tissue a part type in FruitCuttingTissue.choose_part_type, or "
            "check that the prim holds a single part."
        )
