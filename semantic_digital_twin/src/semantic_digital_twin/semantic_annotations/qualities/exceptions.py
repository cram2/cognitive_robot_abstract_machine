"""
The exceptions raised by the qualities of semantic annotations.
"""

from __future__ import annotations

# %% Imports

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

from krrood.exceptions import DataclassException

if TYPE_CHECKING:
    from random_events.interval import SimpleInterval
    from semantic_digital_twin.semantic_annotations.qualities.quality import (
        Quality,
        Ripeness,
    )


# %% Scales


@dataclass(kw_only=True)
class QualityOutsideScale(DataclassException):
    """
    Raised when a quality is given a value that reaches beyond the scale it lies on.
    """

    quality: Quality
    """
    The quality with the value outside the scale.
    """

    value: SimpleInterval
    """
    The value that reaches beyond :attr:`scale`.
    """

    scale: SimpleInterval
    """
    The scale the value has to lie within.
    """

    def error_message(self) -> str:
        return (
            f"The {type(self.quality).__name__} value {self.value} reaches beyond its "
            f"scale {self.scale}."
        )

    def suggest_correction(self) -> str:
        return (
            f"Give the {type(self.quality).__name__} a value within {self.scale}, in the "
            f"unit its docstring states."
        )


# %% Ripeness


@dataclass(kw_only=True)
class RipenessInNoSingleStage(DataclassException):
    """
    Raised when the stage of a ripeness is asked for, but no single stage holds its
    whole value, because the value reaches into several stages.
    """

    ripeness: Ripeness
    """
    The ripeness whose stage was asked for.
    """

    def error_message(self) -> str:
        return (
            f"The ripeness {self.ripeness.value} reaches into more than one ripeness "
            f"stage, so it has no single stage."
        )

    def suggest_correction(self) -> str:
        return (
            "Narrow the ripeness value to lie within one stage, or compare it with the "
            "interval of each stage directly."
        )
