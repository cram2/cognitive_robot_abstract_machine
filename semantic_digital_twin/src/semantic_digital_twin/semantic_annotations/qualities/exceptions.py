"""
The exceptions raised by the qualities of semantic annotations.
"""

from __future__ import annotations

# %% Imports

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

from krrood.exceptions import DataclassException

if TYPE_CHECKING:
    from semantic_digital_twin.semantic_annotations.qualities.quality import (
        Ripeness,
    )


# %% Ripeness


@dataclass(kw_only=True)
class RipenessOutsideScale(DataclassException):
    """
    Raised when a ripeness is given a value that reaches beyond the ripeness scale.
    """

    ripeness: Ripeness
    """
    The ripeness with the value outside the scale.
    """

    def error_message(self) -> str:
        return (
            f"The ripeness {self.ripeness.value} reaches beyond the ripeness scale "
            f"{self.ripeness.scale}."
        )

    def suggest_correction(self) -> str:
        return "Give the ripeness a value within the ripeness scale, from 0 to 1."


@dataclass(kw_only=True)
class RipenessSpansSeveralStages(DataclassException):
    """
    Raised when the stage of a ripeness is asked for, but its value reaches into more
    than one ripeness stage.
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
            "ripeness_interval of each stage directly."
        )
