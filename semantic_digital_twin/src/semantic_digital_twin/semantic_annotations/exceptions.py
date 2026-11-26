"""
The exceptions raised by the qualities of semantic annotations.
"""

from __future__ import annotations

# %% Imports

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

from krrood.exceptions import DataclassException
from semantic_digital_twin.exceptions import UsageError

if TYPE_CHECKING:
    from semantic_digital_twin.semantic_annotations.qualities import Quality, Ripeness
    from semantic_digital_twin.world_description.world_entity import (
        SemanticAnnotation,
    )

# %% Setting qualities


@dataclass(kw_only=True)
class CannotHaveQuality(UsageError):
    """
    Raised when ``set_quality`` is called with a quality that no quality field of the
    annotation accepts.
    """

    annotation: SemanticAnnotation
    """
    The annotation the quality was being set on.
    """

    quality: Quality
    """
    The quality that could not be set.
    """

    def error_message(self) -> str:
        return (
            f"{type(self.annotation).__name__} cannot have a "
            f"{type(self.quality).__name__}: no quality field accepts it."
        )

    def suggest_correction(self) -> str:
        return (
            f"Check the quality mixins of {type(self.annotation).__name__} for the "
            f"qualities it can have, and add a mixin for "
            f"{type(self.quality).__name__} where it belongs."
        )


@dataclass(kw_only=True)
class AmbiguousQuality(UsageError):
    """
    Raised when ``set_quality`` is called with a quality that more than one quality
    field of the annotation accepts.
    """

    annotation: SemanticAnnotation
    """
    The annotation the quality was being set on.
    """

    quality: Quality
    """
    The quality that could not be set unambiguously.
    """

    field_names: list[str]
    """
    The names of the quality fields that accept the quality.
    """

    def error_message(self) -> str:
        return (
            f"{type(self.quality).__name__} cannot be set unambiguously on "
            f"{type(self.annotation).__name__}: the fields "
            f"{', '.join(self.field_names)} all accept it."
        )

    def suggest_correction(self) -> str:
        return (
            f"Give {type(self.annotation).__name__} only one quality field for "
            f"{type(self.quality).__name__}."
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
