"""
The exceptions raised when adding a part to a whole.
"""

from __future__ import annotations

# %% Imports

from dataclasses import dataclass, Field

from typing_extensions import TYPE_CHECKING

from semantic_digital_twin.exceptions import UsageError

if TYPE_CHECKING:
    from semantic_digital_twin.world_description.world_entity import (
        SemanticAnnotation,
    )

# %% Adding parts


@dataclass(kw_only=True)
class CannotBeAPartOf(UsageError):
    """
    Raised when ``add`` is called with a part that no part-whole relationship field of
    the annotation accepts.
    """

    annotation: SemanticAnnotation
    """
    The annotation the part was being added to.
    """

    part: SemanticAnnotation
    """
    The part that could not be added.
    """

    def error_message(self) -> str:
        return (
            f"{type(self.part).__name__} cannot be added as a part of "
            f"{type(self.annotation).__name__}: no part-whole relationship field accepts it."
        )

    def suggest_correction(self) -> str:
        return (
            f"Check out the superclasses of {type(self.annotation).__name__}, to find the currently valid part-whole "
            f"relationships, and if you think its currently incomplete, feel free to adjust it and make a PR for it"
        )


@dataclass(kw_only=True)
class AmbiguousPart(UsageError):
    """
    Raised when ``add`` is called with a part whose type matches more than one part-
    whole relationship field of the annotation, so the target field is ambiguous.
    """

    annotation: SemanticAnnotation
    """
    The annotation the part was being added to.
    """

    part: SemanticAnnotation
    """
    The part that could not be unambiguously added.
    """

    fields: list[Field]
    """
    The names of the part-whole relationship fields whose element type accepts the part.
    """

    def error_message(self) -> str:
        return (
            f"{type(self.part).__name__} cannot be unambiguously added as a part of "
            f"{type(self.annotation).__name__}: it matches multiple part-whole relationship fields "
            f"({', '.join([field_.name for field_ in self.fields])})."
        )

    def suggest_correction(self) -> str:
        return (
            "consider if its practical to use the 'field_name' keyword argument for the `add` method to"
            " disambiguate the matching cases."
        )
