from __future__ import annotations

from dataclasses import dataclass, field

from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.mixins import HasGraspCandidates


@dataclass(eq=False)
class NaturalLanguageDescription(HasRootBody):
    """
    Annotation for descriptions of the root in natural language.
    """

    description: str = field(kw_only=True)
    """
    The natural language description of root entity.
    """


@dataclass(eq=False)
class NaturalLanguageWithTypeDescription(
    NaturalLanguageDescription, HasGraspCandidates
):
    """
    A natural language description of a Sage10k object including the type information of the object.

    Graspable, since a typed description stands for an object a robot may pick up.
    """

    type_description: str | None = field(default=None)
    """
    The cleaned description of the type of the object.
    """
