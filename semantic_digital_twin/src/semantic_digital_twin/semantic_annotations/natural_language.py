from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from semantic_digital_twin.semantic_annotations.mixins import (
    HasGraspPoses,
    HasRootBody,
)


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
class NaturalLanguageWithTypeDescription(NaturalLanguageDescription, HasGraspPoses):
    """
    A natural language description of a Sage10k object including the type information of the object.

    Graspable, unlike its base: a description can name anything in a scene, walls
    included, but the typed ones stand for the objects a robot is asked to pick up.
    """

    type_description: Optional[str] = field(default=None)
    """
    The cleaned description of the type of the object.
    """
