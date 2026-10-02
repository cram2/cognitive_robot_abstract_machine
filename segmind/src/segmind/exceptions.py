"""
Errors SegMind raises when it is asked for something it cannot do.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Type

from semantic_digital_twin.world_description.world_entity import SemanticAnnotation


@dataclass
class NoSemanticAnnotationToWatch(Exception):
    """
    Raised when a run asks for every semantic annotation of a type to be watched and the
    world holds none of that type.
    """

    semantic_annotation_type: Type[SemanticAnnotation]
    """
    The type asked for.
    """

    def __post_init__(self) -> None:
        super().__init__(
            f"The world holds no {self.semantic_annotation_type.__name__} to watch."
        )
