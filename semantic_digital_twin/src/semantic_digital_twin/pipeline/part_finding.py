"""
Pipeline steps that give annotations their parts, see
:mod:`semantic_digital_twin.part_finding`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING, Type

from semantic_digital_twin.part_finding.annotation_part_finder import (
    AnnotationPartFinder,
)
from semantic_digital_twin.part_finding.splitting import PartSplitter
from semantic_digital_twin.pipeline.pipeline import Step
from semantic_digital_twin.semantic_annotations.mixins import PartWholeRelationship

if TYPE_CHECKING:
    from semantic_digital_twin.world import World


@dataclass
class FindParts(Step):
    """
    Gives every annotation of a type the parts its part-whole relationships call for and
    it does not have yet.
    """

    whole_type: Type[PartWholeRelationship]
    """
    The type of annotation whose parts are found.
    """

    finder: AnnotationPartFinder = field(default_factory=AnnotationPartFinder)
    """
    Finds the parts of each annotation.
    """

    def _apply(self, world: World) -> World:
        for whole in world.get_semantic_annotations_by_type(self.whole_type):
            self.finder.find_parts(whole)
        return world


@dataclass
class SplitPartFromShape(Step):
    """
    Gives every annotation of a type that has no such part yet the part its shape has,
    as a body of its own.
    """

    whole_type: Type[PartWholeRelationship]
    """
    The type of annotation whose shapes are split.
    """

    splitter: PartSplitter
    """
    Splits the part off each annotation's shape.
    """

    def _apply(self, world: World) -> World:
        for whole in world.get_semantic_annotations_by_type(self.whole_type):
            self.splitter.part_of(whole)
        return world
