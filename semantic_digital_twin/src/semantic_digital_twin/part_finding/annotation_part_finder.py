"""
Giving an annotation the parts its part-whole relationships call for, found in the shape
of its root body and split off into bodies and annotations of their own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Optional

from semantic_digital_twin.adapters.part_finding.client import RemotePartLocator
from semantic_digital_twin.pipeline.mesh_decomposition.base import MeshDecomposer
from semantic_digital_twin.part_finding.relationships import PartRelationshipInspector
from semantic_digital_twin.part_finding.locating import (
    LocatedFacesFinder,
    LocatedPart,
    PartLocator,
    PartQuery,
)
from semantic_digital_twin.part_finding.splitting import PartSplitter
from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootBody,
    PartWholeRelationship,
)


@dataclass
class AnnotationPartFinder:
    """
    Gives an annotation with a root body the parts its part-whole relationships call for
    and it does not have yet.

    A part the annotation may lack -- an optional one, or one of any number -- is only
    added when the locator is at least :attr:`minimum_confidence` sure of it; a part the
    annotation requires is added whenever it is found.
    """

    locator: PartLocator = field(default_factory=RemotePartLocator)
    """
    Finds the parts' faces in the shape of the annotation's root body.
    """

    minimum_confidence: float = 0.5
    """
    How sure the locator has to be of a part the annotation may lack for it to be added.

    With the default locator the confidence is the share of the views showing the part
    that agree on it; every handle that was found correctly in the handle finder
    benchmark had at least 0.5.
    """

    decomposer: Optional[MeshDecomposer] = None
    """
    Gives the parts and the rest new collision shapes, for annotations whose root body's
    collision shapes differ from its visual shapes; see
    :attr:`~semantic_digital_twin.part_finding.splitting.PartSplitter.decomposer`.
    """

    def find_parts(self, whole: PartWholeRelationship) -> list[HasRootBody]:
        """
        Find the parts the annotation lacks in the shape of its root body, split each
        off into a body of its own, and add it to the annotation.

        :param whole: The annotation, whose root is a body.
        :return: The parts added, in the order of the annotation's part-whole
            relationships.
        """
        missing = [
            relationship.name
            for relationship in PartRelationshipInspector.all_of_type(type(whole))
            if not relationship.is_filled_in(whole)
        ]
        shapes = whole.root.visual or whole.root.collision
        if not missing or not shapes:
            return []
        query = PartQuery.from_mesh(shapes.combined_mesh, type(whole), missing)
        parts = []
        for located in self.locator.locate(query):
            part = self._add(whole, query, located)
            if part is not None:
                parts.append(part)
        return parts

    def _add(
        self, whole: PartWholeRelationship, query: PartQuery, located: LocatedPart
    ) -> HasRootBody | None:
        """
        :param whole: The annotation the part was located in.
        :param query: The query the part was located for.
        :param located: The part.
        :return: The part, split off the annotation's root body and added to the
            annotation; ``None`` if the annotation may lack it and the locator is not
            sure enough of it, or nothing of it is left in the body's shape.
        """
        relationship = PartRelationshipInspector.named(
            query.whole_type, located.relationship_name
        )
        if relationship.may_be_empty and located.confidence < self.minimum_confidence:
            return None
        splitter = PartSplitter(
            part_type=relationship.part_type,
            decomposer=self.decomposer,
            finder=LocatedFacesFinder(
                face_centers=query.mesh.triangles_center[located.faces]
            ),
        )
        return splitter.split(whole, field_name=relationship.name)
