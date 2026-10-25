"""
Locating the parts an annotation's part-whole relationships call for in the mesh of its
root body, as faces of that mesh.

A :class:`PartLocator` answers a :class:`PartQuery` -- a mesh and the type of annotation
it belongs to -- with the faces of each part it finds. It may run on this machine or on
a server, which is why both the query and the answer are plain data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import trimesh
from krrood.entity_query_language.utils import camel_case_to_words
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from typing_extensions import Optional, Type

from semantic_digital_twin.part_finding.relationships import PartRelationshipInspector
from semantic_digital_twin.part_finding.splitting import PartFinder, MeshSplit
from semantic_digital_twin.semantic_annotations.mixins import PartWholeRelationship

# %% the question and its answer


@dataclass
class PartQuery:
    """
    The question which faces of a mesh belong to the parts an annotation's part-whole
    relationships call for.
    """

    vertices: NDArray[np.float64]
    """
    The mesh's vertices, as rows of x, y and z in the frame of the annotation's root
    body, its z-axis pointing up.
    """

    faces: NDArray[np.int64]
    """
    The mesh's triangles, as rows of three indices into :attr:`vertices`.
    """

    whole_type: Type[PartWholeRelationship]
    """
    The type of annotation the mesh belongs to.
    """

    relationship_names: list[str]
    """
    The names of the part-whole relationships whose parts are wanted.
    """

    @classmethod
    def from_mesh(
        cls,
        mesh: trimesh.Trimesh,
        whole_type: Type[PartWholeRelationship],
        relationship_names: list[str],
    ) -> PartQuery:
        """
        :param mesh: The mesh of an annotation's root body, in its frame.
        :param whole_type: The type of the annotation.
        :param relationship_names: The part-whole relationships whose parts are wanted.
        :return: The query for those parts.
        """
        return cls(
            vertices=np.asarray(mesh.vertices, dtype=np.float64),
            faces=np.asarray(mesh.faces, dtype=np.int64),
            whole_type=whole_type,
            relationship_names=list(relationship_names),
        )

    @cached_property
    def mesh(self) -> trimesh.Trimesh:
        """
        :return: The mesh, its faces in the order of :attr:`faces`.
        """
        return trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=False)

    @property
    def relationships(self) -> list[PartRelationshipInspector]:
        """
        :return: The part-whole relationships whose parts are wanted.
        """
        return [
            PartRelationshipInspector.named(self.whole_type, name)
            for name in self.relationship_names
        ]

    @property
    def whole_name(self) -> str:
        """
        :return: What the annotation is called in plain words, such as ``"mug"``.
        """
        return camel_case_to_words(self.whole_type.__name__)


@dataclass
class LocatedPart:
    """
    A part found in the mesh of a :class:`PartQuery`.
    """

    relationship_name: str
    """
    The name of the part-whole relationship the part belongs to.
    """

    faces: NDArray[np.int64]
    """
    The indices of the mesh's faces that make up the part.
    """

    confidence: float
    """
    How sure the locator is that these faces are the part, from 0 to 1.
    """


class PartLocator(ABC):
    """
    Finds which faces of a mesh belong to the parts of an annotation.
    """

    @abstractmethod
    def locate(self, query: PartQuery) -> list[LocatedPart]:
        """
        :param query: The mesh and the parts wanted.
        :return: The parts found: for a relationship that holds a single part at most
            one, for one that holds any number each separate piece. A part that is not
            found is left out.
        """


@dataclass
class KnownPartLocator(PartLocator):
    """
    Answers every query with the same parts: a stand-in for a real locator wherever the
    answer is known beforehand, such as in tests.
    """

    parts: list[LocatedPart] = field(default_factory=list)
    """
    The parts every query is answered with.
    """

    queries: list[PartQuery] = field(default_factory=list, init=False)
    """
    The queries asked so far, oldest first.
    """

    def locate(self, query: PartQuery) -> list[LocatedPart]:
        self.queries.append(query)
        return list(self.parts)


# %% splitting a located part off


@dataclass
class LocatedFacesFinder(PartFinder):
    """
    Finds a part whose faces were located in a mesh before, either in that mesh or in
    what is left of it after other parts were split off, by the faces' centers.
    """

    face_centers: NDArray[np.float64]
    """
    The centers of the part's faces, as rows of x, y and z.
    """

    tolerance: float = 1e-6
    """
    How far a face's center may lie from one of :attr:`face_centers` to still be one of
    the part's faces, in meters.
    """

    def split(self, mesh: trimesh.Trimesh) -> Optional[MeshSplit]:
        distances, _ = cKDTree(self.face_centers).query(mesh.triangles_center)
        return MeshSplit.from_mesh(mesh, distances <= self.tolerance)
