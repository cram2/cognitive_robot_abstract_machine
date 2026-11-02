"""
Locating parts by asking a vision-language model where they are in views of a mesh.

The mesh is rendered from a ring of cameras, a detector puts boxes around the part in
each view, a segmenter masks the part inside the boxes, and the faces of the mesh vote
on whether they belong to the part by how often the views that show them mask them.

The detector and the segmenter are abstract, so this module runs without any model; the
models themselves are in
:mod:`~semantic_digital_twin.adapters.part_finding.vision_language_models`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import jinja2
import numpy as np
import trimesh
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from semantic_digital_twin.part_finding.locating import (
    LocatedPart,
    PartLocator,
    PartQuery,
)
from semantic_digital_twin.spatial_computations.mesh_rendering import (
    CameraRing,
    RenderedViews,
)

# %% finding the part in each view


@dataclass
class PartBoxPrompt:
    """
    The question a vision-language model is asked about each view of a whole.
    """

    template: str = field(
        default_factory=lambda: (Path(__file__).parent / "prompts" / "part_boxes.jinja")
        .read_text()
        .strip()
    )
    """
    A Jinja2 template, with ``part`` and ``whole`` standing for the names of the part
    and the whole.
    """

    def question(self, part_name: str, whole_name: str) -> str:
        """
        :param part_name: What the part is called, such as ``"handle"``.
        :param whole_name: What the whole is called, such as ``"mug"``.
        :return: The question about the part of the whole.
        """
        environment = jinja2.Environment(undefined=jinja2.StrictUndefined)
        return environment.from_string(self.template).render(
            part=part_name, whole=whole_name
        )


class VisionModel(ABC):
    """
    A model that looks at views of a mesh.
    """

    def load(self) -> None:
        """
        Load the model's weights, so that the first use does not wait for them; a model
        without weights has nothing to load.
        """


class PartBoxDetector(VisionModel):
    """
    Puts boxes around a part of a whole in views of it.
    """

    @abstractmethod
    def detect(
        self, views: RenderedViews, part_name: str, whole_name: str
    ) -> list[NDArray[np.float64]]:
        """
        :param views: The views of the whole.
        :param part_name: What the part is called, such as ``"handle"``.
        :param whole_name: What the whole is called, such as ``"mug"``.
        :return: For each view, the boxes around the part in it, as rows of the
            smallest and largest column and the smallest and largest row, in pixels;
            no rows where the view does not show the part.
        """


class BoxSegmenter(VisionModel):
    """
    Cuts out what each box of a :class:`PartBoxDetector` was drawn around: a box is
    rough and also holds pixels of the rest of the whole, the mask marks only the pixels
    that show the part itself.
    """

    @abstractmethod
    def segment(
        self, views: RenderedViews, boxes: list[NDArray[np.float64]]
    ) -> NDArray[np.bool_]:
        """
        :param views: The views of the whole.
        :param boxes: For each view, the boxes around the part, as
            :meth:`PartBoxDetector.detect` returns them.
        :return: For each pixel of each view, whether it shows the part.
        """


@dataclass
class KnownPartBoxDetector(PartBoxDetector):
    """
    Boxes the faces it is told belong to each part, wherever a view shows them: a stand-
    in for a model wherever the parts are known beforehand, such as in tests.
    """

    part_faces: dict[str, list[NDArray[np.int64]]] = field(default_factory=dict)
    """
    For each part name, the indices of the faces of each part of that name.
    """

    def detect(
        self, views: RenderedViews, part_name: str, whole_name: str
    ) -> list[NDArray[np.float64]]:
        return [
            np.array(
                [
                    box
                    for faces in self.part_faces.get(part_name, [])
                    if (box := self._box_around(view_faces, faces)) is not None
                ],
                dtype=np.float64,
            ).reshape(-1, 4)
            for view_faces in views.faces
        ]

    @staticmethod
    def _box_around(
        view_faces: NDArray[np.int64], faces: NDArray[np.int64]
    ) -> list[int] | None:
        """
        :param view_faces: For each pixel of a view, the face it shows.
        :param faces: The faces of a part.
        :return: The box around the pixels that show the part; ``None`` if none does.
        """
        rows, columns = np.nonzero(np.isin(view_faces, faces))
        if len(rows) == 0:
            return None
        return [columns.min(), rows.min(), columns.max() + 1, rows.max() + 1]


@dataclass
class FilledBoxSegmenter(BoxSegmenter):
    """
    Masks all of the whole inside each box: a stand-in for a model wherever the boxes
    fit the part tightly, such as in tests.
    """

    def segment(
        self, views: RenderedViews, boxes: list[NDArray[np.float64]]
    ) -> NDArray[np.bool_]:
        masks = np.zeros(views.faces.shape, dtype=bool)
        for mask, view_boxes in zip(masks, boxes):
            for column_start, row_start, column_end, row_end in view_boxes.astype(int):
                mask[row_start:row_end, column_start:column_end] = True
        return masks & views.silhouettes


# %% voting on the faces


@dataclass
class ViewVote:
    """
    Decides which faces of a mesh belong to a part from the masks of the part in views
    of the mesh.
    """

    threshold: float = 0.5
    """
    The share of the views showing a face that have to mask it for the face to belong to
    the part.
    """

    minimum_visible_pixels: int = 20
    """
    How many pixels of the part a view has to show to have a say in how sure the vote
    is.
    """

    def belongs_to_part_mask(
        self, mesh: trimesh.Trimesh, views: RenderedViews, masks: NDArray[np.bool_]
    ) -> NDArray[np.bool_]:
        """
        :param mesh: The mesh the views show.
        :param views: The views.
        :param masks: For each pixel of each view, whether it shows the part.
        :return: For each face, whether it belongs to the part. A face no view shows
            takes the vote of the closest face a view shows.
        """
        silhouettes = views.silhouettes
        shown_faces = views.faces[silhouettes]
        face_count = len(mesh.faces)
        shown = np.bincount(shown_faces, minlength=face_count)
        masked = np.bincount(
            shown_faces, weights=masks[silhouettes], minlength=face_count
        )
        if not shown.any():
            return np.zeros(face_count, dtype=bool)
        is_shown = shown > 0
        share = np.zeros(face_count)
        share[is_shown] = masked[is_shown] / shown[is_shown]
        centers = mesh.triangles_center
        _, closest = cKDTree(centers[is_shown]).query(centers[~is_shown])
        share[~is_shown] = share[is_shown][closest]
        return share > self.threshold

    def confidence(
        self,
        belongs_to_part_mask: NDArray[np.bool_],
        views: RenderedViews,
        masks: NDArray[np.bool_],
    ) -> float:
        """
        :param belongs_to_part_mask: For each face, whether it belongs to the part.
        :param views: The views the part was found in.
        :param masks: For each pixel of each view, whether it shows the part.
        :return: The share of the views showing at least
            :attr:`minimum_visible_pixels` pixels of the part that mask most of them;
            0 if no view shows that much of it.
        """
        agreeing = showing = 0
        for view_faces, mask in zip(views.faces, masks):
            shows_part = (view_faces >= 0) & belongs_to_part_mask[
                np.maximum(view_faces, 0)
            ]
            if shows_part.sum() < self.minimum_visible_pixels:
                continue
            showing += 1
            agreeing += mask[shows_part].mean() > 0.5
        return agreeing / showing if showing else 0.0


# %% the locator


@dataclass
class VisionLanguagePartLocator(PartLocator):
    """
    Locates parts by rendering the mesh, finding each part in the views with a detector
    and a segmenter, and letting the faces vote.

    The faces found fall apart into connected pieces. A relationship that holds any
    number of parts gets each piece as a part of its own; one that holds a single part
    gets the piece the locator is most sure of.
    """

    detector: PartBoxDetector
    """
    Puts boxes around a part in each view.
    """

    segmenter: BoxSegmenter
    """
    Masks the part inside the boxes.
    """

    cameras: CameraRing = field(default_factory=CameraRing)
    """
    Renders the views.
    """

    vote: ViewVote = field(default_factory=ViewVote)
    """
    Decides from the masks which faces belong to a part.
    """

    def locate(self, query: PartQuery) -> list[LocatedPart]:
        mesh = query.mesh
        views = self.cameras.render(mesh)
        located = []
        for relationship in query.relationships:
            boxes = self.detector.detect(
                views, relationship.part_name, query.whole_name
            )
            masks = self.segmenter.segment(views, boxes)
            belongs_to_part_mask = self.vote.belongs_to_part_mask(mesh, views, masks)
            if not belongs_to_part_mask.any():
                continue
            pieces = [
                LocatedPart(
                    relationship_name=relationship.name,
                    faces=np.flatnonzero(piece),
                    confidence=self.vote.confidence(piece, views, masks),
                )
                for piece in self._connected_pieces(mesh, belongs_to_part_mask)
            ]
            if not relationship.holds_many:
                pieces = [
                    max(pieces, key=lambda part: (part.confidence, len(part.faces)))
                ]
            located.extend(pieces)
        return located

    def load(self) -> None:
        """
        Load the models, so that the first query does not wait for them.
        """
        self.detector.load()
        self.segmenter.load()

    @staticmethod
    def _connected_pieces(
        mesh: trimesh.Trimesh, belongs_to_part_mask: NDArray[np.bool_]
    ) -> list[NDArray[np.bool_]]:
        """
        :param mesh: A mesh.
        :param belongs_to_part_mask: For each face, whether it belongs to the parts.
        :return: For each piece of the faces that is connected over shared vertices,
            which faces belong to it.
        """
        connected = mesh.copy()
        connected.merge_vertices()
        adjacency = connected.face_adjacency
        pieces = []
        for faces in trimesh.graph.connected_components(
            adjacency[belongs_to_part_mask[adjacency].all(axis=1)],
            nodes=np.flatnonzero(belongs_to_part_mask),
        ):
            piece = np.zeros(len(mesh.faces), dtype=bool)
            piece[faces] = True
            pieces.append(piece)
        return pieces
