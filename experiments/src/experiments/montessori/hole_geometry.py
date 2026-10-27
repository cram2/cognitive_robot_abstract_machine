"""
Detect the Montessori shape-sorting board's hole footprints directly from its mesh
(``resources/board.stl``), instead of hand-authoring their positions and sizes as
constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import trimesh
from typing_extensions import ClassVar, List, Type

from experiments.montessori.semantics import (
    CubeShape,
    CylinderShape,
    DiskShape,
    MontessoriShape,
    RectangularPrismShape,
    SphereShape,
    TriangularPrismShape,
)
from semantic_digital_twin.spatial_types.spatial_types import Point2
from semantic_digital_twin.world_description.geometry import Polygon2D, Scale


@dataclass(frozen=True)
class HoleShapeClassifier:
    """
    Names the kind of piece a hole is cut for from the outline of its cross-section.
    """

    circle_vertex_count: int = 20
    """
    Circular hole boundaries are tessellated into far more polygon vertices than the
    straight-edged holes (~65 vs.

    9-13 in the source mesh); above this count an outline is classified as circular.
    """

    triangle_fill_ratio: float = 0.7
    """
    An outline's area divided by its bounding box's area, below which it is classified
    as triangular.

    A triangle inscribed in its bounding box fills at most half of it; the box-shaped
    holes fill all of it.
    """

    disk_aspect_ratio: float = 5.0
    """
    Ratio of an outline's bounding box's longer side to its shorter side, above which it
    is classified as the disk's slot.

    The disk hole is a narrow slot (~10:1); the square and rectangular holes are much
    closer to square.
    """

    rectangle_aspect_ratio: float = 1.3
    """
    Bounding-box aspect ratio above which a box-shaped hole is classified as rectangular
    rather than square.
    """

    sphere_hole_width: float = 0.036
    """
    Width, in metres, above which a circular hole is the sphere's rather than the
    cylinder's.

    The board has two circular holes, 32 and 40 millimetres across, and this sits midway
    between them.
    """

    def classify(self, outline: Polygon2D) -> Type[MontessoriShape]:
        """
        The kind of piece a hole of this cross-section is cut for.

        :param outline: The hole's cross-section.
        """
        bounding_box = outline.bounding_box
        aspect_ratio = max(bounding_box.dimensions) / min(bounding_box.dimensions)
        if len(outline.vertices) > self.circle_vertex_count:
            if min(bounding_box.dimensions) > self.sphere_hole_width:
                return SphereShape
            return CylinderShape
        if outline.area / bounding_box.area < self.triangle_fill_ratio:
            return TriangularPrismShape
        if aspect_ratio > self.disk_aspect_ratio:
            return DiskShape
        if aspect_ratio > self.rectangle_aspect_ratio:
            return RectangularPrismShape
        return CubeShape


@dataclass(frozen=True, eq=False)
class HoleFootprint:
    """
    A single hole's position and 2D footprint, detected from the board mesh.
    """

    category: Type[MontessoriShape]
    """
    The kind of piece the hole is cut for.
    """

    center: Point2
    """
    The hole's area centroid, in the board mesh's local ``(x, y)`` frame.
    """

    boundary: Polygon2D
    """
    The hole's true cross-section outline, relative to :attr:`center` (as opposed to its
    bounding box).
    """

    MARKER_THICKNESS: ClassVar[float] = 0.005
    """
    Thickness (along the board's z-axis) of the thin solid that marks a hole's
    footprint; the marker's top face sits flush with the board's top surface.
    """

    def extrude(self, thickness: float) -> trimesh.Trimesh:
        """
        Extrude this hole's true boundary into a solid of the given thickness, centered
        on its own local origin (i.e. on :attr:`center`, once translated there).

        :param thickness: Extrusion depth along z.
        """
        return self.boundary.extrude(thickness)

    def marker(self) -> trimesh.Trimesh:
        """
        The thin solid, :attr:`MARKER_THICKNESS` thick, that marks this hole's
        footprint.
        """
        return self.extrude(self.MARKER_THICKNESS)

    def cut_through(self, solid: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Cut this hole all the way through a solid, in its true cross-section shape
        rather than its bounding box.

        :param solid: The solid to cut, in the board mesh's local frame.
        :return: The solid with this hole cut clean through it.
        """
        cutter = self.extrude(float(solid.extents[2]) * 2)
        cutter.apply_translation([*self.center.to_np(), 0.0])
        return solid.difference(cutter, engine=None)


@dataclass(frozen=True)
class ShapeSortingBoardMesh:
    """
    The shape-sorting board's mesh, cut with its six real shape holes, and the holes
    read off it.
    """

    path: Path = Path(__file__).parent / "resources" / "board.stl"
    """
    Path to the mesh file.
    """

    hole_classifier: HoleShapeClassifier = field(default_factory=HoleShapeClassifier)
    """
    Names the kind of piece each hole read off the mesh is cut for.
    """

    @cached_property
    def hole_footprints(self) -> List[HoleFootprint]:
        """
        The board's holes, found by slicing its mesh horizontally through the lid and
        reading each interior outline of the cut.

        :return: One :class:`HoleFootprint` per hole cut into the board, ordered by
            ascending y-position.
        """
        lid = self._perforated_body(trimesh.load(self.path))
        mid_z = (lid.bounds[0][2] + lid.bounds[1][2]) / 2
        section = lid.section(
            plane_origin=[0.0, 0.0, mid_z], plane_normal=[0.0, 0.0, 1.0]
        )
        outlines = [Polygon2D(np.asarray(loop)[:, :2]) for loop in section.discrete]
        outer_boundary = max(outlines, key=lambda outline: outline.bounding_box.area)
        footprints = [
            HoleFootprint(
                category=self.hole_classifier.classify(outline),
                center=outline.centroid,
                boundary=outline.centered(),
            )
            for outline in outlines
            if outline is not outer_boundary
        ]
        return sorted(footprints, key=lambda footprint: float(footprint.center.y))

    def cut_blank(self, blank_scale: Scale) -> trimesh.Trimesh:
        """
        A solid board blank with every one of :attr:`hole_footprints` cut all the way
        through it.

        :param blank_scale: Size of the uncut board blank.
        """
        board = trimesh.creation.box(
            extents=(blank_scale.x, blank_scale.y, blank_scale.z)
        )
        for footprint in self.hole_footprints:
            board = footprint.cut_through(board)
        return board

    @staticmethod
    def _perforated_body(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Find the connected part of ``mesh`` that has through-holes cut into it (the
        board's lid), identified as the watertight part with negative genus.

        :raises ValueError: If no such part exists.
        """
        for body in mesh.split(only_watertight=False):
            if body.is_watertight and body.euler_number < 2:
                return body
        raise ValueError("No part of the board mesh has holes cut into it.")
