"""
The shape-sorting board's mesh, and the holes read off it by slicing through its
lid, so the holes a piece is sorted into are the holes the board actually has.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import trimesh
from random_events.interval import Bound, SimpleInterval

from krrood.patterns.path_enumeration import PathEnumeration
from krrood.patterns.role import Role
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.shape_sorting_toy.exceptions import NoPerforatedPart
from semantic_digital_twin.shape_sorting_toy.semantic_annotations import (
    CubePiece,
    CylinderPiece,
    DiskPiece,
    RectangularPrismPiece,
    ShapeSortingHole,
    ShapeSortingPiece,
    SpherePiece,
    TriangularPrismPiece,
)
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    RegionSpecification,
)
from semantic_digital_twin.specifications.semantic_annotations import (
    SemanticAnnotationWithRootSpecification,
)
from semantic_digital_twin.world_description.geometry import Mesh, Polygon2D, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection

# %% the board's files


class BoardResource(PathEnumeration):
    """
    The files the board is built from.
    """

    MESH = Path(__file__).parent / "resources" / "board.stl"
    """
    The board's mesh: a lid with six holes cut through it over three open compartments,
    the drawers' places, which open along the board's -x.
    """


# %% what a hole is cut for


@dataclass(frozen=True, kw_only=True)
class HoleShapeClassifier:
    """
    Names the kind of piece a hole is cut for from the outline of its cross-section.
    """

    circle_vertex_count: int = 20
    """
    An outline with more corners than this is a circle: a circle is tessellated into far
    more corners than a straight-edged hole (65 against 9 to 13 in the mesh).
    """

    triangle_fill_ratio: float = 0.7
    """
    An outline filling less of its bounding box than this is a triangle: a triangle fills
    at most half of it, a box-shaped hole all of it.
    """

    disk_aspect_ratio: float = 5.0
    """
    An outline more elongated than this is the disk's slot, which is about ten times as
    long as it is wide; every other hole is far squarer.
    """

    rectangle_aspect_ratio: float = 1.3
    """
    A box-shaped outline more elongated than this is a rectangle rather than a square.
    """

    sphere_hole_width: float = 0.036
    """
    A circular hole wider than this, in metres, is the sphere's rather than the
    cylinder's: the board's two circular holes are 32 and 40 millimetres across.
    """

    def classify(self, *, outline: Polygon2D) -> type[ShapeSortingPiece]:
        """
        :param outline: A hole's cross-section.
        :return: The kind of piece a hole of this cross-section is cut for.
        """
        if len(outline.vertices) > self.circle_vertex_count:
            if min(outline.bounding_box.dimensions) > self.sphere_hole_width:
                return SpherePiece
            return CylinderPiece
        if outline.fill_ratio < self.triangle_fill_ratio:
            return TriangularPrismPiece
        if outline.aspect_ratio > self.disk_aspect_ratio:
            return DiskPiece
        if outline.aspect_ratio > self.rectangle_aspect_ratio:
            return RectangularPrismPiece
        return CubePiece


# %% the board's mesh


@dataclass(eq=False, kw_only=True)
class ShapeSortingBoardMesh(Role[Mesh]):
    """
    A mesh in its role as a shape-sorting board's: what it says about the board's lid,
    the holes through the lid and the solid parts under it.
    """

    hole_classifier: HoleShapeClassifier = field(default_factory=HoleShapeClassifier)
    """
    Names the kind of piece each hole read off the mesh is cut for.
    """

    minimum_divider_height: float = 0.01
    """
    A solid part longer along x than along y counts as a wall or divider between two
    compartments only when it is taller than this.
    """

    hole_marker_thickness: float = 0.005
    """
    Thickness along the board's z-axis of the region that marks a hole's outline.
    """

    @cached_property
    def parts(self) -> list[trimesh.Trimesh]:
        """
        Every connected part of the mesh.
        """
        return self.mesh.split(only_watertight=False)

    @cached_property
    def lid(self) -> trimesh.Trimesh:
        """
        The lid: the one watertight part with holes cut through it.

        :raises NoPerforatedPart: If no part of the mesh has holes through it.
        """
        for part in self.parts:
            if part.is_watertight and part.euler_number < 2:
                return part
        raise NoPerforatedPart(mesh_filename=self.filename)

    @property
    def solid_parts(self) -> list[trimesh.Trimesh]:
        """
        Every part of the mesh other than the lid: the base, the walls and the
        compartment dividers, each of them a box.
        """
        return [part for part in self.parts if part is not self.lid]

    @cached_property
    def hole_outlines(self) -> list[Polygon2D]:
        """
        The outlines of the holes through the lid in the board's own frame, from -y to
        +y: the inner outlines of a slice half way through the lid's thickness.
        """
        middle = (self.lid.bounds[0][2] + self.lid.bounds[1][2]) / 2
        section = self.lid.section(
            plane_origin=[0.0, 0.0, middle], plane_normal=[0.0, 0.0, 1.0]
        )
        outlines = [
            Polygon2D(vertices=np.asarray(loop)[:, :2]) for loop in section.discrete
        ]
        rim = max(outlines, key=lambda outline: outline.bounding_box.area)
        holes = [outline for outline in outlines if outline is not rim]
        return sorted(holes, key=lambda outline: float(outline.centroid.y))

    def hole_specifications(
        self,
    ) -> list[SemanticAnnotationWithRootSpecification[ShapeSortingHole]]:
        """
        :return: A hole for each of :attr:`hole_outlines`, in the same order and named
            after its kind of piece and its place in that order: a region of the hole's
            outline, :attr:`hole_marker_thickness` thick, centred where the lid's top
            surface crosses the hole's centroid.
        """
        specifications = []
        for index, outline in enumerate(self.hole_outlines):
            piece_kind = self.hole_classifier.classify(outline=outline)
            name = f"{piece_kind.__name__}_hole_{index}"
            region = RegionSpecification(
                name=name,
                shapes=ShapeCollection(
                    [
                        Mesh.from_trimesh(
                            mesh=outline.centered().extrude(
                                thickness=self.hole_marker_thickness
                            )
                        )
                    ]
                ),
                parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    float(outline.centroid.x), float(outline.centroid.y), self.lid_top
                ),
            )
            specifications.append(
                SemanticAnnotationWithRootSpecification(
                    name=name,
                    semantic_annotation_type=ShapeSortingHole,
                    root_specification=region,
                    annotation_kwargs={"piece_kind": piece_kind},
                )
            )
        return specifications

    @property
    def lid_top(self) -> float:
        """
        Height of the lid's top surface above the board's own origin.
        """
        return float(self.lid.bounds[1][2])

    @property
    def lid_bottom(self) -> float:
        """
        Height of the lid's underside above the board's own origin.
        """
        return float(self.lid.bounds[0][2])

    @property
    def bottom(self) -> float:
        """
        Height of the board's underside above its own origin, which is negative.
        """
        return float(min(part.bounds[0][2] for part in self.parts))

    @property
    def height(self) -> float:
        """
        How tall the board stands, from its underside to the top of its lid.
        """
        return self.lid_top - self.bottom

    @property
    def front(self) -> float:
        """
        Where the lid ends on the drawers' side, along the board's x.
        """
        return float(self.lid.bounds[0][0])

    @property
    def base(self) -> trimesh.Trimesh:
        """
        The part the board stands on: the solid part with the largest footprint.
        """
        return max(self.solid_parts, key=lambda part: part.extents[0] * part.extents[1])

    def compartments(self) -> list[SimpleInterval]:
        """
        :return: The stretch along y each compartment under the lid spans, between two
            walls or dividers, from -y to +y.
        """
        dividers = sorted(
            (float(part.bounds[0][1]), float(part.bounds[1][1]))
            for part in self.solid_parts
            if part.extents[0] > part.extents[1]
            and part.extents[2] > self.minimum_divider_height
        )
        return [
            SimpleInterval.from_data(
                upper_of_left, lower_of_right, Bound.CLOSED, Bound.CLOSED
            )
            for (_, upper_of_left), (lower_of_right, _) in zip(dividers, dividers[1:])
        ]

    def cut_blank(self, *, blank_scale: Scale) -> trimesh.Trimesh:
        """
        :param blank_scale: Size of the uncut board blank.
        :return: A solid board blank with every one of :attr:`hole_outlines` cut all
            the way through it.
        """
        board = trimesh.creation.box(
            extents=(blank_scale.x, blank_scale.y, blank_scale.z)
        )
        for outline in self.hole_outlines:
            board = board.difference(
                outline.extrude(thickness=blank_scale.z * 2), engine=None
            )
        return board
