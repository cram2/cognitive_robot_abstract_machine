"""
The Montessori shape-sorting board: its mesh, the holes cut through its lid, and the
body that stands it in a world.

The holes are not written down as constants. They are read off the board's own mesh by
slicing through its lid, so the holes a piece is sorted into are the holes the board
actually has.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, StrEnum
from pathlib import Path

import numpy as np
import trimesh

from random_events.interval import Bound, SimpleInterval

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point2
from semantic_digital_twin.world_description.geometry import (
    Box,
    Color,
    Mesh,
    PlanarBoundingBox,
    Scale,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% the board's files


class BoardResource(Enum):
    """
    The files the board is built from.
    """

    MESH = Path(__file__).parent / "resources" / "board.stl"
    """
    The board's mesh: a lid with six holes cut through it over three open compartments,
    the drawers' places, which open towards the robot along the board's -x.
    """


# %% what a hole is shaped like


class HoleShape(StrEnum):
    """
    The outline of a hole cut through the board's lid, which says which piece fits it.
    """

    SQUARE = "square"
    RECTANGLE = "rectangle"
    TRIANGLE = "triangle"
    CIRCLE = "circle"
    SLOT = "slot"
    """
    The narrow slot a disk is posted through.
    """


@dataclass(frozen=True)
class HoleOutline:
    """
    The closed outline of a hole where the lid is sliced through, in the board's own
    frame.
    """

    points: np.ndarray
    """
    The ordered boundary points, shape ``(n, 2)``.
    """

    @property
    def bounds(self) -> PlanarBoundingBox:
        """
        The outline's bounding box.
        """
        (minimum_x, minimum_y), (maximum_x, maximum_y) = (
            self.points.min(axis=0),
            self.points.max(axis=0),
        )
        return PlanarBoundingBox(
            float(minimum_x),
            float(minimum_y),
            float(maximum_x),
            float(maximum_y),
            HomogeneousTransformationMatrix(),
        )

    @property
    def signed_area(self) -> float:
        """
        The area the outline encloses, by the shoelace formula, positive when its points
        run counter-clockwise.
        """
        x, y = self.points[:, 0], self.points[:, 1]
        return float((x * np.roll(y, -1) - np.roll(x, -1) * y).sum() / 2)

    @property
    def centroid(self) -> Point2:
        """
        The point the enclosed area balances about, which for a triangle is not the
        middle of its bounding box.
        """
        x, y = self.points[:, 0], self.points[:, 1]
        x_next, y_next = np.roll(x, -1), np.roll(y, -1)
        cross = x * y_next - x_next * y
        return Point2(
            float(((x + x_next) * cross).sum() / (6 * self.signed_area)),
            float(((y + y_next) * cross).sum() / (6 * self.signed_area)),
        )

    @property
    def fill_ratio(self) -> float:
        """
        How much of its bounding box the outline fills.
        """
        return abs(self.signed_area) / self.bounds.area

    @property
    def aspect_ratio(self) -> float:
        """
        How many times longer the outline's bounding box is than wide.
        """
        return max(self.bounds.dimensions) / min(self.bounds.dimensions)


@dataclass(frozen=True)
class HoleShapeClassifier:
    """
    Tells a hole's shape from how its outline was tessellated, how much of its bounding
    box it fills, and how elongated that box is.
    """

    circle_point_count: int = 20
    """
    An outline with more points than this is a circle: a circle is tessellated into far
    more boundary points than a straight-edged hole (65 against 9 to 13 in the mesh).
    """

    triangle_fill_ratio: float = 0.7
    """
    An outline filling less of its bounding box than this is a triangle: a triangle fills
    at most half of it, a box-shaped hole all of it.
    """

    slot_aspect_ratio: float = 5.0
    """
    An outline more elongated than this is the slot, which is about ten times as long as
    it is wide; every other hole is far squarer.
    """

    rectangle_aspect_ratio: float = 1.3
    """
    A box-shaped outline more elongated than this is a rectangle rather than a square.
    """

    def shape_of(self, outline: HoleOutline) -> HoleShape:
        """
        :param outline: The outline of a hole.
        :return: What the hole is shaped like.
        """
        if len(outline.points) > self.circle_point_count:
            return HoleShape.CIRCLE
        if outline.fill_ratio < self.triangle_fill_ratio:
            return HoleShape.TRIANGLE
        if outline.aspect_ratio > self.slot_aspect_ratio:
            return HoleShape.SLOT
        if outline.aspect_ratio > self.rectangle_aspect_ratio:
            return HoleShape.RECTANGLE
        return HoleShape.SQUARE


@dataclass(frozen=True)
class BoardHole:
    """
    One hole cut through the board's lid, in the board's own frame.
    """

    shape: HoleShape
    """
    What the hole is shaped like.
    """

    center: Point2
    """
    Where the hole's area balances, which for the triangle is not the middle of its
    bounding box.
    """

    bounds: PlanarBoundingBox
    """
    The hole's bounding box.
    """


# %% the board's geometry


@dataclass(frozen=True)
class BoardGeometry:
    """
    Everything the board's mesh says about the board.
    """

    lid_top: float
    """
    Height of the lid's top surface above the board's own origin.
    """

    bottom: float
    """
    Height of the board's underside above its own origin, which is negative.
    """

    holes: list[BoardHole]
    """
    Every hole cut through the lid, from -y to +y.
    """

    solid_parts: list[trimesh.Trimesh]
    """
    Every part of the mesh other than the lid: the base, the walls and the compartment
    dividers, each of them a box.
    """

    lid: trimesh.Trimesh
    """
    The lid, the one part with holes through it.
    """

    @classmethod
    def from_mesh(
        cls,
        path: Path = BoardResource.MESH.value,
        classifier: HoleShapeClassifier = HoleShapeClassifier(),
    ) -> BoardGeometry:
        """
        Read the board off its mesh.

        :param path: The mesh file.
        :param classifier: What tells the holes' shapes apart.
        :return: The board's geometry.
        """
        parts = trimesh.load(path).split(only_watertight=False)
        [lid] = [part for part in parts if part.euler_number < 2]
        return cls(
            lid_top=float(lid.bounds[1][2]),
            bottom=float(min(part.bounds[0][2] for part in parts)),
            holes=cls.holes_through(lid, classifier),
            solid_parts=[part for part in parts if part is not lid],
            lid=lid,
        )

    @staticmethod
    def holes_through(
        lid: trimesh.Trimesh, classifier: HoleShapeClassifier
    ) -> list[BoardHole]:
        """
        Slice the lid half way through its thickness and read every inner loop as a hole.

        :param lid: The lid.
        :param classifier: What tells the holes' shapes apart.
        :return: The holes, from -y to +y.
        """
        middle = (lid.bounds[0][2] + lid.bounds[1][2]) / 2
        section = lid.section(
            plane_origin=[0.0, 0.0, middle], plane_normal=[0.0, 0.0, 1.0]
        )
        outlines = [HoleOutline(np.asarray(loop)[:, :2]) for loop in section.discrete]
        rim = max(
            outlines,
            key=lambda outline: outline.bounds.area,
        )
        holes = [
            BoardHole(
                shape=classifier.shape_of(outline),
                center=outline.centroid,
                bounds=outline.bounds,
            )
            for outline in outlines
            if outline is not rim
        ]
        return sorted(holes, key=lambda hole: float(hole.center.y))

    @property
    def height(self) -> float:
        """
        How tall the board stands, from its underside to the top of its lid.
        """
        return self.lid_top - self.bottom

    @property
    def lid_bottom(self) -> float:
        """
        Height of the lid's underside above the board's own origin.
        """
        return float(self.lid.bounds[0][2])

    @property
    def front(self) -> float:
        """
        Where the lid ends on the robot's side, along the board's x.
        """
        return float(self.lid.bounds[0][0])

    @property
    def base(self) -> trimesh.Trimesh:
        """
        The part the board stands on: the solid part with the largest footprint.
        """
        return max(self.solid_parts, key=lambda part: part.extents[0] * part.extents[1])

    def hole_of_shape(self, shape: HoleShape) -> BoardHole:
        """
        :param shape: The shape asked for.
        :return: The first hole of ``shape``, in the order the holes are read off the
            mesh.
        """
        return next(hole for hole in self.holes if hole.shape is shape)

    def compartments(self) -> list[SimpleInterval]:
        """
        :return: The stretch along y each compartment under the lid spans, between two
            walls or dividers, from -y to +y.
        """
        dividers = sorted(
            (float(part.bounds[0][1]), float(part.bounds[1][1]))
            for part in self.solid_parts
            if part.extents[0] > part.extents[1] and part.extents[2] > 0.01
        )
        return [
            SimpleInterval.from_data(
                upper_of_left, lower_of_right, Bound.CLOSED, Bound.CLOSED
            )
            for (_, upper_of_left), (lower_of_right, _) in zip(dividers, dividers[1:])
        ]


# %% the board as a body


@dataclass
class ShapeSortingBoard:
    """
    The board as a body: its mesh to look at, and boxes to collide with.

    MuJoCo collides with the convex hull of a mesh, which would close every hole, so the
    collision geometry is built from boxes instead: every solid part of the mesh is a box
    already, and the lid is tiled into boxes around its holes. The compartments are
    closed towards the robot by a drawer front each, with a handle.
    """

    geometry: BoardGeometry = field(default_factory=BoardGeometry.from_mesh)
    """
    The board, as read off its mesh.
    """

    color: Color = field(default_factory=lambda: Color(1.0, 0.32, 0.0))
    """
    Colour the board is drawn in, the orange measured off the real board (hue 19).
    """

    drawer_color: Color = field(default_factory=lambda: Color(0.85, 0.27, 0.0))
    """
    Colour the drawer fronts are drawn in, a shade darker than the board so they read
    apart.
    """

    handle_color: Color = field(default_factory=lambda: Color(0.5, 0.5, 0.5))
    """
    Colour the drawer handles are drawn in.
    """

    drawer_front_thickness: float = 0.006
    """
    How thick a drawer's front is, filling the gap the mesh leaves in front of each
    compartment.
    """

    handle_scale: Scale = field(default_factory=lambda: Scale(0.015, 0.03, 0.015))
    """
    Size of a drawer's handle.
    """

    minimum_cell_size: float = 1e-4
    """
    Narrowest cell the lid's collision grid keeps; anything narrower is a sliver where two
    hole edges almost coincide.
    """

    def body(self, name: PrefixedName) -> Body:
        """
        :param name: What the body is called.
        :return: The board's body.
        """
        drawer_fronts = self.drawer_fronts()
        return Body(
            name=name,
            visual=ShapeCollection(
                [Mesh.from_file(str(BoardResource.MESH.value), color=self.color)]
                + drawer_fronts
                + self.handles()
            ),
            collision=ShapeCollection(
                [self.box_around(part) for part in self.geometry.solid_parts]
                + self.lid_around_its_holes()
                + [self.copy_of(front) for front in drawer_fronts]
            ),
        )

    def box_around(self, part: trimesh.Trimesh) -> Box:
        """
        :param part: A part of the mesh that is a box.
        :return: That box.
        """
        minimum, maximum = part.bounds
        return Box(
            scale=Scale(*part.extents),
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                *(minimum + maximum) / 2
            ),
            color=self.color,
        )

    def lid_around_its_holes(self) -> list[Box]:
        """
        Tile the lid into boxes, leaving the bounding box of every hole open.

        The lid is cut into a grid at every edge of every hole's bounding box; a grid cell
        either lies inside a hole and stays open, or outside all of them and becomes a box.

        :return: The lid's boxes.
        """
        lid, holes = self.geometry.lid, self.geometry.holes
        (minimum_x, minimum_y, bottom), (maximum_x, maximum_y, top) = lid.bounds
        x_edges = sorted(
            {minimum_x, maximum_x}
            | {hole.bounds.min_x for hole in holes}
            | {hole.bounds.max_x for hole in holes}
        )
        y_edges = sorted(
            {minimum_y, maximum_y}
            | {hole.bounds.min_y for hole in holes}
            | {hole.bounds.max_y for hole in holes}
        )
        cells = [
            PlanarBoundingBox(x0, y0, x1, y1, HomogeneousTransformationMatrix())
            for x0, x1 in zip(x_edges, x_edges[1:])
            if x1 - x0 >= self.minimum_cell_size
            for y0, y1 in zip(y_edges, y_edges[1:])
            if y1 - y0 >= self.minimum_cell_size
        ]
        return [
            Box(
                scale=Scale(cell.depth, cell.width, top - bottom),
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    cell.center.x, cell.center.y, (top + bottom) / 2
                ),
                color=self.color,
            )
            for cell in cells
            if not any(
                hole.bounds.simple_event.contains(
                    (float(cell.center.x), float(cell.center.y))
                )
                for hole in holes
            )
        ]

    def drawer_fronts(self) -> list[Box]:
        """
        :return: A drawer front closing each compartment on the robot's side.
        """
        base_top = float(self.geometry.base.bounds[1][2])
        return [
            Box(
                scale=Scale(
                    self.drawer_front_thickness,
                    compartment.size,
                    self.geometry.lid_bottom - base_top,
                ),
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    self.geometry.front + self.drawer_front_thickness / 2,
                    compartment.center(),
                    (self.geometry.lid_bottom + base_top) / 2,
                ),
                color=self.drawer_color,
            )
            for compartment in self.geometry.compartments()
        ]

    def handles(self) -> list[Box]:
        """
        A handle in the middle of each drawer front.

        Handles are only drawn: an arm reaching past the board catches on them otherwise,
        and nothing in the demo opens a drawer.

        :return: The handles.
        """
        return [
            Box(
                scale=self.handle_scale,
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    self.geometry.front - self.handle_scale.x / 2,
                    compartment.center(),
                    (self.geometry.lid_bottom + self.geometry.bottom) / 2,
                ),
                color=self.handle_color,
            )
            for compartment in self.geometry.compartments()
        ]

    @staticmethod
    def copy_of(box: Box) -> Box:
        """
        :param box: A box of the board's visual geometry.
        :return: A second box like ``box``, so the visual and the collision geometry do
            not share a shape.
        """
        return Box(scale=box.scale, origin=box.origin, color=box.color)
