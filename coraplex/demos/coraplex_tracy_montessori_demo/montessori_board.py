"""
The Montessori shape-sorting board: its mesh, the holes cut through its lid, and the
body that stands it in a world.

The holes are not written down as constants. They are read off the board's own mesh by
slicing through its lid, so the holes a piece is sorted into are the holes the board
actually has.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import List, Tuple

import numpy as np
import trimesh

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Box, Color, Mesh, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% the board's mesh

BOARD_MESH_PATH = Path(__file__).parent / "resources" / "board.stl"
"""
The board's mesh: a lid with six holes cut through it over three open compartments, the
drawers' places, which open towards the robot along the board's -x.
"""

BOARD_COLOR = Color(1.0, 0.32, 0.0)
"""
Colour the board is drawn in, the orange measured off the real board (hue 19).
"""

DRAWER_COLOR = Color(0.85, 0.27, 0.0)
"""
Colour the drawer fronts are drawn in, a shade darker than the board so they read apart.
"""

HANDLE_COLOR = Color(0.5, 0.5, 0.5)
"""
Colour the drawer handles are drawn in.
"""

DRAWER_FRONT_THICKNESS = 0.006
"""
How thick a drawer's front is, filling the gap the mesh leaves in front of each
compartment.
"""

HANDLE_SCALE = Scale(0.015, 0.03, 0.015)
"""
Size of a drawer's handle.
"""

_MINIMUM_CELL_SIZE = 1e-4
"""
Narrowest cell the lid's collision grid keeps; anything narrower is a sliver where two
hole edges almost coincide.
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


_CIRCLE_VERTEX_COUNT = 20
"""
A circle is tessellated into far more boundary points than a straight-edged hole (65
against 9 to 13 in the mesh); a loop with more points than this is a circle.
"""

_TRIANGLE_FILL_RATIO = 0.7
"""
A triangle fills at most half of its bounding box, a box-shaped hole all of it.
"""

_SLOT_ASPECT_RATIO = 5.0
"""
The slot is about ten times as long as it is wide; every other hole is far squarer.
"""

_RECTANGLE_ASPECT_RATIO = 1.3
"""
A box-shaped hole this much longer than wide is a rectangle rather than a square.
"""


@dataclass(frozen=True)
class BoardHole:
    """
    One hole cut through the board's lid, in the board's own frame.
    """

    shape: HoleShape
    """
    What the hole is shaped like.
    """

    center: Tuple[float, float]
    """
    Where the hole's area balances, which for the triangle is not the middle of its
    bounding box.
    """

    bounds: Tuple[float, float, float, float]
    """
    The hole's bounding box, as minimum x, maximum x, minimum y, maximum y.
    """

    @property
    def bounding_box_center(self) -> Tuple[float, float]:
        """
        The middle of the hole's bounding box, which a piece centred on its own bounding
        box is released over.
        """
        minimum_x, maximum_x, minimum_y, maximum_y = self.bounds
        return (minimum_x + maximum_x) / 2, (minimum_y + maximum_y) / 2


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

    holes: List[BoardHole]
    """
    Every hole cut through the lid.
    """

    solid_parts: List[trimesh.Trimesh]
    """
    Every part of the mesh other than the lid: the base, the walls and the compartment
    dividers, each of them a box.
    """

    lid: trimesh.Trimesh
    """
    The lid, the one part with holes through it.
    """

    @classmethod
    def from_mesh(cls, path: Path = BOARD_MESH_PATH) -> BoardGeometry:
        """
        Read the board off its mesh.

        :param path: The mesh file.
        :return: The board's geometry.
        """
        parts = trimesh.load(path).split(only_watertight=False)
        [lid] = [part for part in parts if part.euler_number < 2]
        return cls(
            lid_top=float(lid.bounds[1][2]),
            bottom=float(min(part.bounds[0][2] for part in parts)),
            holes=_holes_through(lid),
            solid_parts=[part for part in parts if part is not lid],
            lid=lid,
        )

    @property
    def height(self) -> float:
        """
        How tall the board stands, from its underside to the top of its lid.
        """
        return self.lid_top - self.bottom

    def hole_of_shape(self, shape: HoleShape) -> BoardHole:
        """
        The first hole of a shape, in the order the holes are read off the mesh.

        :param shape: The shape asked for.
        :return: That hole.
        """
        return next(hole for hole in self.holes if hole.shape is shape)

    def compartments(self) -> List[Tuple[float, float]]:
        """
        The compartments under the lid, as the stretch along y each spans between two
        walls or dividers, from -y to +y.
        """
        dividers = sorted(
            (float(part.bounds[0][1]), float(part.bounds[1][1]))
            for part in self.solid_parts
            if _extent(part, 0) > _extent(part, 1) and _extent(part, 2) > 0.01
        )
        return [
            (upper_of_left, lower_of_right)
            for (_, upper_of_left), (lower_of_right, _) in zip(dividers, dividers[1:])
        ]


def _extent(part: trimesh.Trimesh, axis: int) -> float:
    """
    How far a part reaches along one axis.
    """
    return float(part.bounds[1][axis] - part.bounds[0][axis])


def _holes_through(lid: trimesh.Trimesh) -> List[BoardHole]:
    """
    Slice the lid half way through its thickness and read every inner loop as a hole.

    :param lid: The lid.
    :return: The holes, from -y to +y.
    """
    middle = (lid.bounds[0][2] + lid.bounds[1][2]) / 2
    section = lid.section(plane_origin=[0.0, 0.0, middle], plane_normal=[0.0, 0.0, 1.0])
    loops = [np.asarray(loop)[:, :2] for loop in section.discrete]
    outer = max(loops, key=lambda loop: np.prod(loop.max(axis=0) - loop.min(axis=0)))

    holes = []
    for loop in loops:
        if loop is outer:
            continue
        minimum, maximum = loop.min(axis=0), loop.max(axis=0)
        size_x, size_y = maximum - minimum
        area, centroid = _area_and_centroid(loop)
        holes.append(
            BoardHole(
                shape=_shape_of(
                    len(loop),
                    area / (size_x * size_y),
                    max(size_x, size_y) / min(size_x, size_y),
                ),
                center=centroid,
                bounds=(
                    float(minimum[0]),
                    float(maximum[0]),
                    float(minimum[1]),
                    float(maximum[1]),
                ),
            )
        )
    return sorted(holes, key=lambda hole: hole.center[1])


def _area_and_centroid(loop: np.ndarray) -> Tuple[float, Tuple[float, float]]:
    """
    The area a closed polygon encloses and the point it balances about, by the shoelace
    formula.

    :param loop: Ordered boundary points, shape ``(n, 2)``.
    """
    x, y = loop[:, 0], loop[:, 1]
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)
    cross = x * y_next - x_next * y
    signed_area = cross.sum() / 2
    return abs(float(signed_area)), (
        float(((x + x_next) * cross).sum() / (6 * signed_area)),
        float(((y + y_next) * cross).sum() / (6 * signed_area)),
    )


def _shape_of(point_count: int, fill_ratio: float, aspect_ratio: float) -> HoleShape:
    """
    Tell a hole's shape from how its outline was tessellated, how much of its bounding
    box it fills, and how elongated that box is.
    """
    if point_count > _CIRCLE_VERTEX_COUNT:
        return HoleShape.CIRCLE
    if fill_ratio < _TRIANGLE_FILL_RATIO:
        return HoleShape.TRIANGLE
    if aspect_ratio > _SLOT_ASPECT_RATIO:
        return HoleShape.SLOT
    if aspect_ratio > _RECTANGLE_ASPECT_RATIO:
        return HoleShape.RECTANGLE
    return HoleShape.SQUARE


# %% the board as a body


def board_body(name: PrefixedName, geometry: BoardGeometry) -> Body:
    """
    Build the board's body: its mesh to look at, and boxes to collide with.

    MuJoCo collides with the convex hull of a mesh, which would close every hole, so the
    collision geometry is built from boxes instead: every solid part of the mesh is a box
    already, and the lid is tiled into boxes around its holes. The compartments are
    closed towards the robot by a drawer front each, with a handle.

    :param name: What the body is called.
    :param geometry: The board, as read off its mesh.
    :return: The body.
    """
    drawer_fronts = _drawer_fronts(geometry)
    handles = _handles(geometry)
    return Body(
        name=name,
        visual=ShapeCollection(
            [Mesh.from_file(str(BOARD_MESH_PATH), color=BOARD_COLOR)]
            + drawer_fronts
            + handles
        ),
        collision=ShapeCollection(
            [_box_around(part) for part in geometry.solid_parts]
            + _lid_around_its_holes(geometry)
            + [_copy_of(front) for front in drawer_fronts]
        ),
    )


def _box_around(part: trimesh.Trimesh) -> Box:
    """
    The box a part of the mesh is.
    """
    minimum, maximum = part.bounds
    center = (minimum + maximum) / 2
    return Box(
        scale=Scale(*(maximum - minimum)),
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(*center),
        color=BOARD_COLOR,
    )


def _lid_around_its_holes(geometry: BoardGeometry) -> List[Box]:
    """
    Tile the lid into boxes, leaving the bounding box of every hole open.

    The lid is cut into a grid at every edge of every hole's bounding box; a grid cell
    either lies inside a hole and stays open, or outside all of them and becomes a box.

    :param geometry: The board.
    :return: The lid's boxes.
    """
    (minimum_x, minimum_y, bottom), (maximum_x, maximum_y, top) = geometry.lid.bounds
    x_edges = sorted(
        {minimum_x, maximum_x}
        | {hole.bounds[0] for hole in geometry.holes}
        | {hole.bounds[1] for hole in geometry.holes}
    )
    y_edges = sorted(
        {minimum_y, maximum_y}
        | {hole.bounds[2] for hole in geometry.holes}
        | {hole.bounds[3] for hole in geometry.holes}
    )
    boxes = []
    for x0, x1 in zip(x_edges, x_edges[1:]):
        if x1 - x0 < _MINIMUM_CELL_SIZE:
            continue
        for y0, y1 in zip(y_edges, y_edges[1:]):
            if y1 - y0 < _MINIMUM_CELL_SIZE:
                continue
            cell_x, cell_y = (x0 + x1) / 2, (y0 + y1) / 2
            if any(
                hole.bounds[0] < cell_x < hole.bounds[1]
                and hole.bounds[2] < cell_y < hole.bounds[3]
                for hole in geometry.holes
            ):
                continue
            boxes.append(
                Box(
                    scale=Scale(x1 - x0, y1 - y0, top - bottom),
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                        cell_x, cell_y, (top + bottom) / 2
                    ),
                    color=BOARD_COLOR,
                )
            )
    return boxes


def _drawer_fronts(geometry: BoardGeometry) -> List[Box]:
    """
    A drawer front closing each compartment on the robot's side.
    """
    lid_bottom = float(geometry.lid.bounds[0][2])
    base = max(
        geometry.solid_parts, key=lambda part: _extent(part, 0) * _extent(part, 1)
    )
    base_top = float(base.bounds[1][2])
    front_x = float(geometry.lid.bounds[0][0]) + DRAWER_FRONT_THICKNESS / 2
    return [
        Box(
            scale=Scale(DRAWER_FRONT_THICKNESS, right - left, lid_bottom - base_top),
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                front_x, (left + right) / 2, (lid_bottom + base_top) / 2
            ),
            color=DRAWER_COLOR,
        )
        for left, right in geometry.compartments()
    ]


def _handles(geometry: BoardGeometry) -> List[Box]:
    """
    A handle in the middle of each drawer front.

    Handles are only drawn: an arm reaching past the board catches on them otherwise, and
    nothing in the demo opens a drawer.
    """
    handle_x = float(geometry.lid.bounds[0][0]) - HANDLE_SCALE.x / 2
    lid_bottom = float(geometry.lid.bounds[0][2])
    return [
        Box(
            scale=HANDLE_SCALE,
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                handle_x, (left + right) / 2, (lid_bottom + geometry.bottom) / 2
            ),
            color=HANDLE_COLOR,
        )
        for left, right in geometry.compartments()
    ]


def _copy_of(box: Box) -> Box:
    """
    A second box like the given one, so the visual and the collision geometry do not
    share a shape.
    """
    return Box(scale=box.scale, origin=box.origin, color=box.color)
