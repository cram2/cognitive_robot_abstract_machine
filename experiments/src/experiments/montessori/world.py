"""
Build a semantic digital twin world for the Montessori shape-sorting scene: a floor
carrying a table with a shape-sorting board (with its holes and drawers) and the loose
shapes that are dropped through the holes, and (optionally) a robot standing in front of
it.

The scene is constructed directly with the semantic digital twin API (bodies, regions,
shapes, connections, and semantic annotations); nothing is loaded from a recorded
episode or any other external dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh
from typing_extensions import Dict, List, Optional, Tuple, Type

from experiments.montessori.hole_geometry import (
    HOLE_MARKER_THICKNESS,
    HoleFootprint,
    cut_board_mesh,
    detect_hole_footprints,
)
from experiments.montessori.pieces import KNOWN_PIECE_BY_CATEGORY
from experiments.montessori.semantics import (
    MONTESSORI_SHAPE_CLASSES,
    MontessoriShapeCategory,
    ShapeSortingBoard,
    ShapeSortingHole,
)
from semantic_digital_twin.adapters.package_resolver import CompositePathResolver
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import PathResolutionError
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootKinematicStructureEntity,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Floor,
    Handle,
    Table,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    OmniDrive,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    Color,
    Cylinder,
    Mesh,
    Scale,
    Shape,
    Sphere,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body, Region

NAME_PREFIX = "montessori"
"""
Prefix given to every :class:`PrefixedName` created by this module.
"""

FLOOR_Z = 0.0
"""
Height of the ground the Montessori scene stands on.
"""

FLOOR_SCALE = Scale(6.0, 4.0, 0.02)
"""
Size of the floor slab; large enough to comfortably fit the table, the row of loose
shapes, and a standing robot.
"""

TABLE_SCALE = Scale(0.5, 1.0, 0.025)
TABLE_POSITION = Point3(-0.35, 0.0, 0.5)

TABLE_LEG_FOOTPRINT = 0.05
"""
Cross-sectional width/depth of each of the table's four legs.
"""

_LEG_CORNER_SIGNS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
"""
The four ``(x, y)`` sign combinations at which a table's legs are placed, relative to
its center.
"""

ROBOT_STAND_SCALE = Scale(0.3, 0.3, 0.025)
"""
Size of the stand a bolted robot arm sits on (see :meth:`MontessoriWorld.add_robot_stand`).

Wide enough to carry an arm's base without reaching the montessori table itself, so the
two never share a footprint.
"""

BOARD_SCALE = Scale(0.11, 0.282, 0.08)
BOARD_POSITION = Point3(-0.4, 0.0, 0.553)
BOARD_COLOR = Color.BEIGE()

DRAWER_SCALE = Scale(0.09, 0.08, 0.06)
HANDLE_SCALE = Scale(0.03, 0.015, 0.015)

LANDING_REGION_XY_MARGIN = 0.03
"""
Margin added on every side of a hole's own footprint size when sizing the
:class:`Region` a shape is checked for containment against once it has fallen through
that hole (see :meth:`MontessoriWorld._build_shape_sorting_board`); comfortably
tolerates placement/settling inaccuracy.
"""

LANDING_REGION_TOP_CLEARANCE = 0.02
"""
Distance kept below the board's top surface when sizing a hole's landing region's top
face, so a shape merely resting on top of the board (having failed to fall through)
never also registers as being in the landing region.
"""

LANDING_REGION_BOTTOM_MARGIN = 0.005
"""
Distance the landing region's bottom face is dropped below the table's own top
surface.

A shape resting on the table sits with its own lowest vertices exactly at the table's
top surface; checking containment with the landing region's bottom face placed exactly
there as well leaves those vertices sitting exactly on the boundary, which floating-point
rounding can push to either side unpredictably. Dropping the boundary below the table's
surface instead keeps them unambiguously inside.
"""

TABLE_SHAPE_ROW_X = -0.15
"""
X-coordinate, in the world frame, of the row in which loose shapes are placed on the
table; clear of the shape-sorting board's footprint.
"""

TABLE_SHAPE_ROW_START_Y = -0.2
"""
Y-coordinate of the first loose shape in the row on the table.
"""

TABLE_SHAPE_ROW_SPACING = 0.07
"""
Distance, along y, between adjacent loose shapes in the row on the table.
"""

DEFAULT_ROBOT_STANDOFF_DISTANCE = 0.6
"""
Default distance the spawned robot stands in front of the Montessori table's near edge.
"""

_SHAPE_COLORS: Dict[MontessoriShapeCategory, Color] = {
    MontessoriShapeCategory.DISK: Color.YELLOW(),
    MontessoriShapeCategory.SPHERE: Color.MAGENTA(),
} | {category: piece.color for category, piece in KNOWN_PIECE_BY_CATEGORY.items()}
"""
The color used to render a loose shape and the hole it fits through, keyed by their
shared :class:`~experiments.montessori.semantics.MontessoriShapeCategory`.

Every category the physical set contains is drawn in the colour measured off the real
piece (see :class:`~experiments.montessori.pieces.KnownPiece`), so the simulated scene
and the camera are looking at the same thing. The disk and the sphere, which this set
has none of, keep a colour of their own.
"""


@dataclass(frozen=True)
class _HoleSpec:
    """
    Layout of a single hole cut into the lid of the shape-sorting board.
    """

    key: str
    category: MontessoriShapeCategory
    position: Point3
    shape: HoleFootprint
    """
    The hole's true, mesh-detected shape; used to build both the board's cut and this
    hole's own marker region.
    """


_HOLE_KEY_BY_CATEGORY = {
    MontessoriShapeCategory.CUBE: "square_hole",
    MontessoriShapeCategory.TRIANGULAR_PRISM: "triangle_hole",
    MontessoriShapeCategory.RECTANGULAR_PRISM: "rectangular_hole",
    MontessoriShapeCategory.DISK: "disk_hole",
}
"""
Name given to a hole of a given category, for the categories that occur at most once on
the board.

The :attr:`~MontessoriShapeCategory.CYLINDER` category occurs twice and is numbered
instead (``circular_hole_1``, ``circular_hole_2``).
"""


def _hole_spec_from_footprint(footprint: HoleFootprint, key: str) -> _HoleSpec:
    """
    Place a mesh-detected :class:`HoleFootprint` onto the board, flush with its top
    surface, and pair it with a semantic key.
    """
    position = Point3(
        BOARD_POSITION.x + footprint.center.x,
        BOARD_POSITION.y + footprint.center.y,
        BOARD_POSITION.z + BOARD_SCALE.z / 2 - HOLE_MARKER_THICKNESS / 2,
    )
    return _HoleSpec(key, footprint.category, position, footprint)


def _build_hole_specs(footprints: List[HoleFootprint]) -> List[_HoleSpec]:
    """
    Build the board's hole specifications from its mesh-detected hole shapes.
    """
    circular_hole_count = 0
    hole_specs = []
    for footprint in footprints:
        if footprint.category is MontessoriShapeCategory.CYLINDER:
            circular_hole_count += 1
            key = f"circular_hole_{circular_hole_count}"
        else:
            key = _HOLE_KEY_BY_CATEGORY[footprint.category]
        hole_specs.append(_hole_spec_from_footprint(footprint, key))
    return hole_specs


_HOLE_FOOTPRINTS: List[HoleFootprint] = detect_hole_footprints()
"""
The board's hole shapes, detected once from its mesh
(:func:`~experiments.montessori.hole_geometry.detect_hole_footprints`); the single
source both :const:`_HOLES` and the board's cut mesh are built from.
"""

_HOLES: List[_HoleSpec] = _build_hole_specs(_HOLE_FOOTPRINTS)
"""
One hole per :class:`~experiments.montessori.semantics.MontessoriShapeCategory` that has
a matching shape (two holes, both circular, accept the
:attr:`~experiments.montessori.semantics.MontessoriShapeCategory.CYLINDER` shape); the
sphere has no matching hole, mirroring the real Montessori board this scene is modelled
after.

Detected from the board's mesh by
:func:`~experiments.montessori.hole_geometry.detect_hole_footprints`.
"""

_BOARD_MESH: trimesh.Trimesh = cut_board_mesh(BOARD_SCALE, _HOLE_FOOTPRINTS)
"""
The shape-sorting board's mesh, with all of :const:`_HOLE_FOOTPRINTS` cut fully through
it.

Used directly as the board's visual geometry.
"""

_MINIMUM_COLLISION_CELL_SIZE = 1e-4
"""
Minimum width or depth (in meters) a board collision grid cell (see
:func:`_board_collision_boxes`) must have to be kept; smaller cells are floating-point
slivers from two hole edges landing almost exactly on top of each other, not real
geometry.
"""


def _footprint_bounds(footprint: HoleFootprint) -> Tuple[float, float, float, float]:
    """
    A hole footprint's true axis-aligned bounding box, in the board's local frame.

    :attr:`HoleFootprint.center` is the polygon's *area* centroid, which for an
    asymmetric outline (e.g. the triangular hole) is not the geometric middle of its own
    bounding box; reading the bounds from the boundary polygon itself, rather than
    assuming ``center +/- size / 2``, keeps this correct for every hole shape.

    :param footprint: The hole to compute bounds for.
    :return: ``(min_x, max_x, min_y, max_y)``.
    """
    boundary_x = [point.x for point in footprint.boundary]
    boundary_y = [point.y for point in footprint.boundary]
    return (
        footprint.center.x + min(boundary_x),
        footprint.center.x + max(boundary_x),
        footprint.center.y + min(boundary_y),
        footprint.center.y + max(boundary_y),
    )


def _tile_footprint_avoiding_holes(
    outer_bounds: Tuple[float, float, float, float],
    hole_bounds: List[Tuple[float, float, float, float]],
    thickness: float,
) -> List[Box]:
    """
    Tile a rectangular footprint into solid collision :class:`Box`\\ es, each spanning
    ``thickness``, leaving every given hole's bounding box entirely open.

    Physics engines that only support convex collision geometry (MuJoCo among them)
    cannot use a single holed mesh directly as collision geometry; the usual fix is a
    convex decomposition of that mesh, but a general-purpose decomposer optimizes for
    volume, not for keeping small or narrow openings (like the ``disk`` category's hole)
    fully clear, so a shape can still get caught on a decomposition artifact instead of
    passing through. Native box primitives, cut exactly around each hole's bounding box,
    avoid that approximation entirely: every hole is either fully open or fully solid,
    with nothing in between to get caught on.

    Splits the footprint into a rectangular grid at every hole's bounding box edge (a
    standard "rectangle minus rectangles" tiling: a grid cell is either entirely inside
    one hole's bounding box, so it is left open, or entirely outside every hole's
    bounding box, so it becomes one solid collision box), rather than assuming any
    particular hole count or layout. A hole outside ``outer_bounds`` is clipped to it
    first, so a hole that only partly overlaps the tiled footprint (e.g. one of the
    board's holes passing through a drawer sitting underneath it, rather than the board
    itself) still cuts exactly the overlapping part open.

    :param outer_bounds: ``(min_x, max_x, min_y, max_y)`` of the footprint to tile, in
        whatever local frame the returned boxes should be positioned in.
    :param hole_bounds: ``(min_x, max_x, min_y, max_y)`` of each hole to leave open, in
        the same frame and units as ``outer_bounds``.
    :param thickness: Extent of every solid box along z, centered on ``z=0``.
    :return: One solid :class:`Box` per occupied grid cell.
    """
    outer_min_x, outer_max_x, outer_min_y, outer_max_y = outer_bounds
    clipped_hole_bounds = [
        (
            min(max(min_x, outer_min_x), outer_max_x),
            min(max(max_x, outer_min_x), outer_max_x),
            min(max(min_y, outer_min_y), outer_max_y),
            min(max(max_y, outer_min_y), outer_max_y),
        )
        for min_x, max_x, min_y, max_y in hole_bounds
    ]
    x_edges = sorted(
        {outer_min_x, outer_max_x}
        | {bound[0] for bound in clipped_hole_bounds}
        | {bound[1] for bound in clipped_hole_bounds}
    )
    y_edges = sorted(
        {outer_min_y, outer_max_y}
        | {bound[2] for bound in clipped_hole_bounds}
        | {bound[3] for bound in clipped_hole_bounds}
    )

    boxes = []
    for x0, x1 in zip(x_edges, x_edges[1:]):
        if x1 - x0 < _MINIMUM_COLLISION_CELL_SIZE:
            continue
        for y0, y1 in zip(y_edges, y_edges[1:]):
            if y1 - y0 < _MINIMUM_COLLISION_CELL_SIZE:
                continue
            cell_x, cell_y = (x0 + x1) / 2, (y0 + y1) / 2
            if any(
                min_x < cell_x < max_x and min_y < cell_y < max_y
                for min_x, max_x, min_y, max_y in clipped_hole_bounds
            ):
                continue
            boxes.append(
                Box(
                    scale=Scale(x1 - x0, y1 - y0, thickness),
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=cell_x, y=cell_y, z=0.0
                    ),
                )
            )
    return boxes


def _board_collision_boxes(
    board_scale: Scale, footprints: List[HoleFootprint]
) -> List[Box]:
    """
    Tile the board's footprint into solid collision boxes, leaving every hole's true
    bounding box entirely open; see :func:`_tile_footprint_avoiding_holes`.

    :param board_scale: Size of the board blank the holes are cut into.
    :param footprints: The holes to leave open.
    :return: One solid :class:`Box` per occupied grid cell.
    """
    half_x, half_y = board_scale.x / 2, board_scale.y / 2
    hole_bounds = [_footprint_bounds(footprint) for footprint in footprints]
    return _tile_footprint_avoiding_holes(
        (-half_x, half_x, -half_y, half_y), hole_bounds, board_scale.z
    )


def _drawer_collision_boxes(
    drawer_scale: Scale,
    drawer_position: Point3,
    board_position: Point3,
    footprints: List[HoleFootprint],
) -> List[Box]:
    """
    Tile a drawer's footprint into solid collision boxes, leaving open whichever part
    (if any) of a hole's shaft passes through it; see
    :func:`_tile_footprint_avoiding_holes`.

    Every hole cuts all the way through the board's full thickness (see
    :func:`~experiments.montessori.hole_geometry.cut_board_mesh`), and a drawer can sit
    anywhere within that thickness, directly behind a hole it has no relation to
    otherwise; without this, a dropped shape could pass the board's own collision
    cleanly and then be stopped by a drawer immediately behind it.

    :param drawer_scale: Size of the drawer.
    :param drawer_position: The drawer's position, in the same frame as
        ``board_position``.
    :param board_position: The board's position; :attr:`HoleFootprint.center` is
        relative to it.
    :param footprints: The holes to leave open wherever they overlap this drawer.
    :return: One solid :class:`Box` per occupied grid cell, in the drawer's own frame.
    """
    half_x, half_y = drawer_scale.x / 2, drawer_scale.y / 2
    offset_x = float(board_position.x) - float(drawer_position.x)
    offset_y = float(board_position.y) - float(drawer_position.y)
    hole_bounds = [
        (
            min_x + offset_x,
            max_x + offset_x,
            min_y + offset_y,
            max_y + offset_y,
        )
        for min_x, max_x, min_y, max_y in (
            _footprint_bounds(footprint) for footprint in footprints
        )
    ]
    return _tile_footprint_avoiding_holes(
        (-half_x, half_x, -half_y, half_y), hole_bounds, drawer_scale.z
    )


_DRAWER_POSITIONS: List[Point3] = [
    Point3(-0.403, 0.087, 0.553),
    Point3(-0.403, 0.0, 0.553),
    Point3(-0.403, -0.087, 0.553),
]

_HANDLE_OFFSET = Point3(-0.061, 0.0, 0.001)
"""
Position of a drawer's handle relative to the drawer.
"""


def _name(name: str) -> PrefixedName:
    return PrefixedName(name, NAME_PREFIX)


def _body_with_shapes(name: PrefixedName, shapes: List[Shape]) -> Body:
    """
    Build a :class:`Body` whose collision and visual geometry are the given shapes.
    """
    return Body.from_shape_collection(name, ShapeCollection(shapes))


def _body_with_shape(name: PrefixedName, shape: Shape) -> Body:
    """
    Build a :class:`Body` whose collision and visual geometry are a single shape.
    """
    return _body_with_shapes(name, [shape])


def _body_with_visual_only_shape(name: PrefixedName, shape: Shape) -> Body:
    """
    Build a :class:`Body` with a visual-only shape and no collision geometry.

    Used for the floor: CRAM's navigation reachability costmaps assume the ground
    itself has no collision mesh, and treat any collidable geometry at ground level
    as an obstacle blocking every standing spot.
    """
    return Body(name=name, visual=ShapeCollection([shape]))


def _board_body(
    name: PrefixedName, board_shape: Mesh, footprints: List[HoleFootprint]
) -> Body:
    """
    Build the shape-sorting board's :class:`Body`: ``board_shape`` as visual geometry,
    and a hand-built grid of solid boxes (see :func:`_board_collision_boxes`) as
    collision geometry, so the cut holes stay physically open to physics engines that
    only support convex collision geometry.

    :param name: Name of the body.
    :param board_shape: The board's (single, concave) visual mesh shape.
    :param footprints: The holes cut into ``board_shape``, kept open in the collision
        geometry.
    """
    return Body(
        name=name,
        visual=ShapeCollection([board_shape]),
        collision=ShapeCollection(_board_collision_boxes(BOARD_SCALE, footprints)),
    )


def _drawer_body(
    name: PrefixedName,
    drawer_scale: Scale,
    color: Color,
    drawer_position: Point3,
    board_position: Point3,
    footprints: List[HoleFootprint],
) -> Body:
    """
    Build a drawer's :class:`Body`: a solid box as visual geometry, and a hand-built
    grid of solid boxes (see :func:`_drawer_collision_boxes`) as collision geometry, so
    the drawer does not block a hole shaft that happens to pass through it.

    :param name: Name of the body.
    :param drawer_scale: Size of the drawer.
    :param color: Color of the drawer's visual box.
    :param drawer_position: The drawer's position, in the same frame as
        ``board_position``.
    :param board_position: The board's position.
    :param footprints: The holes to leave open wherever they overlap this drawer.
    """
    return Body(
        name=name,
        visual=ShapeCollection([Box(scale=drawer_scale, color=color)]),
        collision=ShapeCollection(
            _drawer_collision_boxes(
                drawer_scale, drawer_position, board_position, footprints
            )
        ),
    )


def _table_shapes(
    table_scale: Scale,
    table_center_z: float,
    leg_footprint: float,
    support_z: float,
    color: Color,
) -> List[Shape]:
    """
    Build a tabletop :class:`Box` plus four leg :class:`Box`\\ es that support it from
    ``support_z`` up to its underside, at its four corners.

    :param table_scale: Size of the tabletop.
    :param table_center_z: Height of the tabletop's own center, in the world frame.
    :param leg_footprint: Cross-sectional width/depth of each leg.
    :param support_z: Height of the surface the legs stand on.
    :param color: Color shared by the tabletop and its legs.
    :return: The tabletop shape followed by its four leg shapes, positioned relative to
        the tabletop's own origin.
    """
    half_x = table_scale.x / 2 - leg_footprint / 2
    half_y = table_scale.y / 2 - leg_footprint / 2
    leg_height = table_center_z - table_scale.z / 2 - support_z
    leg_center_local_z = support_z + leg_height / 2 - table_center_z

    shapes = [Box(scale=table_scale, color=color)]
    for sign_x, sign_y in _LEG_CORNER_SIGNS:
        leg_origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=sign_x * half_x, y=sign_y * half_y, z=leg_center_local_z
        )
        shapes.append(
            Box(
                scale=Scale(leg_footprint, leg_footprint, leg_height),
                color=color,
                origin=leg_origin,
            )
        )
    return shapes


SHAPE_FOOTPRINT_CLEARANCE_SCALE = 0.7
"""
In-plane scale applied to a hole's true footprint when deriving the matching loose
shape's cross-section from it, so the shape is a smaller copy of the hole rather than an
exact fit, leaving clearance to actually pass through.

Widened from the original 0.85 (roughly doubling the per-side gap) as a hacky attempt to
make a shape's release pose more forgiving of the small positioning error that otherwise
leaves it resting on the hole's rim instead of falling through. circular_hole_1 has the
smallest true footprint of the six holes, so the same fraction leaves it the least
absolute clearance in mm and was observed to have the worst pass rate (50% over 20 runs)
of any shape. Widening further to 0.6 was tested (5 full runs): circular_hole_1 scored
2/5, no better than the 0.7 baseline, so 0.6 was reverted. circular_hole_1's failure mode
is not explained by clearance alone.
"""


def _footprint_shape_mesh(
    footprint: HoleFootprint, thickness: float, color: Color
) -> Mesh:
    """
    Build a solid :class:`Mesh` whose cross-section is a clearance-scaled copy of a
    hole's true footprint, centered on its own origin.

    Deriving the shape from the same :class:`HoleFootprint` its hole is cut from (rather
    than independently hand-authoring a same-category shape, as done for a hole's
    non-rectangular category, e.g. :attr:`MontessoriShapeCategory.TRIANGULAR_PRISM`)
    keeps the two in the same local orientation by construction, since both are read
    from one detected outline instead of risking two independent authors picking
    different reference orientations for the same nominal shape.

    :param footprint: The hole this shape is sized and oriented after.
    :param thickness: Extrusion depth of the solid along its own z-axis.
    :param color: Color of the resulting shape.
    """
    scale = SHAPE_FOOTPRINT_CLEARANCE_SCALE
    solid = footprint.extrude(thickness)
    solid.apply_transform(np.diag([scale, scale, 1.0, 1.0]))
    mesh = Mesh.from_trimesh(mesh=solid)
    mesh.color = color
    return mesh


def _hole_marker_shape(footprint: HoleFootprint, color: Color) -> Mesh:
    """
    Build a thin :class:`Mesh` matching a hole's true cross-section shape, for its
    :class:`~experiments.montessori.semantics.ShapeSortingHole` region.
    """
    marker = Mesh.from_trimesh(mesh=footprint.extrude(HOLE_MARKER_THICKNESS))
    marker.color = color
    return marker


def _landing_region_height(table_top_z: float, board_top_z: float) -> float:
    """
    Height of a hole's landing region: ``board_top_z``, minus
    :data:`LANDING_REGION_TOP_CLEARANCE`, down to :data:`LANDING_REGION_BOTTOM_MARGIN`
    below ``table_top_z``.

    A shape that has fallen all the way through a hole comes to rest directly on the
    surface carrying the board, inside the open shaft the hole cuts through the board's
    full thickness; a shape that never fell through instead rests on the board's own top
    surface, well above this range.

    Takes the two surfaces' heights as parameters (rather than reading
    :const:`TABLE_POSITION`/:const:`TABLE_SCALE`/:const:`BOARD_POSITION` directly) so
    :mod:`~experiments.montessori.world2`'s differently-positioned board and stand can
    reuse this unchanged.

    :param table_top_z: Height of the surface the board sits on (a table, a stand, ...).
    :param board_top_z: Height of the board's own top surface.
    """
    return (
        board_top_z
        - LANDING_REGION_TOP_CLEARANCE
        - (table_top_z - LANDING_REGION_BOTTOM_MARGIN)
    )


def _landing_region(
    name: PrefixedName, footprint: HoleFootprint, height: float
) -> Region:
    """
    Build the :class:`Region` a shape is checked for containment against once it has
    fallen through the hole ``footprint`` describes.

    A hole's own thin, flush-with-the-top marker region (see :func:`_hole_marker_shape`)
    cannot serve this purpose: it only ever brushes a passing shape's mesh for an
    instant as the shape crosses its plane, both before and after it has actually
    fallen through, so containment checked against it cannot tell "still at the hole"
    from "now resting below it" apart.

    :param name: Name of the resulting region.
    :param footprint: The hole this landing region belongs to; sized after its
        bounding box plus :data:`LANDING_REGION_XY_MARGIN` on every side.
    :param height: This region's extent along z; see :func:`_landing_region_height`.
    """
    box = Box(
        scale=Scale(
            footprint.size.x + 2 * LANDING_REGION_XY_MARGIN,
            footprint.size.y + 2 * LANDING_REGION_XY_MARGIN,
            height,
        )
    )
    return Region(name=name, area=ShapeCollection([box]))


def _landing_region_position(
    hole_position: Point3, table_top_z: float, height: float
) -> Point3:
    """
    Position, in the world root frame, a hole's landing region (see
    :func:`_landing_region`) must be placed at so it spans from
    :data:`LANDING_REGION_BOTTOM_MARGIN` below ``table_top_z`` up to
    :data:`LANDING_REGION_TOP_CLEARANCE` below the board's top surface.

    :param hole_position: The matching hole's own position (shares its ``x``, ``y``).
    :param table_top_z: Height of the surface the board sits on; see
        :func:`_landing_region_height`.
    :param height: The landing region's height; see :func:`_landing_region_height`.
    """
    region_bottom_z = table_top_z - LANDING_REGION_BOTTOM_MARGIN
    return Point3(hole_position.x, hole_position.y, region_bottom_z + height / 2)


def _shape_body(
    name: PrefixedName,
    category: MontessoriShapeCategory,
    footprint: Optional[HoleFootprint],
) -> Body:
    """
    Build the :class:`Body` of a loose Montessori shape, its geometry depending on its
    category.

    :param footprint: The footprint of the hole this shape is meant to be dropped
        through, used to derive the shape's cross-section (see
        :func:`_footprint_shape_mesh`) for every category except disk and sphere: a
        disk fits through its hole at any fixed size and yaw (mirroring the board's
        single hole of that category), and the sphere has no hole at all. A cube's
        cross-section is derived the same way rather than kept at a fixed size, so it
        gets the same :data:`SHAPE_FOOTPRINT_CLEARANCE_SCALE` clearance every other
        footprint-derived shape does instead of whatever clearance the hole's own true
        size happens to leave against a hardcoded constant; unlike the others, its own
        thickness is derived from that same clearance-scaled edge too (not the shared
        fixed thickness the others use), so it comes out an actual cube rather than a
        flat square tile.
    """
    color = _SHAPE_COLORS[category]
    match category:
        case MontessoriShapeCategory.CUBE:
            # A cube's hole footprint is square, so giving it the same thickness its
            # own footprint edge scales down to (rather than the fixed 0.03 every other
            # footprint-derived category uses) makes all three of its edges equal --
            # an actual cube, not a flat square tile.
            cube_edge = footprint.size.x * SHAPE_FOOTPRINT_CLEARANCE_SCALE
            shape = _footprint_shape_mesh(footprint, thickness=cube_edge, color=color)
        case MontessoriShapeCategory.CYLINDER:
            shape = _footprint_shape_mesh(footprint, thickness=0.03, color=color)
        case MontessoriShapeCategory.DISK:
            shape = Cylinder(width=0.044, height=0.004, color=color)
        case MontessoriShapeCategory.SPHERE:
            shape = Sphere(radius=0.02, color=color)
        case MontessoriShapeCategory.RECTANGULAR_PRISM:
            shape = _footprint_shape_mesh(footprint, thickness=0.03, color=color)
        case MontessoriShapeCategory.TRIANGULAR_PRISM:
            shape = _footprint_shape_mesh(footprint, thickness=0.02, color=color)
    return _body_with_shape(name, shape)


def robot_installed(robot_class: Type[AbstractRobot]) -> bool:
    """
    Whether ``robot_class``'s description (its URDF and everything it references via
    ``package://`` URIs) can currently be resolved, e.g. because the ROS package it
    lives in is built and sourced.

    Generic replacement for a robot-specific check like
    :func:`~semantic_digital_twin.utils.hsrb_installed`, so :meth:`MontessoriWorld.spawn_robot`
    is not limited to robots that happen to already have one.

    :param robot_class: The robot to check.
    """
    try:
        CompositePathResolver().resolve(robot_class.get_ros_file_path())
        return True
    except PathResolutionError:
        return False


def mount_stationary_robot(
    world: World,
    robot_class: Type[AbstractRobot],
    robot_world: World,
    mount_position: Point3,
    mount_yaw: float = 0.0,
) -> AbstractRobot:
    """
    Bolt an already-parsed, fixed-base robot into ``world`` at ``mount_position``.

    Takes a parsed world rather than a robot class to parse, unlike
    :meth:`MontessoriWorld.spawn_robot`: a robot whose description is not a ROS package
    (its :meth:`~semantic_digital_twin.robots.robot_parts.AbstractRobot.get_ros_file_path`
    raising) has to be read by its caller, from whichever format it does ship in. Its
    root is attached with a :class:`FixedConnection` rather than ``spawn_robot``'s
    :class:`OmniDrive`: a robot with no mobile base has nothing for an active drive
    connection to move.

    Any actuator ``robot_world`` carries is dropped first: an actuator parsed into one
    world cannot be merged into another. Callers that need actuators add them to the
    merged world afterwards.

    Not tied to :class:`MontessoriWorld`: usable on any world (e.g. a bare scene built
    just to isolate a robot's own pick-and-place mechanics from a larger scene's
    geometry).

    :param world: The world to mount the robot into, modified in place.
    :param robot_class: The robot to read out of the merged world.
    :param robot_world: The parsed robot, consumed by the merge.
    :param mount_position: Where the robot's root is bolted, in ``world``'s root frame.
    :param mount_yaw: Which way the robot is turned to face. A bolted arm has no base to
        move afterwards, so it has to face its whole task from this one pose.
    :return: The mounted robot.
    """
    with robot_world.modify_world():
        for actuator in list(robot_world.actuators):
            robot_world.remove_actuator(actuator)
    with world.modify_world():
        mount = FixedConnection(
            parent=world.root,
            child=robot_world.root,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=mount_position.x,
                y=mount_position.y,
                z=mount_position.z,
                yaw=mount_yaw,
            ),
        )
        world.merge_world(robot_world, mount)
    return robot_class.from_world(world)


@dataclass(eq=False)
class MontessoriWorld:
    """
    The Montessori shape-sorting scene: a semantic digital twin world containing a
    floor, a table carrying a shape-sorting board (with its holes and drawers), and the
    loose shapes dropped through the holes.

    The scene is built as soon as an instance is constructed; use :meth:`spawn_robot`
    afterwards to add a robot, since none is spawned by default.
    """

    shapes_are_movable: bool = False
    """
    Whether the loose shapes are attached with a free (:class:`Connection6DoF`) joint
    rather than welded to the world with a :class:`FixedConnection`.

    Off by default, matching the HSR :mod:`~experiments.montessori.montessori_demo`,
    which places each shape kinematically and only makes it movable for the moment it
    settles under gravity. Turn it on for a robot that grasps and lifts a shape by
    physical contact (see :mod:`~experiments.tracy_experiments.montessori.montessori_demo_mujoco`):
    a welded shape cannot be moved by the gripper at all, so it can be reached and
    pressed against but never picked up.
    """

    world: World = field(init=False, default_factory=World)
    """
    The assembled semantic digital twin world.
    """

    board: ShapeSortingBoard = field(init=False)
    """
    The shape-sorting board spawned into :attr:`world`.
    """

    landing_regions: dict[str, Region] = field(init=False, default_factory=dict)
    """
    Each hole's landing region (see :func:`_landing_region`), keyed by the hole's own
    :attr:`_HoleSpec.key`; the volume a shape rests in once it has fallen through that
    hole, for a containment check to be run against.
    """

    robot: Optional[AbstractRobot] = field(init=False, default=None)
    """
    The robot spawned into :attr:`world` by :meth:`spawn_robot`, or ``None`` if none has
    been spawned.
    """

    def __post_init__(self) -> None:
        root = Body(name=PrefixedName(name="root", prefix="world"))
        with self.world.modify_world():
            self.world.add_kinematic_structure_entity(root)

        with self.world.modify_world():
            self._build_floor_and_table()
            self.board = self._build_shape_sorting_board()
            self._build_shapes()

    def spawn_robot(
        self,
        robot_class: Type[AbstractRobot],
        standoff_distance: float = DEFAULT_ROBOT_STANDOFF_DISTANCE,
    ) -> AbstractRobot:
        """
        Spawn a robot of the given class standing in front of the Montessori table,
        facing it, and store it as :attr:`robot`.

        Attached via an :class:`OmniDrive` (a real, hardware-controlled mobile-base
        connection), not a plain 6DoF join: CRAM's motion planner needs a proper drive
        connection to navigate the robot at all. This assumes ``robot_class`` has a
        drivable mobile base; a robot description without one would need a different
        connection here.

        :param robot_class: The robot to spawn, e.g. :class:`~semantic_digital_twin.robots.hsrb.HSRB`.
        :param standoff_distance: How far in front of the table's near edge the robot
            stands.
        :return: The spawned robot.
        """
        table_bounding_box = (
            self.world.get_body_by_name("table")
            .collision.as_bounding_box_collection_in_frame(self.world.root)
            .bounding_box()
        )
        robot_world = URDFParser.from_file(robot_class.get_ros_file_path()).parse()
        with self.world.modify_world():
            drive = OmniDrive.create_with_dofs(
                parent=self.world.root, child=robot_world.root, world=self.world
            )
            self.world.merge_world(robot_world, drive)
            drive.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
                table_bounding_box.min_x - standoff_distance,
                0.0,
                0.0,
                reference_frame=self.world.root,
            )
        self.robot = robot_class.from_world(self.world)
        return self.robot

    def mount_stationary_robot(
        self,
        robot_class: Type[AbstractRobot],
        robot_world: World,
        mount_position: Point3,
        mount_yaw: float = 0.0,
    ) -> AbstractRobot:
        """
        Bolt an already-parsed, fixed-base robot to the scene at ``mount_position`` and
        store it as :attr:`robot`; see the module-level :func:`mount_stationary_robot`
        this delegates to.

        :param robot_class: The robot to read out of the merged world.
        :param robot_world: The parsed robot, consumed by the merge.
        :param mount_position: Where the robot's root is bolted, in the world root frame.
        :param mount_yaw: Which way the robot is turned to face. A bolted arm has no base
            to move afterwards, so it has to face its whole task from this one pose.
        :return: The mounted robot.
        """
        self.robot = mount_stationary_robot(
            self.world, robot_class, robot_world, mount_position, mount_yaw
        )
        return self.robot

    def add_robot_stand(self, mount_position: Point3) -> Table:
        """
        Build a stand whose top surface sits exactly at ``mount_position``, for a robot
        bolted there by :meth:`mount_stationary_robot` to visibly sit on.

        A bolted robot is held at ``mount_position`` whether or not anything supports it
        there, so without this the scene would show an arm floating above the floor.

        :param mount_position: Where the robot is bolted, in the world root frame.
        :return: The spawned stand.
        """
        top_z = float(mount_position.z)
        stand = Table(
            name=_name("robot_stand"),
            root=_body_with_shapes(
                _name("robot_stand"),
                _table_shapes(
                    ROBOT_STAND_SCALE,
                    top_z - ROBOT_STAND_SCALE.z / 2,
                    TABLE_LEG_FOOTPRINT,
                    FLOOR_Z,
                    BOARD_COLOR,
                ),
            ),
        )
        with self.world.modify_world():
            return self._spawn(
                stand,
                Point3(
                    mount_position.x,
                    mount_position.y,
                    top_z - ROBOT_STAND_SCALE.z / 2,
                ),
            )

    def _spawn(
        self,
        annotation: HasRootKinematicStructureEntity,
        position: Point3,
    ) -> HasRootKinematicStructureEntity:
        """
        Connect a semantic annotation's root entity to the world root at ``position``
        with a fixed connection, and register the annotation with the world.

        :param annotation: The semantic annotation to spawn.
        :param position: The annotation's position, expressed in the world root frame.
        :return: The spawned annotation.
        """
        self.world.add_connection(
            FixedConnection(
                parent=self.world.root,
                child=annotation.root,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=position.x, y=position.y, z=position.z
                ),
            )
        )
        self.world.add_semantic_annotation(annotation)
        return annotation

    def _spawn_region(self, region: Region, position: Point3) -> Region:
        """
        Connect a bare :class:`Region` (one with no owning semantic annotation, e.g. a
        hole's landing region) to the world root at ``position`` with a fixed
        connection.

        :param region: The region to spawn.
        :param position: The region's position, expressed in the world root frame.
        :return: The spawned region.
        """
        self.world.add_connection(
            FixedConnection(
                parent=self.world.root,
                child=region,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=position.x, y=position.y, z=position.z
                ),
            )
        )
        return region

    def _spawn_free_body(
        self,
        annotation: HasRootKinematicStructureEntity,
        position: Point3,
    ) -> HasRootKinematicStructureEntity:
        """
        Like :meth:`_spawn`, but connects the annotation's root to the world root with a
        free (:class:`Connection6DoF`) joint, so gravity and contacts can move it.

        A simulator treats an unjointed body as welded to its parent, so an annotation
        spawned with :meth:`_spawn` cannot be picked up, dropped, or settle under gravity
        at all.

        :param annotation: The semantic annotation to spawn.
        :param position: The annotation's position, expressed in the world root frame.
        :return: The spawned annotation.
        """
        connection = Connection6DoF.create_with_dofs(
            world=self.world,
            parent=self.world.root,
            child=annotation.root,
        )
        self.world.add_connection(connection)
        # Bakes the pose into the connection's own dof values (x, y, z, qx, qy, qz, qw)
        # rather than passing it as create_with_dofs' parent_T_connection_expression:
        # MuJoCo export only reads a Connection6DoF's dof values into the free joint's
        # keyframe qpos, which is MuJoCo's absolute world pose for a free joint, not a
        # fixed offset applied on top of it. Passing it as parent_T_connection_expression
        # leaves the dofs at create_with_dofs' identity default, so MuJoCo starts the
        # body at the world origin regardless of where it was spawned.
        connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=position.x, y=position.y, z=position.z, reference_frame=self.world.root
        )
        self.world.add_semantic_annotation(annotation)
        return annotation

    def _build_floor_and_table(self) -> None:
        floor = Floor(
            name=_name("floor"),
            root=_body_with_visual_only_shape(
                _name("floor"), Box(scale=FLOOR_SCALE, color=Color.GREY())
            ),
        )
        self._spawn(floor, Point3(0.0, 0.0, FLOOR_Z - FLOOR_SCALE.z / 2))

        table = Table(
            name=_name("table"),
            root=_body_with_shapes(
                _name("table"),
                _table_shapes(
                    TABLE_SCALE,
                    float(TABLE_POSITION.z),
                    TABLE_LEG_FOOTPRINT,
                    FLOOR_Z,
                    BOARD_COLOR,
                ),
            ),
        )
        self._spawn(table, TABLE_POSITION)

    def _build_shape_sorting_board(self) -> ShapeSortingBoard:
        board_shape = Mesh.from_trimesh(mesh=_BOARD_MESH)
        board_shape.color = BOARD_COLOR
        board = ShapeSortingBoard(
            name=_name("board"),
            root=_board_body(_name("board"), board_shape, _HOLE_FOOTPRINTS),
        )
        self._spawn(board, BOARD_POSITION)

        table_top_z = float(TABLE_POSITION.z) + TABLE_SCALE.z / 2
        board_top_z = float(BOARD_POSITION.z) + BOARD_SCALE.z / 2
        landing_region_height = _landing_region_height(table_top_z, board_top_z)
        for hole_spec in _HOLES:
            hole = ShapeSortingHole(
                name=_name(hole_spec.key),
                root=Region(
                    name=_name(hole_spec.key),
                    area=ShapeCollection(
                        [
                            _hole_marker_shape(
                                hole_spec.shape, _SHAPE_COLORS[hole_spec.category]
                            )
                        ]
                    ),
                ),
                shape_category=hole_spec.category,
            )
            self._spawn(hole, hole_spec.position)
            board.add(hole)

            landing_region = _landing_region(
                _name(f"{hole_spec.key}_landing_region"),
                hole_spec.shape,
                landing_region_height,
            )
            self._spawn_region(
                landing_region,
                _landing_region_position(
                    hole_spec.position, table_top_z, landing_region_height
                ),
            )
            self.landing_regions[hole_spec.key] = landing_region

        for index, drawer_position in enumerate(_DRAWER_POSITIONS, start=1):
            drawer = Drawer(
                name=_name(f"drawer_{index}"),
                root=_drawer_body(
                    _name(f"drawer_{index}"),
                    DRAWER_SCALE,
                    BOARD_COLOR,
                    drawer_position,
                    BOARD_POSITION,
                    _HOLE_FOOTPRINTS,
                ),
            )
            self._spawn(drawer, drawer_position)
            board.add(drawer)

            handle = Handle(
                name=_name(f"drawer_{index}_handle"),
                root=_body_with_shape(
                    _name(f"drawer_{index}_handle"),
                    Box(scale=HANDLE_SCALE, color=Color.GREY()),
                ),
            )
            handle_position = Point3(
                drawer_position.x + _HANDLE_OFFSET.x,
                drawer_position.y + _HANDLE_OFFSET.y,
                drawer_position.z + _HANDLE_OFFSET.z,
            )
            self._spawn(handle, handle_position)
            drawer.add(handle)

        return board

    def _build_shapes(self) -> None:
        categories = [hole_spec.category for hole_spec in _HOLES] + [
            MontessoriShapeCategory.SPHERE
        ]
        keys = [hole_spec.key for hole_spec in _HOLES] + ["sphere"]
        footprints = [hole_spec.shape for hole_spec in _HOLES] + [None]

        for index, (key, category, footprint) in enumerate(
            zip(keys, categories, footprints)
        ):
            shape_key = f"{key}_shape"
            body = _shape_body(_name(shape_key), category, footprint)
            shape_class = MONTESSORI_SHAPE_CLASSES[category]
            shape = shape_class(name=_name(shape_key), root=body)
            y = TABLE_SHAPE_ROW_START_Y + index * TABLE_SHAPE_ROW_SPACING
            spawn = self._spawn_free_body if self.shapes_are_movable else self._spawn
            spawn(shape, self._resting_position_on_table(body, y))

    @staticmethod
    def _resting_position_on_table(body: Body, y: float) -> Point3:
        """
        Position, at ``y`` along :const:`TABLE_SHAPE_ROW_X`, at which ``body`` rests
        exactly on the table's surface, given its own local geometry.
        """
        lowest_local_z = body.collision.combined_mesh.bounds[0][2]
        table_top_z = float(TABLE_POSITION.z) + TABLE_SCALE.z / 2
        return Point3(TABLE_SHAPE_ROW_X, y, table_top_z - lowest_local_z)
