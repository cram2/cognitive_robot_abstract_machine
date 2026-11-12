"""
The Montessori shape-sorting board as a body to stand in the demo's world, built from
the board's mesh and the holes read off it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import trimesh

from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.shape_sorting_toy.board import (
    BoardResource,
    ShapeSortingBoardMesh,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    Color,
    Mesh,
    PlanarBoundingBox,
    Scale,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% the board as a body


@dataclass(kw_only=True)
class ShapeSortingBoardModel:
    """
    The board as a body: its mesh to look at, and boxes to collide with.

    MuJoCo collides with the convex hull of a mesh, which would close every hole, so the
    collision geometry is built from boxes instead: every solid part of the mesh is a box
    already, and the lid is tiled into boxes around its holes. The compartments are
    closed towards the robot by a drawer front each, with a handle.
    """

    mesh: ShapeSortingBoardMesh = field(default_factory=ShapeSortingBoardMesh)
    """
    The board, as read off its mesh.
    """

    color: Color = field(
        default_factory=lambda: Color.from_rgba(red=1.0, green=0.32, blue=0.0)
    )
    """
    Colour the board is drawn in, the orange measured off the real board (hue 19).
    """

    drawer_color: Color = field(
        default_factory=lambda: Color.from_rgba(red=0.85, green=0.27, blue=0.0)
    )
    """
    Colour the drawer fronts are drawn in, a shade darker than the board so they read
    apart.
    """

    handle_color: Color = field(
        default_factory=lambda: Color.from_rgba(red=0.5, green=0.5, blue=0.5)
    )
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
    Narrowest cell the lid's collision grid keeps; anything narrower is a sliver where
    two hole edges almost coincide.
    """

    def body(self, *, name: PrefixedName) -> Body:
        """
        :param name: What the body is called.
        :return: The board's body.
        """
        drawer_fronts = self.drawer_fronts()
        return Body(
            name=name,
            visual=ShapeCollection(
                [Mesh.from_file(str(BoardResource.MESH), color=self.color)]
                + drawer_fronts
                + self.handles()
            ),
            collision=ShapeCollection(
                [self.box_around(part=part) for part in self.mesh.solid_parts]
                + self.lid_around_its_holes()
                + [self.copy_of(box=front) for front in drawer_fronts]
            ),
        )

    def box_around(self, *, part: trimesh.Trimesh) -> Box:
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

        The lid is cut into a grid at every edge of every hole's bounding box; a grid
        cell either lies inside a hole and stays open, or outside all of them and
        becomes a box.

        :return: The lid's boxes.
        """
        (minimum_x, minimum_y, bottom), (maximum_x, maximum_y, top) = (
            self.mesh.lid.bounds
        )
        holes = [
            footprint.outline_on_board.bounding_box
            for footprint in self.mesh.hole_footprints
        ]
        x_edges = sorted(
            {minimum_x, maximum_x}
            | {hole.min_x for hole in holes}
            | {hole.max_x for hole in holes}
        )
        y_edges = sorted(
            {minimum_y, maximum_y}
            | {hole.min_y for hole in holes}
            | {hole.max_y for hole in holes}
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
                hole.simple_event.contains((float(cell.center.x), float(cell.center.y)))
                for hole in holes
            )
        ]

    def drawer_fronts(self) -> list[Box]:
        """
        :return: A drawer front closing each compartment on the robot's side.
        """
        base_top = float(self.mesh.base.bounds[1][2])
        return [
            Box(
                scale=Scale(
                    self.drawer_front_thickness,
                    compartment.size,
                    self.mesh.lid_bottom - base_top,
                ),
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    self.mesh.front + self.drawer_front_thickness / 2,
                    compartment.center(),
                    (self.mesh.lid_bottom + base_top) / 2,
                ),
                color=self.drawer_color,
            )
            for compartment in self.mesh.compartments()
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
                    self.mesh.front - self.handle_scale.x / 2,
                    compartment.center(),
                    (self.mesh.lid_bottom + self.mesh.bottom) / 2,
                ),
                color=self.handle_color,
            )
            for compartment in self.mesh.compartments()
        ]

    @staticmethod
    def copy_of(*, box: Box) -> Box:
        """
        :param box: A box of the board's visual geometry.
        :return: A second box like ``box``, so the visual and the collision geometry do
            not share a shape.
        """
        return Box(scale=box.scale, origin=box.origin, color=box.color)
