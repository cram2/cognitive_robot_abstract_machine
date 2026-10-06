"""
Scenes for tests about containment: trays - a floor with four walls - standing in a world,
and a box free to move that a test sets down in one, holds up inside one, or lifts out.
"""

from __future__ import annotations

from typing_extensions import Tuple

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

TRAY_WIDTH = 0.3
"""
The edge length, in metres, of a square tray.
"""

TRAY_WALL_HEIGHT = 0.2
"""
How high, in metres, a tray's walls stand above the ground.
"""

TRAY_WALL_THICKNESS = 0.01
"""
How thick, in metres, a tray's floor and walls are.
"""

BOX_SIZE = 0.05
"""
The edge length, in metres, of the box a test puts into a tray.
"""

HELD_ABOVE_THE_FLOOR = 0.05
"""
How far, in metres, a box held up inside a tray hangs above its floor.
"""

LIFTED_ABOVE_THE_TRAY = 0.5
"""
How high, in metres, above a tray's walls a box lifted out of it hangs.
"""

SET_ASIDE_X = -1.0
"""
Where, in metres along the x axis, a box stands before a test moves it, clear of every
tray a test stands at or beyond the origin.
"""


def world_with_a_box() -> Tuple[World, Body]:
    """
    :return: A world holding only its root and a box free to move, set aside from where
        trays stand.
    """
    world = World()
    root = Body(name=PrefixedName("root"))
    box = Body(
        name=PrefixedName("box"),
        collision=ShapeCollection([Box(scale=Scale(BOX_SIZE, BOX_SIZE, BOX_SIZE))]),
    )
    with world.modify_world():
        world.add_kinematic_structure_entity(root)
        world.add_connection(
            Connection6DoF.create_with_dofs(world=world, parent=root, child=box)
        )
    box.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=SET_ASIDE_X, reference_frame=root
    )
    return world, box


def add_tray(world: World, name: str, x: float = 0.0) -> Body:
    """
    Stand a tray on the ground of ``world``, ``x`` metres along the x axis.

    :return: The tray.
    """
    half_width = TRAY_WIDTH / 2
    wall_middle = TRAY_WALL_HEIGHT / 2
    floor = Box(
        scale=Scale(TRAY_WIDTH, TRAY_WIDTH, TRAY_WALL_THICKNESS),
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(z=TRAY_WALL_THICKNESS / 2),
    )
    walls_across_x = [
        Box(
            scale=Scale(TRAY_WALL_THICKNESS, TRAY_WIDTH, TRAY_WALL_HEIGHT),
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=side * half_width, z=wall_middle
            ),
        )
        for side in (-1, 1)
    ]
    walls_across_y = [
        Box(
            scale=Scale(TRAY_WIDTH, TRAY_WALL_THICKNESS, TRAY_WALL_HEIGHT),
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                y=side * half_width, z=wall_middle
            ),
        )
        for side in (-1, 1)
    ]
    tray = Body(
        name=PrefixedName(name),
        collision=ShapeCollection([floor, *walls_across_x, *walls_across_y]),
    )
    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=tray,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=x, reference_frame=world.root
                ),
            )
        )
    return tray


def _put_box_over(box: Body, tray: Body, bottom: float) -> None:
    """
    Stand ``box`` over the middle of ``tray`` with its bottom ``bottom`` metres above
    the ground.
    """
    x, y, _ = tray.global_pose.to_position().to_np()[:3]
    box.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x, y, bottom + BOX_SIZE / 2, reference_frame=box.parent_connection.parent
    )


def set_down_in(box: Body, tray: Body) -> None:
    """
    Stand ``box`` on the floor of ``tray``.
    """
    _put_box_over(box, tray, TRAY_WALL_THICKNESS)


def hold_up_in(box: Body, tray: Body) -> None:
    """
    Hold ``box`` inside the walls of ``tray``, clear of its floor.
    """
    _put_box_over(box, tray, TRAY_WALL_THICKNESS + HELD_ABOVE_THE_FLOOR)


def lift_out_of(box: Body, tray: Body) -> None:
    """
    Hold ``box`` high above ``tray``, clear of its walls.
    """
    _put_box_over(box, tray, TRAY_WALL_HEIGHT + LIFTED_ABOVE_THE_TRAY)
