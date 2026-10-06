"""
The world SegMind's tests watch.
"""

from __future__ import annotations

import numpy as np
import pytest

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

RESTING_ON_THE_TABLE = (-1.7, 0.0, 0.93)
"""
Where the milk stands resting on the apartment's second box.
"""

WHERE_THE_MILK_STOOD = (-1.7, 0.0, 1.07)
"""
Where the milk stands when the apartment is built, which a test puts it back to.
"""

TRAY_WIDTH = 0.3
"""
The edge length of a square tray.
"""

TRAY_WALL_HEIGHT = 0.2
"""
How high a tray's walls stand above the ground.
"""

TRAY_WALL_THICKNESS = 0.01
"""
How thick a tray's floor and walls are.
"""

BOX_SIZE = 0.05
"""
The edge length of the box a test puts into a tray.
"""

HOLE_X = 2 * TRAY_WIDTH
"""
Where along the x axis the tray named as a hole stands; the other tray stands at the
origin.
"""

SET_ASIDE = (-1.0, 0.0, BOX_SIZE / 2)
"""
Where the box stands when the trays are built, clear of both.
"""

SET_DOWN_IN_THE_TRAY = (0.0, 0.0, TRAY_WALL_THICKNESS + BOX_SIZE / 2)
"""
Where the box stands on the tray's floor.
"""

SET_DOWN_IN_THE_HOLE = (HOLE_X, 0.0, TRAY_WALL_THICKNESS + BOX_SIZE / 2)
"""
Where the box stands on the floor of the tray named as a hole.
"""

HELD_UP_IN_THE_TRAY = (0.0, 0.0, TRAY_WALL_THICKNESS + 1.5 * BOX_SIZE)
"""
Where the box hangs inside the tray's walls, a box's height clear of its floor.
"""

LIFTED_OUT_OF_THE_TRAY = (0.0, 0.0, 2 * TRAY_WALL_HEIGHT)
"""
Where the box hangs high above the tray, clear of its walls.
"""


@pytest.fixture
def milk_in_the_apartment(_simple_apartment_setup):
    """
    The apartment with its milk and boxes, the milk put back where it stood afterwards.
    """
    world = _simple_apartment_setup
    milk = world.get_body_by_name("milk.stl")
    yield world, milk, world.get_body_by_name("box")
    x, y, z = WHERE_THE_MILK_STOOD
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x, y, z, yaw=np.pi, reference_frame=milk.parent_connection.parent
    )


@pytest.fixture
def box_and_trays():
    """
    A world of its own with a box free to move and two trays standing on the ground -
    each a floor with four walls - the second named as a hole, so an insertion can be
    made into it.
    """
    world = World()
    root = Body(name=PrefixedName("root"))
    box = Body(
        name=PrefixedName("box"),
        collision=ShapeCollection([Box(scale=Scale(BOX_SIZE, BOX_SIZE, BOX_SIZE))]),
    )
    trays = []
    with world.modify_world():
        world.add_kinematic_structure_entity(root)
        world.add_connection(
            Connection6DoF.create_with_dofs(world=world, parent=root, child=box)
        )
        for name, x in (("tray", 0.0), ("tray_hole", HOLE_X)):
            walls_across_x = [
                Box(
                    scale=Scale(TRAY_WALL_THICKNESS, TRAY_WIDTH, TRAY_WALL_HEIGHT),
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=side * TRAY_WIDTH / 2, z=TRAY_WALL_HEIGHT / 2
                    ),
                )
                for side in (-1, 1)
            ]
            walls_across_y = [
                Box(
                    scale=Scale(TRAY_WIDTH, TRAY_WALL_THICKNESS, TRAY_WALL_HEIGHT),
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                        y=side * TRAY_WIDTH / 2, z=TRAY_WALL_HEIGHT / 2
                    ),
                )
                for side in (-1, 1)
            ]
            floor = Box(
                scale=Scale(TRAY_WIDTH, TRAY_WIDTH, TRAY_WALL_THICKNESS),
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=TRAY_WALL_THICKNESS / 2
                ),
            )
            tray = Body(
                name=PrefixedName(name),
                collision=ShapeCollection([floor, *walls_across_x, *walls_across_y]),
            )
            world.add_connection(
                FixedConnection(
                    parent=root,
                    child=tray,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=x, reference_frame=root
                    ),
                )
            )
            trays.append(tray)
    box.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        *SET_ASIDE, reference_frame=root
    )
    return world, box, *trays


@pytest.fixture
def milk_annotated_in_the_apartment(milk_in_the_apartment):
    """
    The apartment with its milk annotated as :class:`Milk`, the annotation removed
    afterwards.
    """
    world, milk, box = milk_in_the_apartment
    annotation = Milk(root=milk)
    with world.modify_world():
        world.add_semantic_annotation(annotation)
    yield world, annotation, box
    with world.modify_world():
        world.remove_semantic_annotation(annotation)
