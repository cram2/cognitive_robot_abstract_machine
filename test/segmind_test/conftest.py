"""
The world SegMind's tests watch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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


@dataclass
class BoxAndTrays:
    """
    A world of its own with a box free to move and two trays standing on the ground -
    each a floor with four walls - the second named as a hole, so an insertion can be
    made into it.
    """

    tray_width: float = 0.3
    """
    The edge length of a square tray.
    """

    tray_wall_height: float = 0.2
    """
    How high a tray's walls stand above the ground.
    """

    tray_wall_thickness: float = 0.01
    """
    How thick a tray's floor and walls are.
    """

    box_size: float = 0.05
    """
    The edge length of the box a test puts into a tray.
    """

    world: World = field(init=False)
    """
    The world holding the box and the trays.
    """

    box: Body = field(init=False)
    """
    The box, free to move, set aside clear of both trays.
    """

    tray: Body = field(init=False)
    """
    The tray standing at the origin.
    """

    tray_hole: Body = field(init=False)
    """
    The tray named as a hole, standing at :attr:`hole_x`.
    """

    def __post_init__(self):
        self.world = World()
        root = Body(name=PrefixedName("root"))
        self.box = Body(
            name=PrefixedName("box"),
            collision=ShapeCollection(
                [Box(scale=Scale(self.box_size, self.box_size, self.box_size))]
            ),
        )
        with self.world.modify_world():
            self.world.add_kinematic_structure_entity(root)
            self.world.add_connection(
                Connection6DoF.create_with_dofs(
                    world=self.world, parent=root, child=self.box
                )
            )
            self.tray = self._add_tray("tray", 0.0)
            self.tray_hole = self._add_tray("tray_hole", self.hole_x)
        self.box.parent_connection.origin = self.set_aside

    @property
    def hole_x(self) -> float:
        """
        Where along the x axis the tray named as a hole stands, clear of the other one.
        """
        return 2 * self.tray_width

    @property
    def set_aside(self) -> HomogeneousTransformationMatrix:
        """
        Where the box stands when the trays are built, clear of both.
        """
        return self._box_at(-1.0, 0.0, self.box_size / 2)

    @property
    def set_down_in_the_tray(self) -> HomogeneousTransformationMatrix:
        """
        Where the box stands on the tray's floor.
        """
        return self._box_at(0.0, 0.0, self.tray_wall_thickness + self.box_size / 2)

    @property
    def set_down_in_the_hole(self) -> HomogeneousTransformationMatrix:
        """
        Where the box stands on the floor of the tray named as a hole.
        """
        return self._box_at(
            self.hole_x, 0.0, self.tray_wall_thickness + self.box_size / 2
        )

    @property
    def held_up_in_the_tray(self) -> HomogeneousTransformationMatrix:
        """
        Where the box hangs inside the tray's walls, a box's height clear of its floor.
        """
        return self._box_at(0.0, 0.0, self.tray_wall_thickness + 1.5 * self.box_size)

    @property
    def lifted_out_of_the_tray(self) -> HomogeneousTransformationMatrix:
        """
        Where the box hangs high above the tray, clear of its walls.
        """
        return self._box_at(0.0, 0.0, 2 * self.tray_wall_height)

    def _box_at(self, x: float, y: float, z: float) -> HomogeneousTransformationMatrix:
        """
        :return: The origin that puts the box's centre at the given point.
        """
        return HomogeneousTransformationMatrix.from_xyz_rpy(
            x, y, z, reference_frame=self.box.parent_connection.parent
        )

    def _add_tray(self, name: str, x: float) -> Body:
        """
        Stand a tray on the ground, centred at ``x`` along the x axis.

        :param name: The tray's name.
        :param x: Where it stands.
        :return: The tray.
        """
        walls_across_x = [
            Box(
                scale=Scale(
                    self.tray_wall_thickness, self.tray_width, self.tray_wall_height
                ),
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=side * self.tray_width / 2, z=self.tray_wall_height / 2
                ),
            )
            for side in (-1, 1)
        ]
        walls_across_y = [
            Box(
                scale=Scale(
                    self.tray_width, self.tray_wall_thickness, self.tray_wall_height
                ),
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=side * self.tray_width / 2, z=self.tray_wall_height / 2
                ),
            )
            for side in (-1, 1)
        ]
        floor = Box(
            scale=Scale(self.tray_width, self.tray_width, self.tray_wall_thickness),
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=self.tray_wall_thickness / 2
            ),
        )
        tray = Body(
            name=PrefixedName(name),
            collision=ShapeCollection([floor, *walls_across_x, *walls_across_y]),
        )
        self.world.add_connection(
            FixedConnection(
                parent=self.world.root,
                child=tray,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=x, reference_frame=self.world.root
                ),
            )
        )
        return tray


@pytest.fixture
def box_and_trays() -> BoxAndTrays:
    return BoxAndTrays()


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
