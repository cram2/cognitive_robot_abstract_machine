from dataclasses import dataclass

import pytest
from typing_extensions import Self

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.spatial_types import Vector3, HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    OmniDrive,
    RevoluteConnection,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Cylinder
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    Body,
    CollisionCheckingConfig,
)


@pytest.fixture()
def mini_world():
    world = World()
    with world.modify_world():
        body = Body(name=PrefixedName("root"))
        body2 = Body(name=PrefixedName("tip"))
        connection = RevoluteConnection.create_with_dofs(
            world=world, parent=body, child=body2, axis=Vector3.Z()
        )
        world.add_connection(connection)
    return world


@dataclass
class BoxBot(AbstractRobot):
    """
    Class that describes the Human Support Robot variant B (https://upmroboticclub.wordpress.com/robot/).
    """

    def __hash__(self):
        return hash(
            tuple(
                [self.__class__]
                + sorted([kse.name for kse in self.kinematic_structure_entities])
            )
        )

    def setup_collision_config(self):
        """
        Loads the SRDF file for the PR2 robot, if it exists.
        """
        ...

    @classmethod
    def from_world(cls, world: World) -> Self:
        with world.modify_world():
            boxbot = cls(
                name=PrefixedName("boxbot", prefix=world.name),
                root=world.get_body_by_name("bot"),
                _world=world,
            )
            world.add_semantic_annotation(boxbot)


@pytest.fixture()
def box_bot_world():
    world = World()
    with world.modify_world():
        body = Body(
            name=PrefixedName("map"),
        )
        body2 = Body(
            name=PrefixedName("bot"),
            collision=ShapeCollection(shapes=[Cylinder(width=0.1, height=0.5)]),
            collision_config=CollisionCheckingConfig(
                buffer_zone_distance=0.05, violated_distance=0.0, max_avoided_bodies=3
            ),
        )
        connection = OmniDrive.create_with_dofs(world=world, parent=body, child=body2)
        world.add_connection(connection)
        connection.has_hardware_interface = True

        environment = Body(
            name=PrefixedName("environment"),
            collision=ShapeCollection(shapes=[Cylinder(width=0.1, height=0.5)]),
        )
        env_connection = FixedConnection(
            parent=body,
            child=environment,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                1
            ),
        )
        world.add_connection(env_connection)
        BoxBot.from_world(world)

    return world
