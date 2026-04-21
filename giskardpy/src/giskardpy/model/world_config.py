from __future__ import annotations

import abc
from abc import ABC
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import select

from krrood.ormatic.data_access_objects.helper import get_dao_class
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.orm.utils import semantic_digital_twin_sessionmaker
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.spatial_types.derivatives import Derivatives
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    OmniDrive,
    FixedConnection,
    DifferentialDrive,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)


@dataclass
class WorldConfig(ABC):
    world: World = field(default_factory=World)

    @abc.abstractmethod
    def setup_world(self, *args, **kwargs):
        """
        Implement this method to configure the initial world using it's self. methods.
        """


class EmptyWorld(WorldConfig):
    def setup_world(self):
        # self._default_limits = {
        #     Derivatives.velocity: 1,
        #     Derivatives.acceleration: np.inf,
        #     Derivatives.jerk: None
        # }
        # self.set_default_limits(self._default_limits)
        self.add_empty_link(PrefixedName("map"))


@dataclass
class WorldWithFixedRobot(WorldConfig):
    urdf: str = field(kw_only=True)
    root_name: PrefixedName = field(default=PrefixedName("map"))
    robot_name: PrefixedName = field(default=PrefixedName("robot"))
    robot_root: KinematicStructureEntity = field(init=False)
    urdf_view: AbstractRobot = field(kw_only=True, default=MinimalRobot)

    def setup_world(self):
        map = Body(name=self.root_name)

        urdf_parser = URDFParser(urdf=self.urdf, prefix="")
        world_with_robot = urdf_parser.parse()
        self.urdf_view.from_world(world_with_robot)
        self.robot_root = world_with_robot.root
        map_C_robot = FixedConnection(parent=map, child=self.robot_root)

        self.world.merge_world(world_with_robot, map_C_robot)


@dataclass
class WorldWithOmniDriveRobot(WorldConfig):
    urdf: str = field(kw_only=True)
    root_name: PrefixedName = field(default=PrefixedName("map"))
    robot_name: PrefixedName = field(default=PrefixedName("robot"))
    odom_body_name: PrefixedName = field(default=PrefixedName("odom"))
    urdf_view: AbstractRobot = field(kw_only=True, default=MinimalRobot)
    localization: Connection6DoF = field(init=False)
    robot: AbstractRobot = field(init=False)

    def setup_world(self):
        map = Body(name=self.root_name)
        odom = Body(name=self.odom_body_name)
        self.localization = Connection6DoF.create_with_dofs(
            parent=map, child=odom, world=self.world
        )
        self.world.add_connection(self.localization)

        urdf_parser = URDFParser(urdf=self.urdf, prefix="")
        world_with_robot = urdf_parser.parse()
        self.robot = self.urdf_view.from_world(world_with_robot)

        odom = OmniDrive.create_with_dofs(
            parent=odom,
            child=world_with_robot.root,
            translation_velocity_limits=0.2,
            rotation_velocity_limits=0.2,
            world=self.world,
        )

        self.world.merge_world(world_with_robot, odom)


@dataclass
class WorldWithDiffDriveRobot(WorldConfig):
    urdf: str = field(kw_only=True)
    root_name: PrefixedName = field(default=PrefixedName("map"))
    robot_name: PrefixedName = field(default=PrefixedName("robot"))
    odom_body_name: PrefixedName = field(default=PrefixedName("odom"))
    urdf_view: AbstractRobot = field(kw_only=True, default=MinimalRobot)
    localization: Connection6DoF = field(init=False)
    robot: AbstractRobot = field(init=False)

    def setup_world(self):
        map = Body(name=self.root_name)
        odom = Body(name=self.odom_body_name)
        self.localization = Connection6DoF.create_with_dofs(
            parent=map, child=odom, world=self.world
        )
        self.world.add_connection(self.localization)

        urdf_parser = URDFParser(urdf=self.urdf, prefix="")
        world_with_robot = urdf_parser.parse()
        self.robot = self.urdf_view.from_world(world_with_robot)

        odom = DifferentialDrive.create_with_dofs(
            parent=odom,
            child=world_with_robot.root,
            translation_velocity_limits=0.2,
            rotation_velocity_limits=0.2,
            world=self.world,
        )

        self.world.merge_world(world_with_robot, odom)


@dataclass
class WorldFromDatabaseConfig(WorldConfig):
    """
    This world config loads a world from the semantic digital twin database at the given primary key.
    """

    primary_key: int = 1
    """Primary key of the world in the semantic digital twin database."""

    def setup_collision_config(self):
        pass

    def setup_world(self):
        ormatic_world_class = get_dao_class(World)
        session = semantic_digital_twin_sessionmaker()()
        world_dao = session.scalar(
            select(ormatic_world_class).where(
                ormatic_world_class.database_id == self.primary_key
            )
        )
        self.world = world_dao.from_dao()
