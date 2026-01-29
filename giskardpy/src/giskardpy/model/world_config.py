from __future__ import annotations

import abc
import os
from abc import ABC
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import select

from krrood.ormatic.utils import create_engine
from sqlalchemy.orm import sessionmaker
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.orm.ormatic_interface import WorldMappingDAO
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.spatial_types.derivatives import Derivatives
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    OmniDrive,
    FixedConnection,
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

    @abc.abstractmethod
    def setup_collision_config(self):
        """
        This method is called after the robot is connected to the controller and connections have
        the controlled flag.
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


class WorldWithDiffDriveRobot(WorldConfig):
    map_name: str
    localization_joint_name: str
    odom_link_name: str
    drive_joint_name: str

    def __init__(
        self,
        urdf: str,
        map_name: str = "map",
        localization_joint_name: str = "localization",
        odom_link_name: str = "odom",
        drive_joint_name: str = "brumbrum",
    ):
        super().__init__()
        self.urdf = urdf
        self.map_name = map_name
        self.localization_joint_name = localization_joint_name
        self.odom_link_name = odom_link_name
        self.drive_joint_name = drive_joint_name

    def setup_world(self):
        self.set_default_limits(
            {
                Derivatives.velocity: 1,
                Derivatives.acceleration: np.inf,
                Derivatives.jerk: None,
            }
        )
        self.add_empty_link(PrefixedName(self.map_name))
        self.add_empty_link(PrefixedName(self.odom_link_name))
        self.add_6dof_joint(
            parent_link=self.map_name,
            child_link=self.odom_link_name,
            joint_name=self.localization_joint_name,
        )
        self.add_robot_urdf(urdf=self.urdf)
        root_link_name = self.get_root_link_of_group(self.robot_group_name)
        self.add_diff_drive_joint(
            name=self.drive_joint_name,
            parent_link_name=self.odom_link_name,
            child_link_name=root_link_name,
            translation_limits={
                Derivatives.velocity: 0.2,
                Derivatives.acceleration: np.inf,
                Derivatives.jerk: None,
            },
            rotation_limits={
                Derivatives.velocity: 0.2,
                Derivatives.acceleration: np.inf,
                Derivatives.jerk: None,
            },
            robot_group_name=self.robot_group_name,
        )


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
        semantic_digital_twin_database_uri = os.environ.get(
            "SEMANTIC_DIGITAL_TWIN_DATABASE_URI"
        )
        assert (
            semantic_digital_twin_database_uri is not None
        ), "Please set the SEMANTIC_DIGITAL_TWIN_DATABASE_URI environment variable."

        engine = create_engine(semantic_digital_twin_database_uri)
        session = sessionmaker(bind=engine)()
        world_dao = session.scalar(
            select(WorldMappingDAO).where(
                WorldMappingDAO.database_id == self.primary_key
            )
        )
        self.world = world_dao.from_dao()
