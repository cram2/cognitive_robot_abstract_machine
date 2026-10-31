from __future__ import annotations

import abc
from abc import ABC
from dataclasses import dataclass, field

from sqlalchemy import select

from giskardpy.model.exceptions import RobotDescriptionMissingError
from krrood.ormatic.data_access_objects.helper import get_data_access_object_class
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.orm.utils import semantic_digital_twin_sessionmaker
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    OmniDrive,
    WheeledDrive,
)
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class WorldConfig(ABC):
    world: World = field(default_factory=World)

    @abc.abstractmethod
    def setup_world(self, *args, **kwargs):
        """
        Implement this method to configure the initial world using it's self.

        methods.
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
class WorldWithRobot(WorldConfig, ABC):
    """
    A world holding one robot that is described by a URDF.
    """

    urdf: str | None = field(kw_only=True, default=None)
    """
    The URDF of the robot as XML, not a path to it.

    ``None`` loads the description file :attr:`robot_class` declares.
    """

    root_name: PrefixedName = field(default=PrefixedName("map"))
    """
    Name of the root body of the world.
    """

    robot_class: type[AbstractRobot] = field(kw_only=True, default=MinimalRobot)
    """
    The robot class that is applied to the parsed description.
    """

    def parse_robot(self) -> World:
        """
        :return: A world of its own holding the described robot, with the robot class
            applied to it.
        """
        world_with_robot = self.create_urdf_parser().parse()
        self.robot_class.from_world(world_with_robot)
        return world_with_robot

    def create_urdf_parser(self) -> URDFParser:
        """
        :return: A parser of :attr:`urdf`, or of the description file
            :attr:`robot_class` declares when no URDF is given.
        :raises RobotDescriptionMissingError: If no URDF is given and the robot class
            declares no description file.
        """
        if self.urdf is not None:
            return URDFParser(urdf=self.urdf, prefix="")
        description_file_path = self.robot_class.get_ros_file_path()
        if not description_file_path:
            raise RobotDescriptionMissingError(robot_class=self.robot_class)
        return URDFParser.from_file(description_file_path, prefix="")


@dataclass
class WorldWithFixedRobot(WorldWithRobot):
    """
    A world whose robot is fixed to the root body.
    """

    def setup_world(self):
        world_with_robot = self.parse_robot()

        root = Body(name=self.root_name)
        self.world.add_body(root)
        root_C_robot = FixedConnection(parent=root, child=world_with_robot.root)

        self.world.merge_world(world_with_robot, root_C_robot)


@dataclass
class WorldWithMobileRobot(WorldWithRobot):
    """
    A world whose robot drives relative to an odometry body, which is itself free to
    move relative to the root body.
    """

    odometry_body_name: PrefixedName = field(default=PrefixedName("odom"))
    """
    Name of the body the robot drives relative to.
    """

    drive_class: type[WheeledDrive] = field(kw_only=True, default=OmniDrive)
    """
    The kind of connection between the odometry body and the robot.
    """

    translation_velocity_limit: float = field(kw_only=True, default=0.2)
    """
    Velocity limit of the translation of the drive.
    """

    rotation_velocity_limit: float = field(kw_only=True, default=0.2)
    """
    Velocity limit of the rotation of the drive.
    """

    def setup_world(self):
        world_with_robot = self.parse_robot()

        root = Body(name=self.root_name)
        odometry_body = Body(name=self.odometry_body_name)
        localization = Connection6DoF.create_with_dofs(
            parent=root, child=odometry_body, world=self.world
        )
        self.world.add_connection(localization)

        drive = self.drive_class.create_with_dofs(
            parent=odometry_body,
            child=world_with_robot.root,
            translation_velocity_limits=self.translation_velocity_limit,
            rotation_velocity_limits=self.rotation_velocity_limit,
            world=self.world,
        )
        drive.has_hardware_interface = True

        self.world.merge_world(world_with_robot, drive)


@dataclass
class WorldFromDatabaseConfig(WorldConfig):
    """
    This world config loads a world from the semantic digital twin database at the given
    primary key.
    """

    primary_key: int = 1
    """
    Primary key of the world in the semantic digital twin database.
    """

    def setup_world(self):
        ormatic_world_class = get_data_access_object_class(World)
        session = semantic_digital_twin_sessionmaker()()
        world_dao = session.scalar(
            select(ormatic_world_class).where(
                ormatic_world_class.database_id == self.primary_key
            )
        )
        self.world = world_dao.from_dao()
