"""
Specifications of robots placed into a world.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING, cast

from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.exceptions import StartPoseWithoutMobileBaseError
from semantic_digital_twin.robots.robot_parts import HasMobileBase
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_source import WorldModelParser

if TYPE_CHECKING:
    from semantic_digital_twin.robots.robot_parts import AbstractRobot

# %% robot specifications


@dataclass
class RobotSpecification:
    """
    World-independent description of a robot placed into a world: which robot, where it
    is localized, and where a robot with a mobile base starts relative to its odometry
    frame.

    Materialized via :meth:`spawn`, which merges a robot with a mobile base as
    ``world.root -> localization -> odometry -> drive -> robot`` and any other robot as
    ``world.root -> localization -> robot``.
    """

    semantic_annotation_type: type[AbstractRobot]
    """
    The robot to create.
    """

    description_parser: WorldModelParser | None = None
    """
    The parser reading the robot's description, whose settings, such as the prefix of
    the names it reads, the robot keeps.

    If None, the robot is read from its own description with
    :meth:`~semantic_digital_twin.robots.robot_parts.AbstractRobot.from_description`.
    """

    localization_pose: HomogeneousTransformationMatrix | None = None
    """
    The pose, in the ``world.root`` frame, of the body the localization attaches: the
    odometry body of a robot with a mobile base, and the root of the robot's description
    otherwise.

    If None, identity is used.
    """

    odometry_T_robot_start: HomogeneousTransformationMatrix | None = None
    """
    The start pose of a robot with a mobile base in the frame of its odometry body.

    If None, identity is used.
    """

    localization_connection_type: type[FixedConnection | Connection6DoF] = (
        FixedConnection
    )
    """
    The connection attaching the localized body to the ``world.root``.
    """

    def __post_init__(self) -> None:
        """
        :raises StartPoseWithoutMobileBaseError: If a start pose is given for a robot
            without a mobile base.
        """
        if self.odometry_T_robot_start is not None and not issubclass(
            self.semantic_annotation_type, HasMobileBase
        ):
            raise StartPoseWithoutMobileBaseError(
                robot_class=self.semantic_annotation_type
            )

    def spawn(self, world: World) -> AbstractRobot:
        """
        Read the robot and merge it into ``world``.

        The robot is annotated while it still owns the world it was read into, so that
        the annotation's name-based lookups cannot be confused by an equally named joint
        of a robot already present in ``world``.

        :param world: The world the robot is merged into.
        :return: The semantic annotation of the merged robot.
        :raises RobotDescriptionMissingError: If no :attr:`description_parser` is given
            and the robot class declares no description file.
        """
        described_robot = self._create_described_robot()
        robot_id = described_robot.id
        if isinstance(described_robot, HasMobileBase):
            self._merge_behind_its_odometry(world, described_robot)
        else:
            self._merge_at_its_localization(world, described_robot)
        return cast("AbstractRobot", world.get_semantic_annotation_by_id(robot_id))

    def _create_described_robot(self) -> AbstractRobot:
        """
        :return: The robot, in a world holding nothing but it, read by the
            :attr:`description_parser` if given and from its own description otherwise.
        """
        if self.description_parser is None:
            return self.semantic_annotation_type.from_description()
        return self.semantic_annotation_type.from_world(
            self.description_parser.create_world()
        )

    def _merge_at_its_localization(
        self, world: World, described_robot: AbstractRobot
    ) -> None:
        """
        Merge a robot without a mobile base as ``world.root -> localization -> robot``.

        :param world: The world the robot is merged into.
        :param described_robot: The robot, in the world it was read into.
        """
        robot_world = described_robot._world
        with world.modify_world():
            localization = self._create_localization(
                world, cast(Body, robot_world.root)
            )
            world.merge_world(robot_world, root_connection=localization)
        self._move_localization_to_its_pose(world, localization)

    def _merge_behind_its_odometry(
        self, world: World, described_robot: HasMobileBase
    ) -> None:
        """
        Merge a robot with a mobile base as ``world.root -> localization -> odometry ->
        drive -> robot``, where the drive is the one its mobile base creates, controlled
        by the servos the base declares.

        :param world: The world the robot is merged into.
        :param described_robot: The robot, in the world it was read into.
        """
        robot_world = described_robot._world
        with world.modify_world():
            odometry_body = self._create_odometry_body(described_robot)
            localization = self._create_localization(world, odometry_body)
            world.add_connection(localization)
            drive = described_robot.mobile_base.create_drive(
                world=world, parent=odometry_body, child=cast(Body, robot_world.root)
            )
            world.merge_world(robot_world, root_connection=drive)
            drive.has_hardware_interface = True
            merged_robot = cast(
                HasMobileBase, world.get_semantic_annotation_by_id(described_robot.id)
            )
            merged_robot.mobile_base.declare_drive_servos(drive)

        # The drive's pose is degree of freedom state, set after the modification block.
        self._move_localization_to_its_pose(world, localization)
        if self.odometry_T_robot_start is not None:
            drive.origin = self.odometry_T_robot_start

    def _create_odometry_body(self, robot: HasMobileBase) -> Body:
        """
        :param robot: The robot whose odometry body is created.
        :return: The odometry body, named as the robot's mobile base declares, with the
            prefix of the robot's own names.
        """
        return Body(
            name=PrefixedName(
                name=robot.mobile_base.odometry_body_name,
                prefix=robot.root.name.prefix,
            )
        )

    def _create_localization(
        self, world: World, localized_body: Body
    ) -> FixedConnection | Connection6DoF:
        """
        :param world: The world the connection is created in.
        :param localized_body: The body the localization attaches to the world root.
        :return: The connection between the world root and ``localized_body``, holding
            the localization pose only if it is fixed.
        """
        is_fixed = issubclass(self.localization_connection_type, FixedConnection)
        return self.localization_connection_type.create_with_dofs(
            world=world,
            parent=cast(Body, world.root),
            child=localized_body,
            parent_T_connection_expression=(
                self._localization_pose_in_frames_of(world, localized_body)
                if is_fixed
                else None
            ),
        )

    def _move_localization_to_its_pose(
        self, world: World, localization: FixedConnection | Connection6DoF
    ) -> None:
        """
        Set the localization pose on a movable localization, whose pose is degree of
        freedom state and so cannot be given at creation.

        :param world: The world holding the localization.
        :param localization: The connection between the world root and the localized
            body.
        """
        if (
            not isinstance(localization, Connection6DoF)
            or self.localization_pose is None
        ):
            return
        localization.origin = self._localization_pose_in_frames_of(
            world, localization.child
        )

    def _localization_pose_in_frames_of(
        self, world: World, localized_body: Body
    ) -> HomogeneousTransformationMatrix | None:
        """
        :param world: The world whose root is the reference frame.
        :param localized_body: The body the localization attaches, which is the child
            frame.
        :return: The localization pose between these frames, if one is given.
        """
        if self.localization_pose is None:
            return None
        return self.localization_pose.copy_with_new_reference_frames(
            new_reference_frame=world.root, new_child_frame=localized_body
        )
