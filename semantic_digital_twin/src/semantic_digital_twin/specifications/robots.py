"""
Specifications of robots placed into a world.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING, cast

from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection,
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
    World-independent description of a robot placed into a world: which robot, where its
    odometry frame sits, and where the robot starts within it.

    Materialized via :meth:`spawn`, which merges the robot as ``world.root -> odometry
    -> drive -> robot``. Whether the robot can move relative to its odometry body is a
    matter of its drive alone, which is a fixed connection for a robot without a mobile
    base.
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

    world_T_odometry: HomogeneousTransformationMatrix | None = None
    """
    The localization pose of the robot's odometry body in the ``world.root`` frame.

    If None, identity is used.
    """

    odometry_T_robot_start: HomogeneousTransformationMatrix | None = None
    """
    The start pose of the robot in the frame of its odometry body.

    If None, identity is used.
    """

    localization_connection_type: type[FixedConnection | Connection6DoF] = (
        FixedConnection
    )
    """
    The connection attaching the robot's odometry body to the ``world.root``.
    """

    def spawn(self, world: World) -> AbstractRobot:
        """
        Read the robot and merge it into ``world`` as ``world.root -> odometry ->
        connection -> robot``.

        The odometry body is attached to the world root by a
        :attr:`localization_connection_type` at the localization pose. The connection
        attaching the robot to it is the drive the robot's mobile base creates, or a
        fixed connection when the robot has no mobile base. An active drive is marked as
        controlled, driven by the servos its mobile base declares. Poses backed by
        degrees of freedom are applied after the merge, those of fixed connections at
        creation.

        The robot is annotated while it still owns the world it was read into, so that
        the annotation's name-based lookups cannot be confused by an equally named joint
        of a robot already present in ``world``.

        :param world: The world the robot is merged into.
        :return: The semantic annotation of the merged robot.
        :raises RobotDescriptionMissingError: If no :attr:`description_parser` is given
            and the robot class declares no description file.
        """
        is_active = issubclass(
            self.semantic_annotation_type.get_drive_connection_type(), ActiveConnection
        )

        described_robot = self._create_described_robot()
        robot_world = described_robot._world
        robot_root = cast(Body, robot_world.root)
        robot_id = described_robot.id

        with world.modify_world():
            odometry_body = self._create_odometry_body(described_robot)
            localization = self._create_localization(world, odometry_body)
            world.add_connection(localization)

            # A fixed connection has no DoFs, so its start pose must be set at creation;
            # an active drive carries it as DoF state applied after the block.
            if is_active:
                odometry_C_robot = described_robot.mobile_base.create_drive(
                    world=world, parent=odometry_body, child=robot_root
                )
            else:
                odometry_C_robot = FixedConnection.create_with_dofs(
                    world=world,
                    parent=odometry_body,
                    child=robot_root,
                    parent_T_connection_expression=self.odometry_T_robot_start,
                )
            world.merge_world(robot_world, root_connection=odometry_C_robot)
            robot = cast("AbstractRobot", world.get_semantic_annotation_by_id(robot_id))
            if is_active:
                odometry_C_robot.has_hardware_interface = True
                robot.mobile_base.declare_drive_servos(odometry_C_robot)

        # These poses touch DoF state, so they are set after the modification block.
        if (
            isinstance(localization, Connection6DoF)
            and self.world_T_odometry is not None
        ):
            localization.origin = self._world_T_odometry_in_frames_of(
                world, odometry_body
            )
        if is_active and self.odometry_T_robot_start is not None:
            odometry_C_robot.origin = self.odometry_T_robot_start

        return robot

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

    def _create_localization(
        self, world: World, odometry_body: Body
    ) -> FixedConnection | Connection6DoF:
        """
        :param world: The world the connection is created in.
        :param odometry_body: The robot's odometry body.
        :return: The connection between the world root and ``odometry_body``, holding
            the localization pose only if it is fixed.
        """
        is_fixed = issubclass(self.localization_connection_type, FixedConnection)
        return self.localization_connection_type.create_with_dofs(
            world=world,
            parent=cast(Body, world.root),
            child=odometry_body,
            parent_T_connection_expression=(
                self._world_T_odometry_in_frames_of(world, odometry_body)
                if is_fixed
                else None
            ),
        )

    def _world_T_odometry_in_frames_of(
        self, world: World, odometry_body: Body
    ) -> HomogeneousTransformationMatrix | None:
        """
        :param world: The world whose root is the reference frame.
        :param odometry_body: The robot's odometry body, which is the child frame.
        :return: The localization pose between these frames, if one is given.
        """
        if self.world_T_odometry is None:
            return None
        return self.world_T_odometry.copy_with_new_reference_frames(
            new_reference_frame=world.root, new_child_frame=odometry_body
        )

    def _create_odometry_body(self, robot: AbstractRobot) -> Body:
        """
        :param robot: The robot whose odometry body is created.
        :return: The odometry body, named as the robot class declares, with the prefix
            of the robot's own names.
        """
        return Body(
            name=PrefixedName(
                name=robot.odometry_body_name, prefix=robot.root.name.prefix
            )
        )
