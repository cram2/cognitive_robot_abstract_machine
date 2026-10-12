"""
Specifications of robots placed into a world.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Type, cast
from uuid import uuid4

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection,
    FixedConnection,
)
from semantic_digital_twin.world_description.world_entity import Body

if TYPE_CHECKING:
    from semantic_digital_twin.robots.robot_parts import AbstractRobot

# %% robot specifications


@dataclass
class RobotSpecification:
    """
    World-independent description of a robot placed into a world: which robot, where its
    localization frame sits, and where the robot starts within it.

    Materialized via :meth:`spawn`, which merges the robot as ``world.root -> odom ->
    drive -> robot``. The ``odom`` is fixed to the world root; whether the robot can
    move is a matter of its drive alone, which is a fixed connection for a robot without
    a mobile base.
    """

    semantic_annotation_type: Type[AbstractRobot]
    """
    The robot to merge into the world.
    """

    world_T_odom: HomogeneousTransformationMatrix | None = None
    """
    The localization pose of the robot's ``odom`` in the ``world.root`` frame.

    If None, identity is used.
    """

    odom_T_robot_start: HomogeneousTransformationMatrix | None = None
    """
    The start pose of the robot in its ``odom`` frame.

    If None, identity is used.
    """

    def spawn(self, world: World) -> AbstractRobot:
        """
        Parse the robot from its own description and merge it into ``world`` as
        ``world.root -> odom -> connection -> robot``.

        The ``odom`` is fixed to the world root at the localization pose. The connection
        attaching the robot to it is the drive declared by the robot's mobile base, or a
        fixed connection when the robot has no mobile base. An active drive is marked as
        controlled and its start pose is applied afterwards, that of a fixed one at
        creation.

        The robot is annotated while it still owns the world it was parsed into, so that
        the annotation's name-based lookups cannot be confused by an equally named joint
        of a robot already present in ``world``.

        :param world: The world the robot is merged into.
        :return: The semantic annotation of the merged robot.
        """
        connection_type = self.semantic_annotation_type.get_drive_connection_type()
        is_active = issubclass(connection_type, ActiveConnection)

        robot_world = URDFParser.from_file(
            self.semantic_annotation_type.get_ros_file_path()
        ).parse()
        robot_id = self.semantic_annotation_type.from_world(robot_world).id

        with world.modify_world():
            odom_body = self._create_odom_body()
            world.add_connection(
                FixedConnection(
                    parent=cast(Body, world.root),
                    child=odom_body,
                    parent_T_connection_expression=(
                        None
                        if self.world_T_odom is None
                        else self.world_T_odom.copy_with_new_reference_frames(
                            new_reference_frame=world.root, new_child_frame=odom_body
                        )
                    ),
                )
            )

            # A fixed connection has no DoFs, so its start pose must be set at creation;
            # an active drive carries it as DoF state applied after the block.
            odom_C_robot = connection_type.create_with_dofs(
                world=world,
                parent=odom_body,
                child=cast(Body, robot_world.root),
                parent_T_connection_expression=(
                    None if is_active else self.odom_T_robot_start
                ),
            )
            world.merge_world(robot_world, root_connection=odom_C_robot)
            if is_active:
                odom_C_robot.has_hardware_interface = True

        # The start pose touches DoF state, so it is set after the modification block.
        if is_active and self.odom_T_robot_start is not None:
            odom_C_robot.origin = self.odom_T_robot_start

        return cast("AbstractRobot", world.get_semantic_annotation_by_id(robot_id))

    @staticmethod
    def _create_odom_body() -> Body:
        """
        Create the localization body of a single robot.

        Body names are not unique across a world, so the body's own identifier prefixes
        its name. Identifiers are unique even across processes, which keeps the odom
        bodies of several robots distinguishable.

        :return: The created odom body.
        """
        identifier = uuid4()
        return Body(
            name=PrefixedName(name="odom", prefix=str(identifier)), id=identifier
        )
