"""
Synthetic robot description used to test that the Montessori demo's fixed-base robot
handling (:meth:`~experiments.montessori.world.MontessoriWorld.mount_stationary_robot`,
the navigation-free branch of
:class:`~experiments.montessori.insert_shape_action.InsertMontessoriShapeAction`)
generalizes to a robot with no mobile base, without depending on a real fixed-base
robot's own MJCF (e.g. the Panda's) or a network fetch of its meshes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Self

from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import HasOneArm
from semantic_digital_twin.robots.robot_parts import AbstractRobot, Arm, EndEffector
from semantic_digital_twin.spatial_types import Quaternion
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

SYNTHETIC_FIXED_ARM_ROBOT_URDF_PATH = str(
    Path(__file__).with_name("synthetic_fixed_arm_robot.urdf")
)
"""
Local, self-contained URDF (needs no ROS package or network access to resolve, unlike
every :class:`~semantic_digital_twin.robots.robot_parts.AbstractRobot` with a real ROS
package, and unlike the Panda's MJCF) describing :class:`SyntheticFixedArmRobot`.
"""


@dataclass(eq=False)
class SyntheticGripper(EndEffector):
    """
    The single end effector of :class:`SyntheticFixedArmRobot`, at ``gripper_link``.
    """

    def setup_hardware_interfaces(self):
        self._world.get_connection_by_name("gripper_joint").has_hardware_interface = (
            True
        )

    def setup_joint_states(self) -> list[JointState]:
        # gripper_joint drives the single sliding finger. It is the connection into this
        # gripper's root rather than one within it, so it is referenced by name rather
        # than through active_connections (which spans only the gripper's own chain).
        connection = self._world.get_connection_by_name("gripper_joint")
        gripper_open = JointState.from_mapping(
            name=PrefixedName("gripper_open", prefix=self.name.name),
            mapping={connection: 0.05},
            state_type=GripperState.OPEN,
        )
        gripper_close = JointState.from_mapping(
            name=PrefixedName("gripper_close", prefix=self.name.name),
            mapping={connection: 0.0},
            state_type=GripperState.CLOSE,
        )
        return [gripper_open, gripper_close]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        gripper_link = world.get_body_in_branch_by_name(robot_root, "gripper_link")
        return cls(
            root=gripper_link,
            tool_frame=gripper_link,
            front_facing_orientation=Quaternion(0, 0, 0, 1),
        )


@dataclass(eq=False)
class SyntheticArm(Arm[SyntheticGripper]):
    """
    The single arm of :class:`SyntheticFixedArmRobot`, from ``base_link`` to
    ``arm_link``.
    """

    def setup_hardware_interfaces(self):
        self._world.get_connection_by_name("arm_joint").has_hardware_interface = True

    def setup_joint_states(self) -> list[JointState]:
        [connection] = self.active_connections
        arm_park = JointState.from_mapping(
            name=PrefixedName("arm_park", prefix=self.name.name),
            mapping={connection: 0.0},
            state_type=StaticJointState.PARK,
        )
        return [arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        return cls(
            root=world.get_body_in_branch_by_name(robot_root, "base_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "arm_link"),
        )


@dataclass(eq=False)
class SyntheticFixedArmRobot(AbstractRobot, HasOneArm[SyntheticArm]):
    """
    Minimal, self-contained :class:`~semantic_digital_twin.robots.robot_parts.AbstractRobot`
    standing in for "some fixed-base robot with no mobile base", to prove that
    fixed-base robot-handling code is not accidentally specific to the Panda's own MJCF
    description or to a mobile-base robot.

    Its description (:data:`SYNTHETIC_FIXED_ARM_ROBOT_URDF_PATH`) is a small local file
    rather than a real robot's ROS package or MJCF, so resolving it needs no external
    dependency. ``base_link`` carries no mobile-base connection at all -- unlike
    :class:`~test.experiments_test.dataset.synthetic_wheeled_arm_robot.SyntheticWheeledArmRobot`,
    it is not :class:`~semantic_digital_twin.robots.robot_part_mixins.HasMobileBase`.
    """

    @classmethod
    def get_ros_file_path(cls) -> str:
        return SYNTHETIC_FIXED_ARM_ROBOT_URDF_PATH

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "base_link"

    def _setup_collision_rules(self):
        pass
