from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing_extensions import Self, List

from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import HasOneArm, HasTwoFingers
from semantic_digital_twin.robots.robot_parts import (
    AbstractRobot,
    Arm,
    Finger,
    EndEffector,
)
from semantic_digital_twin.spatial_types import Quaternion
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@dataclass(eq=False)
class PandaLeftFinger(Finger):
    """
    The Panda's fingers have no separate fingertip link, so root and tip are the same body.
    """

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        finger = robot_root._world.get_body_in_branch_by_name(
            robot_root, "/left_finger"
        )
        return cls(root=finger, tip=finger)


@dataclass(eq=False)
class PandaRightFinger(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        finger = robot_root._world.get_body_in_branch_by_name(
            robot_root, "/right_finger"
        )
        return cls(root=finger, tip=finger)


@dataclass(eq=False)
class PandaGripper(EndEffector, HasTwoFingers[PandaLeftFinger, PandaRightFinger]):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        finger_joints = self.active_connections

        gripper_open = JointState.from_mapping(
            name=PrefixedName("gripper_open", prefix=self.name.name),
            mapping=dict(zip(finger_joints, [0.04, 0.04])),
            state_type=GripperState.OPEN,
        )

        gripper_close = JointState.from_mapping(
            name=PrefixedName("gripper_close", prefix=self.name.name),
            mapping=dict(zip(finger_joints, [0.0, 0.0])),
            state_type=GripperState.CLOSE,
        )

        return [gripper_open, gripper_close]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "/hand"),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "/tool_frame"
            ),
            # Rotates the tool frame's forward-facing axis onto the hand's local
            # +Z, which is where the fingers reach out to (the "/tool_frame"
            # body in the MJCF sits 0.1034m further along that same axis, at
            # the point where the gripper actually grasps an object).
            front_facing_orientation=Quaternion(
                0, 0.7071067811865476, 0, 0.7071067811865476
            ),
        )


@dataclass(eq=False)
class PandaArm(Arm[PandaGripper]):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        connections = self.active_connections
        # Franka's standard "ready" pose.
        arm_park = JointState.from_mapping(
            name=PrefixedName("arm_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    connections,
                    [0.0, -0.785398163, 0.0, -2.35619449, 0.0, 1.57079632679, 0.785398163397],
                )
            ),
            state_type=StaticJointState.PARK,
        )
        return [arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "link0"),
            tip=robot_root._world.get_body_in_branch_by_name(robot_root, "link7"),
        )


@dataclass(eq=False)
class Panda(AbstractRobot, HasOneArm[PandaArm]):
    """
    The Franka Emika Panda arm, as used in the stacking demo. https://franka.de/
    """

    @classmethod
    def get_ros_file_path(cls) -> str:
        raise NotImplementedError("We dont have the ROS Package yet")

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "link0"

    def _setup_collision_rules(self):
        # No SRDF-based self-collision matrix exists for the Panda yet (see
        # resources/collision_configs), so unlike Stretch/Tiago/HSRB this only
        # skips self-collisions instead of ignoring a curated adjacency list.
        pass

    def _setup_velocity_limits(self):
        vel_limits = defaultdict(lambda: 2.175)
        vel_limits[self._world.get_connection_by_name("joint5")] = 2.61
        vel_limits[self._world.get_connection_by_name("joint6")] = 2.61
        vel_limits[self._world.get_connection_by_name("joint7")] = 2.61
        vel_limits[self._world.get_connection_by_name("/finger_joint1")] = 0.2
        vel_limits[self._world.get_connection_by_name("/finger_joint2")] = 0.2
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)