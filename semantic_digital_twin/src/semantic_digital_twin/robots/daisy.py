from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Self, List

from semantic_digital_twin.collision_checking.collision_rules import (
    SelfCollisionMatrixRule,
    AvoidExternalCollisions,
    AvoidSelfCollisions,
)
from semantic_digital_twin.datastructures.definitions import (
    StaticJointState,
    GripperState,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import (
    HasLeftRightArm,
    HasTwoFingers,
)
from semantic_digital_twin.robots.robot_parts import (
    AbstractRobot,
    Arm,
    EndEffector,
    Finger,
)
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@dataclass(eq=False)
class DAiSyLeftGripperLeftFinger(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_gripper_left_finger_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_gripper_left_finger_tip_link"
            ),
        )


@dataclass(eq=False)
class DAiSyLeftGripperRightFinger(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_gripper_right_finger_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_gripper_right_finger_tip_link"
            ),
        )


@dataclass(eq=False)
class DAiSyRightGripperLeftFinger(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_gripper_left_finger_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_gripper_left_finger_tip_link"
            ),
        )


@dataclass(eq=False)
class DAiSyRightGripperRightFinger(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_gripper_right_finger_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_gripper_right_finger_tip_link"
            ),
        )


@dataclass(eq=False)
class DAiSyLeftGripper(
    EndEffector, HasTwoFingers[DAiSyLeftGripperLeftFinger, DAiSyLeftGripperRightFinger]
):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        left_gripper_joints = [
            self._world.get_connection_by_name("left_gripper_finger_joint"),
        ]

        gripper_open = JointState.from_mapping(
            name=PrefixedName("left_gripper_open", prefix=self.name.name),
            mapping=dict(zip(left_gripper_joints, [0.0])),
            state_type=GripperState.OPEN,
        )

        gripper_close = JointState.from_mapping(
            name=PrefixedName("left_gripper_close", prefix=self.name.name),
            mapping=dict(
                zip(
                    left_gripper_joints,
                    [
                        0.04,
                    ],
                )
            ),
            state_type=GripperState.CLOSE,
        )
        return [gripper_open, gripper_close]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_gripper_base_link"
            ),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_gripper_tool_frame"
            ),
            front_facing_orientation=Quaternion(0, 0, 0, 1),
        )


@dataclass(eq=False)
class DAiSyRightGripper(
    EndEffector,
    HasTwoFingers[DAiSyRightGripperLeftFinger, DAiSyRightGripperRightFinger],
):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        right_gripper_joints = [
            self._world.get_connection_by_name("right_gripper_finger_joint"),
        ]

        gripper_open = JointState.from_mapping(
            name=PrefixedName("right_gripper_open", prefix=self.name.name),
            mapping=dict(zip(right_gripper_joints, [0.0])),
            state_type=GripperState.OPEN,
        )

        gripper_close = JointState.from_mapping(
            name=PrefixedName("right_gripper_close", prefix=self.name.name),
            mapping=dict(zip(right_gripper_joints, [0.04])),
            state_type=GripperState.CLOSE,
        )

        return [gripper_open, gripper_close]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_gripper_base_link"
            ),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_gripper_tool_frame"
            ),
            front_facing_orientation=Quaternion(0, 0, 0, 1),
        )


@dataclass(eq=False)
class DAiSyLeftArm(Arm[DAiSyLeftGripper]):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        connections = self.active_connections
        arm_park = JointState.from_mapping(
            name=PrefixedName("left_arm_park", prefix=self.name.name),
            mapping=dict(zip(connections, [0, -1.57, 1, 0, 0, 0.785])),
            state_type=StaticJointState.PARK,
        )
        return [arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "table"),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_wrist_3_link"
            ),
        )


@dataclass(eq=False)
class DAiSyRightArm(Arm[DAiSyRightGripper]):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        connections = self.active_connections
        arm_park = JointState.from_mapping(
            name=PrefixedName("right_arm_park", prefix=self.name.name),
            mapping=dict(zip(connections, [2.355, -1.57, 1, 0, 0, 0.785])),
            state_type=StaticJointState.PARK,
        )
        return [arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "table"),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_wrist_3_link"
            ),
        )


@dataclass(eq=False)
class DAiSy(AbstractRobot, HasLeftRightArm[DAiSyLeftArm, DAiSyRightArm]):
    """
    Represents two UR5 Arms mounted on a table.
    The arms are equipped with WEISS WPG 300-120 grippers
    """

    @classmethod
    def get_ros_file_path(cls) -> str:
        return "package://iai_daisy_description/robots/daisy.urdf.xacro"

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "table"

    def _setup_collision_rules(self):
        srdf_path = os.path.join(
            Path(files("semantic_digital_twin")).parent.parent,
            "resources",
            "collision_configs",
            "daisy.srdf",
        )
        self._world.collision_manager.add_ignore_collision_rule(
            SelfCollisionMatrixRule.from_collision_srdf(srdf_path, self._world)
        )

        self._world.collision_manager.extend_default_rules(
            [
                AvoidExternalCollisions(
                    buffer_zone_distance=0.05, violated_distance=0.0, robot=self
                ),
                AvoidSelfCollisions(
                    buffer_zone_distance=0.03,
                    violated_distance=0.0,
                    robot=self,
                ),
            ]
        )

    def _setup_velocity_limits(self):
        self.tighten_dof_velocity_limits_proportionally(maximum_velocity=0.2)

    def get_end_effectors(self) -> list[EndEffector]:
        return [self.left_arm.end_effector, self.right_arm.end_effector]
