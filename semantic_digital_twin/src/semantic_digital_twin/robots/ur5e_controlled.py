from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Self

from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import (
    HasOneArm,
    HasParallelGripper,
)
from semantic_digital_twin.robots.robot_parts import (
    AbstractRobot,
    Arm,
    Finger,
    ParallelGripper,
)
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@dataclass(eq=False)
class UR5ControlledFinger(Finger, ABC):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass


@dataclass(eq=False)
class UR5ControlledLeftFinger(UR5ControlledFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(
                robot_root, "robotiq_85_left_finger_link"
            ),
            tip=world.get_body_in_branch_by_name(
                robot_root, "robotiq_85_left_finger_tip_link"
            ),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class UR5ControlledRightFinger(UR5ControlledFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(
                robot_root, "robotiq_85_right_finger_link"
            ),
            tip=world.get_body_in_branch_by_name(
                robot_root, "robotiq_85_right_finger_tip_link"
            ),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class UR5ControlledGripper(ParallelGripper):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        gripper_joints = self.active_connections

        gripper_open = JointState.from_mapping(
            name=PrefixedName("gripper_open", prefix=self.name.name),
            mapping=dict(
                zip(
                    gripper_joints,
                    [
                        0.798,
                        0.00366,
                        0.796,
                        -0.793,
                        0.798,
                        0.00366,
                        0.796,
                        -0.793,
                    ],
                )
            ),
            state_type=GripperState.OPEN,
        )

        gripper_close = JointState.from_mapping(
            name=PrefixedName("gripper_close", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
            state_type=GripperState.CLOSE,
        )

        self.add_joint_state(gripper_open)
        self.add_joint_state(gripper_close)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        gripper = cls(
            root=world.get_body_in_branch_by_name(
                robot_root, "robotiq_gripper-2F-85_link"
            ),
            tool_frame=world.get_body_in_branch_by_name(robot_root, "right_pad"),
            front_facing_orientation=Quaternion(0, 0, 0, 1),
        )
        world.add_semantic_annotation(gripper)
        return gripper

    def setup_finger_semantic_annotations(self):
        thumb = UR5ControlledLeftFinger.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_thumb(thumb)
        finger = UR5ControlledRightFinger.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_finger(finger)


@dataclass(eq=False)
class UR5ControlledArm(Arm, HasParallelGripper):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        arm_park = JointState.from_mapping(
            name=PrefixedName("arm_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    self.active_connections,
                    [3.14, -1.56, 1.58, -1.57, -1.57, 0.0],
                )
            ),
            state_type=StaticJointState.PARK,
        )

        self.add_joint_state(arm_park)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        arm = cls(
            root=world.get_body_in_branch_by_name(robot_root, "base_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "wrist_3_link"),
        )
        world.add_semantic_annotation(arm)
        return arm

    def setup_end_effector_semantic_annotation(self):
        gripper = (
            UR5ControlledGripper.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_end_effector(gripper)
        gripper.setup_finger_semantic_annotations()


@dataclass(eq=False)
class UR5Controlled(AbstractRobot, HasOneArm):

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "base_link"

    def setup_arm_semantic_annotations(self):
        arm = UR5ControlledArm.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_arm(arm)
        arm.setup_end_effector_semantic_annotation()

    def setup_robot_part_semantic_annotations(self):
        self.setup_arm_semantic_annotations()
