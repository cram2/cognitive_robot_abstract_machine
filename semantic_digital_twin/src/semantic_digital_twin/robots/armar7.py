from __future__ import annotations

import os
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Self

from semantic_digital_twin.collision_checking.collision_rules import (
    SelfCollisionMatrixRule,
)
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
    TorsoState,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import (
    HasCameras,
    HasHumanoidHand,
    HasLeftRightArm,
    HasNeck,
    HasTorso,
    HasMobileBase,
)
from semantic_digital_twin.robots.robot_parts import (
    AbstractRobot,
    Arm,
    Camera,
    FieldOfView,
    Finger,
    HumanoidHand,
    Neck,
    Torso,
    MobileBase,
)
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
)
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@dataclass(eq=False)
class Armar7Finger(Finger, ABC):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass


@dataclass(eq=False)
class Armar7LeftThumb(Armar7Finger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "Hand L Palm_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "Thumb L Tip_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class Armar7LeftRingFinger(Armar7Finger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "Hand L Palm_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "Ring L Tip_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class Armar7LeftPinkyFinger(Armar7Finger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "Hand L Palm_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "Pinky L Tip_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class Armar7LeftMiddleFinger(Armar7Finger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "Hand L Palm_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "Middle L Tip_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class Armar7LeftIndexFinger(Armar7Finger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "Hand L Palm_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "Index L Tip_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class Armar7RightThumb(Armar7Finger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "Hand R Palm_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "Thumb R Tip_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class Armar7RightRingFinger(Armar7Finger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "Hand R Palm_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "Ring R Tip_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class Armar7RightPinkyFinger(Armar7Finger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "Hand R Palm_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "Pinky R Tip_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class Armar7RightMiddleFinger(Armar7Finger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "Hand R Palm_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "Middle R Tip_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class Armar7RightIndexFinger(Armar7Finger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "Hand R Palm_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "Index R Tip_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class Armar7Hand(HumanoidHand):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        gripper_joints = self.active_connections

        gripper_open = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_open", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.0] * len(gripper_joints))),
            state_type=GripperState.OPEN,
        )

        gripper_close = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_close", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [1.57] * len(gripper_joints))),
            state_type=GripperState.CLOSE,
        )

        self.add_joint_state(gripper_open)
        self.add_joint_state(gripper_close)


@dataclass(eq=False)
class Armar7LeftGripper(Armar7Hand):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        gripper = cls(
            root=world.get_body_in_branch_by_name(
                robot_root, "ArmL8_Wrist_Hemisphere_B_link"
            ),
            tool_frame=world.get_body_in_branch_by_name(robot_root, "Hand L TCP_link"),
            front_facing_orientation=Quaternion(-0.5, 0.5, -0.5, 0.5),
        )
        world.add_semantic_annotation(gripper)
        return gripper

    def setup_finger_semantic_annotations(self):
        self.add_thumb(
            Armar7LeftThumb.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_finger(
            Armar7LeftRingFinger.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_finger(
            Armar7LeftPinkyFinger.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_finger(
            Armar7LeftMiddleFinger.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_finger(
            Armar7LeftIndexFinger.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )


@dataclass(eq=False)
class Armar7RightGripper(Armar7Hand):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        gripper = cls(
            root=world.get_body_in_branch_by_name(
                robot_root, "ArmR8_Wrist_Hemisphere_B_link"
            ),
            tool_frame=world.get_body_in_branch_by_name(robot_root, "Hand R TCP_link"),
            front_facing_orientation=Quaternion(-0.5, 0.5, -0.5, 0.5),
        )
        world.add_semantic_annotation(gripper)
        return gripper

    def setup_finger_semantic_annotations(self):
        self.add_thumb(
            Armar7RightThumb.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_finger(
            Armar7RightRingFinger.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_finger(
            Armar7RightPinkyFinger.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_finger(
            Armar7RightMiddleFinger.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_finger(
            Armar7RightIndexFinger.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )


@dataclass(eq=False)
class Armar7LeftArm(Arm, HasHumanoidHand):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        vals = [0.0, 0.0, 0.25, 0.5, 1.0, 1.0, 0.0, 0.0]
        arm_park = JointState.from_mapping(
            name=PrefixedName("left_arm_park", prefix=self.name.name),
            mapping=dict(zip(self.active_connections, vals)),
            state_type=StaticJointState.PARK,
        )
        self.add_joint_state(arm_park)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        arm = cls(
            root=world.get_body_in_branch_by_name(robot_root, "CenterArms_fixed_link"),
            tip=world.get_body_in_branch_by_name(
                robot_root, "ArmL8_Wrist_Hemisphere_B_link"
            ),
        )
        world.add_semantic_annotation(arm)
        return arm

    def setup_end_effector_semantic_annotation(self):
        hand = Armar7LeftGripper.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_end_effector(hand)
        hand.setup_finger_semantic_annotations()


@dataclass(eq=False)
class Armar7RightArm(Arm, HasHumanoidHand):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        vals = [0.0, 0.0, 0.25, -0.5, 1.0, -1.0, 0.0, 0.0]
        arm_park = JointState.from_mapping(
            name=PrefixedName("right_arm_park", prefix=self.name.name),
            mapping=dict(zip(self.active_connections, vals)),
            state_type=StaticJointState.PARK,
        )
        self.add_joint_state(arm_park)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        arm = cls(
            root=world.get_body_in_branch_by_name(robot_root, "CenterArms_fixed_link"),
            tip=world.get_body_in_branch_by_name(
                robot_root, "ArmR8_Wrist_Hemisphere_B_link"
            ),
        )
        world.add_semantic_annotation(arm)
        return arm

    def setup_end_effector_semantic_annotation(self):
        hand = Armar7RightGripper.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_end_effector(hand)
        hand.setup_finger_semantic_annotations()


@dataclass(eq=False)
class AzureKinectRGB(Camera):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        self = cls(
            root=world.get_body_in_branch_by_name(robot_root, "AzureKinect_RGB_link"),
            forward_facing_axis=Vector3.Z(),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=1.3715,
            maximal_height=1.7365,
        )
        world.add_semantic_annotation(self)
        return self

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass


@dataclass(eq=False)
class Armar7Neck(Neck, HasCameras):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        pass

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        neck = cls(
            root=world.get_body_in_branch_by_name(robot_root, "Neck_Root_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "Head_Root_link"),
        )
        world.add_semantic_annotation(neck)
        return neck

    def setup_sensor_semantic_annotations(self):
        self.add_camera(
            AzureKinectRGB.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )


@dataclass(eq=False)
class Armar7Torso(Torso, HasLeftRightArm, HasNeck):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        torso_joints = self.active_connections
        torso_low = JointState.from_mapping(
            name=PrefixedName("torso_low", prefix=self.name.name),
            mapping=dict(zip(torso_joints, [-0.757037, 1.74533, 2.18166 / 2])),
            state_type=TorsoState.LOW,
        )
        torso_mid = JointState.from_mapping(
            name=PrefixedName("torso_mid", prefix=self.name.name),
            mapping=dict(zip(torso_joints, [-0.757037 / 2, 1.74533 / 2, 2.18166 / 4])),
            state_type=TorsoState.MID,
        )
        torso_high = JointState.from_mapping(
            name=PrefixedName("torso_high", prefix=self.name.name),
            mapping=dict(zip(torso_joints, [0.0, 0.0, 0.0])),
            state_type=TorsoState.HIGH,
        )
        self.add_joint_state(torso_low)
        self.add_joint_state(torso_mid)
        self.add_joint_state(torso_high)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        torso = cls(
            root=world.get_body_in_branch_by_name(robot_root, "Platform_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "CenterArms_fixed_link"),
        )
        world.add_semantic_annotation(torso)
        return torso

    def setup_arm_semantic_annotations(self):
        left_arm = Armar7LeftArm.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_arm(left_arm)
        left_arm.setup_end_effector_semantic_annotation()

        right_arm = (
            Armar7RightArm.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_arm(right_arm)
        right_arm.setup_end_effector_semantic_annotation()

    def setup_neck_semantic_annotation(self):
        neck = Armar7Neck.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        neck.setup_sensor_semantic_annotations()
        self.add_neck(neck)


@dataclass(eq=False)
class Armar7MobileBase(MobileBase, HasTorso):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        mobile_base = cls(
            root=world.get_body_in_branch_by_name(robot_root, "Platform_body_link"),
            forward_axis=Vector3.Y(),
        )
        world.add_semantic_annotation(mobile_base)
        return mobile_base

    def setup_torso_semantic_annotation(self):
        torso = Armar7Torso.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        torso.setup_arm_semantic_annotations()
        torso.setup_neck_semantic_annotation()
        self.add_torso(torso)


@dataclass(eq=False)
class Armar7(AbstractRobot, HasMobileBase):

    @classmethod
    def get_ros_file_path(cls) -> str:
        pass

    def setup_mobile_base_semantic_annotation(self):
        mobile_base = (
            Armar7MobileBase.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        mobile_base.setup_torso_semantic_annotation()
        self.add_mobile_base(mobile_base)

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "Dummy_Platform_link"

    def setup_robot_part_semantic_annotations(self):
        self.setup_mobile_base_semantic_annotation()

    def _setup_collision_rules(self):
        pass

    def _setup_velocity_limits(self):
        pass
