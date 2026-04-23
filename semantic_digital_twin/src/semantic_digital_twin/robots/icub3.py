from __future__ import annotations

from abc import ABC

import numpy as np
from dataclasses import dataclass
from typing import Self

from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import (
    HasCameras,
    HasLeftRightArm,
    HasNeck,
    HasParallelGripper,
    HasTorso,
    HasMobileBase,
)
from semantic_digital_twin.robots.robot_parts import (
    AbstractRobot,
    Arm,
    Camera,
    FieldOfView,
    Finger,
    Neck,
    ParallelGripper,
    Torso,
    MobileBase,
)
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@dataclass(eq=False)
class ICub3Finger(Finger, ABC):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass


@dataclass(eq=False)
class ICub3LeftThumb(ICub3Finger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "l_hand_thumb_0"),
            tip=world.get_body_in_branch_by_name(robot_root, "l_hand_thumb_tip"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class ICub3LeftIndexFinger(ICub3Finger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "l_hand_index_0"),
            tip=world.get_body_in_branch_by_name(robot_root, "l_hand_index_tip"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class ICub3LeftMiddleFinger(ICub3Finger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "l_hand_middle_0"),
            tip=world.get_body_in_branch_by_name(robot_root, "l_hand_middle_tip"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class ICub3LeftRingFinger(ICub3Finger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "l_hand_ring_0"),
            tip=world.get_body_in_branch_by_name(robot_root, "l_hand_ring_tip"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class ICub3LeftLittleFinger(ICub3Finger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "l_hand_little_0"),
            tip=world.get_body_in_branch_by_name(robot_root, "l_hand_little_tip"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class ICub3RightThumb(ICub3Finger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "r_hand_thumb_0"),
            tip=world.get_body_in_branch_by_name(robot_root, "r_hand_thumb_tip"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class ICub3RightIndexFinger(ICub3Finger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "r_hand_index_0"),
            tip=world.get_body_in_branch_by_name(robot_root, "r_hand_index_tip"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class ICub3RightMiddleFinger(ICub3Finger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "r_hand_middle_0"),
            tip=world.get_body_in_branch_by_name(robot_root, "r_hand_middle_tip"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class ICub3RightRingFinger(ICub3Finger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "r_hand_ring_0"),
            tip=world.get_body_in_branch_by_name(robot_root, "r_hand_ring_tip"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class ICub3RightLittleFinger(ICub3Finger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "r_hand_little_0"),
            tip=world.get_body_in_branch_by_name(robot_root, "r_hand_little_tip"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class ICub3Gripper(ParallelGripper, ABC):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        gripper_joints = self.active_connections

        gripper_open = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_open", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.0] * len(gripper_joints))),
            state_type=GripperState.OPEN,
        )

        close_vals = [
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
            -0.3490658503988659,
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
            0.3490658503988659,
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
            0.3490658503988659,
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
            0.3490658503988659,
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
        ]

        gripper_close = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_close", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, close_vals)),
            state_type=GripperState.CLOSE,
        )

        self.add_joint_state(gripper_open)
        self.add_joint_state(gripper_close)


@dataclass(eq=False)
class ICub3LeftHand(ICub3Gripper):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        gripper = cls(
            root=world.get_body_in_branch_by_name(robot_root, "l_hand"),
            tool_frame=world.get_body_in_branch_by_name(
                robot_root, "l_gripper_tool_frame"
            ),
            front_facing_orientation=Quaternion(0.5, 0.5, 0.5, 0.5),
        )
        world.add_semantic_annotation(gripper)
        return gripper

    def setup_finger_semantic_annotations(self):
        self.add_thumb(
            ICub3LeftThumb.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )

        self.add_finger(
            ICub3LeftIndexFinger.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_finger(
            ICub3LeftMiddleFinger.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_finger(
            ICub3LeftRingFinger.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_finger(
            ICub3LeftLittleFinger.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )


@dataclass(eq=False)
class ICub3RightHand(ICub3Gripper):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        gripper = cls(
            root=world.get_body_in_branch_by_name(robot_root, "r_hand"),
            tool_frame=world.get_body_in_branch_by_name(
                robot_root, "r_gripper_tool_frame"
            ),
            front_facing_orientation=Quaternion(0.5, 0.5, 0.5, 0.5),
        )
        world.add_semantic_annotation(gripper)
        return gripper

    def setup_finger_semantic_annotations(self):
        self.add_thumb(
            ICub3RightThumb.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )

        self.add_finger(
            ICub3RightIndexFinger.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_finger(
            ICub3RightMiddleFinger.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_finger(
            ICub3RightRingFinger.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_finger(
            ICub3RightLittleFinger.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )


@dataclass(eq=False)
class ICub3LeftArm(Arm, HasParallelGripper):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        arm_park = JointState.from_mapping(
            name=PrefixedName("left_arm_park", prefix=self.name.name),
            mapping=dict(
                zip(self.active_connections, [0.0] * len(self.active_connections))
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
            root=world.get_body_in_branch_by_name(robot_root, "root_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "l_hand"),
        )
        world.add_semantic_annotation(arm)
        return arm

    def setup_end_effector_semantic_annotation(self):
        gripper = ICub3LeftHand.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_end_effector(gripper)
        gripper.setup_finger_semantic_annotations()


@dataclass(eq=False)
class ICub3RightArm(Arm, HasParallelGripper):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        arm_park = JointState.from_mapping(
            name=PrefixedName("right_arm_park", prefix=self.name.name),
            mapping=dict(
                zip(self.active_connections, [0.0] * len(self.active_connections))
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
            root=world.get_body_in_branch_by_name(robot_root, "root_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "r_hand"),
        )
        world.add_semantic_annotation(arm)
        return arm

    def setup_end_effector_semantic_annotation(self):
        gripper = ICub3RightHand.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_end_effector(gripper)
        gripper.setup_finger_semantic_annotations()


@dataclass(eq=False)
class ICub3Neck(Neck, HasCameras):

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
            root=world.get_body_in_branch_by_name(robot_root, "chest"),
            tip=world.get_body_in_branch_by_name(robot_root, "head"),
        )
        world.add_semantic_annotation(neck)
        return neck

    def setup_sensor_semantic_annotations(self):
        world = self.root._world
        camera = Camera(
            root=world.get_body_in_branch_by_name(self.root, "head"),
            forward_facing_axis=Vector3.Z(),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=0.75049,
            maximal_height=0.99483,
        )
        world.add_semantic_annotation(camera)
        self.add_sensor(camera)


@dataclass(eq=False)
class ICub3Torso(Torso, HasLeftRightArm, HasNeck):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        pass

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        torso = cls(
            root=world.get_body_in_branch_by_name(robot_root, "root_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "chest"),
        )
        world.add_semantic_annotation(torso)
        return torso

    def setup_arm_semantic_annotations(self):
        left_arm = ICub3LeftArm.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_arm(left_arm)
        left_arm.setup_end_effector_semantic_annotation()

        right_arm = ICub3RightArm.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_arm(right_arm)
        right_arm.setup_end_effector_semantic_annotation()

    def setup_neck_semantic_annotation(self):
        neck = ICub3Neck.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        neck.setup_sensor_semantic_annotations()
        self.add_neck(neck)


@dataclass(eq=False)
class ICub3MobileBase(MobileBase):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        mobile_base = cls(
            root=world.get_body_in_branch_by_name(robot_root, "l_hip_1"),
        )
        world.add_semantic_annotation(mobile_base)
        return mobile_base

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass


@dataclass(eq=False)
class ICub3(AbstractRobot, HasTorso, HasMobileBase):

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "base_footprint"

    def setup_torso_semantic_annotation(self):
        torso = ICub3Torso.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_torso(torso)
        torso.setup_arm_semantic_annotations()
        torso.setup_neck_semantic_annotation()

    def setup_mobile_base_semantic_annotation(self):
        mobile_base = (
            ICub3MobileBase.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_mobile_base(mobile_base)

    def setup_robot_part_semantic_annotations(self):
        self.setup_mobile_base_semantic_annotation()
        self.setup_torso_semantic_annotation()
