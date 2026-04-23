from __future__ import annotations

import os
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Self

from semantic_digital_twin.collision_checking.collision_matrix import (
    MaxAvoidedCollisionsOverride,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidExternalCollisions,
    AvoidSelfCollisions,
    SelfCollisionMatrixRule,
)
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
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
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@dataclass(eq=False)
class UnitreeG1Finger(Finger, ABC):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass


@dataclass(eq=False)
class UnitreeG1LeftThumb(UnitreeG1Finger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "left_hand_thumb_0_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "left_hand_thumb_2_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class UnitreeG1LeftIndexFinger(UnitreeG1Finger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "left_hand_index_0_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "left_hand_index_1_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class UnitreeG1LeftMiddleFinger(UnitreeG1Finger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(
                robot_root, "left_hand_middle_0_link"
            ),
            tip=world.get_body_in_branch_by_name(robot_root, "left_hand_middle_1_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class UnitreeG1RightThumb(UnitreeG1Finger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(
                robot_root, "right_hand_thumb_0_link"
            ),
            tip=world.get_body_in_branch_by_name(robot_root, "right_hand_thumb_2_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class UnitreeG1RightIndexFinger(UnitreeG1Finger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(
                robot_root, "right_hand_index_0_link"
            ),
            tip=world.get_body_in_branch_by_name(robot_root, "right_hand_index_1_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class UnitreeG1RightMiddleFinger(UnitreeG1Finger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(
                robot_root, "right_hand_middle_0_link"
            ),
            tip=world.get_body_in_branch_by_name(
                robot_root, "right_hand_middle_1_link"
            ),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class UnitreeG1Hand(HumanoidHand, ABC):

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
            mapping=dict(zip(gripper_joints, [1.0] * len(gripper_joints))),
            state_type=GripperState.CLOSE,
        )

        self.add_joint_state(gripper_open)
        self.add_joint_state(gripper_close)


@dataclass(eq=False)
class UnitreeG1LeftHand(UnitreeG1Hand):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        gripper = cls(
            root=world.get_body_in_branch_by_name(robot_root, "left_hand_palm_link"),
            tool_frame=world.get_body_in_branch_by_name(
                robot_root, "left_hand_tool_frame"
            ),
            front_facing_orientation=Quaternion(),
        )
        world.add_semantic_annotation(gripper)
        return gripper

    def setup_finger_semantic_annotations(self):
        self.add_thumb(
            UnitreeG1LeftThumb.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_finger(
            UnitreeG1LeftIndexFinger.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_finger(
            UnitreeG1LeftMiddleFinger.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )


@dataclass(eq=False)
class UnitreeG1RightHand(UnitreeG1Hand):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        gripper = cls(
            root=world.get_body_in_branch_by_name(robot_root, "right_hand_palm_link"),
            tool_frame=world.get_body_in_branch_by_name(
                robot_root, "right_hand_tool_frame"
            ),
            front_facing_orientation=Quaternion(),
        )
        world.add_semantic_annotation(gripper)
        return gripper

    def setup_finger_semantic_annotations(self):
        self.add_thumb(
            UnitreeG1RightThumb.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_finger(
            UnitreeG1RightIndexFinger.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_finger(
            UnitreeG1RightMiddleFinger.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )


@dataclass(eq=False)
class UnitreeG1LeftArm(Arm, HasHumanoidHand):

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
            root=world.get_body_in_branch_by_name(
                robot_root, "left_shoulder_pitch_link"
            ),
            tip=world.get_body_in_branch_by_name(robot_root, "left_wrist_yaw_link"),
        )
        world.add_semantic_annotation(arm)
        return arm

    def setup_end_effector_semantic_annotation(self):
        hand = UnitreeG1LeftHand.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_end_effector(hand)
        hand.setup_finger_semantic_annotations()


@dataclass(eq=False)
class UnitreeG1RightArm(Arm, HasHumanoidHand):

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
            root=world.get_body_in_branch_by_name(
                robot_root, "right_shoulder_pitch_link"
            ),
            tip=world.get_body_in_branch_by_name(robot_root, "right_wrist_yaw_link"),
        )
        world.add_semantic_annotation(arm)
        return arm

    def setup_end_effector_semantic_annotation(self):
        hand = UnitreeG1RightHand.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_end_effector(hand)
        hand.setup_finger_semantic_annotations()


@dataclass(eq=False)
class D435(Camera):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        camera = cls(
            root=world.get_body_in_branch_by_name(robot_root, "d435_link"),
            forward_facing_axis=Vector3.Z(),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=1.27,
            maximal_height=1.60,
        )
        world.add_semantic_annotation(camera)

        return camera

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass


@dataclass(eq=False)
class UnitreeG1Neck(Neck, HasCameras):

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
            root=world.get_body_in_branch_by_name(robot_root, "torso_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "d435_link"),
        )
        world.add_semantic_annotation(neck)
        return neck

    def setup_sensor_semantic_annotations(self):
        camera = D435.setup_default_configuration_in_world_below_robot_root(self.root)
        self.add_sensor(camera)


@dataclass(eq=False)
class UnitreeG1Torso(Torso, HasLeftRightArm, HasNeck):

    def setup_arm_semantic_annotations(self):
        left_arm = (
            UnitreeG1LeftArm.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_arm(left_arm)
        left_arm.setup_end_effector_semantic_annotation()

        right_arm = (
            UnitreeG1RightArm.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_arm(right_arm)
        right_arm.setup_end_effector_semantic_annotation()

    def setup_neck_semantic_annotation(self):
        neck = UnitreeG1Neck.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_neck(neck)
        neck.setup_sensor_semantic_annotations()

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
            root=world.get_body_in_branch_by_name(robot_root, "pelvis"),
            tip=world.get_body_in_branch_by_name(robot_root, "torso_link"),
        )
        world.add_semantic_annotation(torso)
        return torso


@dataclass(eq=False)
class UnitreeG1MobileBase(MobileBase):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        mobile_base = cls(
            root=world.get_body_in_branch_by_name(robot_root, "pelvis"),
        )
        world.add_semantic_annotation(mobile_base)
        return mobile_base

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass


@dataclass(eq=False)
class UnitreeG1(AbstractRobot, HasMobileBase, HasTorso):

    def setup_mobile_base_semantic_annotation(self):
        mobile_base = (
            UnitreeG1MobileBase.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_mobile_base(mobile_base)

    def setup_torso_semantic_annotation(self):
        torso = UnitreeG1Torso.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_torso(torso)
        torso.setup_neck_semantic_annotation()
        torso.setup_arm_semantic_annotations()

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "pelvis"

    def setup_robot_part_semantic_annotations(self):
        self.setup_torso_semantic_annotation()
        self.setup_mobile_base_semantic_annotation()

    def _setup_collision_rules(self):
        srdf_path = os.path.join(
            Path(files("semantic_digital_twin")).parent.parent,
            "resources",
            "collision_configs",
            "unitree_g1.srdf",
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

        self._world.collision_manager.extend_max_avoided_bodies_rules(
            [
                MaxAvoidedCollisionsOverride(
                    4,
                    bodies=set(
                        self._world.get_direct_child_bodies_with_collision(
                            self._world.get_body_in_branch_by_name(
                                self.root, "left_wrist_yaw_link"
                            )
                        )
                    )
                    | set(
                        self._world.get_direct_child_bodies_with_collision(
                            self._world.get_body_in_branch_by_name(
                                self.root, "right_wrist_yaw_link"
                            )
                        )
                    ),
                ),
            ]
        )

    def _setup_velocity_limits(self):
        vel_limits = defaultdict(lambda: 1.0)
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)
