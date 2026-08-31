from __future__ import annotations

import os
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from importlib.resources import files
from pathlib import Path
from typing import Self, Union, List

from krrood.ormatic.utils import classproperty
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
    HasLeftRightArm,
    HasNeck,
    HasTorso,
    HasMobileBase,
    HasFingers,
    TGenericFingerOtherThanThumb,
    HasEndEffector,
    HasSensors,
)
from semantic_digital_twin.world_description.connections import OmniDrive
from semantic_digital_twin.robots.robot_parts import (
    AbstractRobot,
    Arm,
    Camera,
    Finger,
    Neck,
    Torso,
    MobileBase,
    EndEffector,
)
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


class UnitreeG1Joint(StrEnum):
    """
    Names of the UnitreeG1's commandable connections, as spelled in its URDF.

    Members are usable wherever a connection name is expected, so a configuration keyed
    by them stays a plain mapping of names to positions.
    """

    WAIST_YAW = "waist_yaw_joint"
    WAIST_ROLL = "waist_roll_joint"
    WAIST_PITCH = "waist_pitch_joint"

    LEFT_SHOULDER_PITCH = "left_shoulder_pitch_joint"
    LEFT_SHOULDER_ROLL = "left_shoulder_roll_joint"
    LEFT_SHOULDER_YAW = "left_shoulder_yaw_joint"
    LEFT_ELBOW = "left_elbow_joint"
    LEFT_WRIST_ROLL = "left_wrist_roll_joint"
    LEFT_WRIST_PITCH = "left_wrist_pitch_joint"
    LEFT_WRIST_YAW = "left_wrist_yaw_joint"
    LEFT_THUMB_0 = "left_hand_thumb_0_joint"
    LEFT_THUMB_1 = "left_hand_thumb_1_joint"
    LEFT_THUMB_2 = "left_hand_thumb_2_joint"
    LEFT_INDEX_0 = "left_hand_index_0_joint"
    LEFT_INDEX_1 = "left_hand_index_1_joint"
    LEFT_MIDDLE_0 = "left_hand_middle_0_joint"
    LEFT_MIDDLE_1 = "left_hand_middle_1_joint"
    LEFT_HIP_PITCH = "left_hip_pitch_joint"
    LEFT_HIP_ROLL = "left_hip_roll_joint"
    LEFT_HIP_YAW = "left_hip_yaw_joint"
    LEFT_KNEE = "left_knee_joint"
    LEFT_ANKLE_PITCH = "left_ankle_pitch_joint"
    LEFT_ANKLE_ROLL = "left_ankle_roll_joint"

    RIGHT_SHOULDER_PITCH = "right_shoulder_pitch_joint"
    RIGHT_SHOULDER_ROLL = "right_shoulder_roll_joint"
    RIGHT_SHOULDER_YAW = "right_shoulder_yaw_joint"
    RIGHT_ELBOW = "right_elbow_joint"
    RIGHT_WRIST_ROLL = "right_wrist_roll_joint"
    RIGHT_WRIST_PITCH = "right_wrist_pitch_joint"
    RIGHT_WRIST_YAW = "right_wrist_yaw_joint"
    RIGHT_THUMB_0 = "right_hand_thumb_0_joint"
    RIGHT_THUMB_1 = "right_hand_thumb_1_joint"
    RIGHT_THUMB_2 = "right_hand_thumb_2_joint"
    RIGHT_INDEX_0 = "right_hand_index_0_joint"
    RIGHT_INDEX_1 = "right_hand_index_1_joint"
    RIGHT_MIDDLE_0 = "right_hand_middle_0_joint"
    RIGHT_MIDDLE_1 = "right_hand_middle_1_joint"
    RIGHT_HIP_PITCH = "right_hip_pitch_joint"
    RIGHT_HIP_ROLL = "right_hip_roll_joint"
    RIGHT_HIP_YAW = "right_hip_yaw_joint"
    RIGHT_KNEE = "right_knee_joint"
    RIGHT_ANKLE_PITCH = "right_ankle_pitch_joint"
    RIGHT_ANKLE_ROLL = "right_ankle_roll_joint"


@dataclass(eq=False)
class UnitreeG1LeftThumb(Finger):

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
                robot_root, "left_hand_thumb_0_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_hand_thumb_2_link"
            ),
        )


@dataclass(eq=False)
class UnitreeG1LeftIndexFinger(Finger):

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
                robot_root, "left_hand_index_0_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_hand_index_1_link"
            ),
        )


@dataclass(eq=False)
class UnitreeG1LeftMiddleFinger(Finger):

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
                robot_root, "left_hand_middle_0_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_hand_middle_1_link"
            ),
        )


@dataclass(eq=False)
class UnitreeG1RightThumb(Finger):

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
                robot_root, "right_hand_thumb_0_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_hand_thumb_2_link"
            ),
        )


@dataclass(eq=False)
class UnitreeG1RightIndexFinger(Finger):

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
                robot_root, "right_hand_index_0_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_hand_index_1_link"
            ),
        )


@dataclass(eq=False)
class UnitreeG1RightMiddleFinger(Finger):

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
                robot_root, "right_hand_middle_0_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_hand_middle_1_link"
            ),
        )


@dataclass(eq=False)
class UnitreeG1LeftHand(
    EndEffector,
    HasFingers[
        UnitreeG1LeftThumb,
        UnitreeG1LeftIndexFinger,
        UnitreeG1LeftMiddleFinger,
    ],
):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
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

        return [gripper_open, gripper_close]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_hand_palm_link"
            ),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_hand_tool_frame"
            ),
            front_facing_orientation=Quaternion(),
        )


@dataclass(eq=False)
class UnitreeG1RightHand(
    EndEffector,
    HasFingers[
        UnitreeG1RightThumb,
        UnitreeG1RightIndexFinger,
        UnitreeG1RightMiddleFinger,
    ],
):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
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

        return [gripper_open, gripper_close]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_hand_palm_link"
            ),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_hand_tool_frame"
            ),
            front_facing_orientation=Quaternion(),
        )


@dataclass(eq=False)
class UnitreeG1LeftArm(Arm[UnitreeG1LeftHand]):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        arm_park = JointState.from_mapping(
            name=PrefixedName("left_arm_park", prefix=self.name.name),
            mapping=dict(
                zip(self.active_connections, [0.0] * len(self.active_connections))
            ),
            state_type=StaticJointState.PARK,
        )
        return [arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_shoulder_pitch_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_wrist_yaw_link"
            ),
        )


@dataclass(eq=False)
class UnitreeG1RightArm(Arm[UnitreeG1RightHand]):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        arm_park = JointState.from_mapping(
            name=PrefixedName("right_arm_park", prefix=self.name.name),
            mapping=dict(
                zip(self.active_connections, [0.0] * len(self.active_connections))
            ),
            state_type=StaticJointState.PARK,
        )
        return [arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_shoulder_pitch_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_wrist_yaw_link"
            ),
        )


@dataclass(eq=False)
class D435(Camera):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "d435_link"),
            forward_facing_axis=Vector3.Z(),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=1.27,
            maximal_height=1.60,
            default_camera=True,
        )

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []


@dataclass(eq=False)
class UnitreeG1Neck(Neck[D435]):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "torso_link"),
            tip=robot_root._world.get_body_in_branch_by_name(robot_root, "d435_link"),
        )


@dataclass(eq=False)
class UnitreeG1Torso(
    Torso, HasLeftRightArm[UnitreeG1LeftArm, UnitreeG1RightArm], HasNeck[UnitreeG1Neck]
):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "pelvis"),
            tip=robot_root._world.get_body_in_branch_by_name(robot_root, "torso_link"),
        )


@dataclass(eq=False)
class UnitreeG1MobileBase(MobileBase[OmniDrive]):

    @classproperty
    def forward_axis(cls) -> Vector3:
        return Vector3.X()

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "pelvis"),
        )

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []


@dataclass(eq=False)
class UnitreeG1(
    AbstractRobot, HasMobileBase[UnitreeG1MobileBase], HasTorso[UnitreeG1Torso]
):
    """
    The Unitree G1 robot by Unitree Robotics.

    https://www.unitree.com/g1
    """

    @classmethod
    def get_ros_file_path(cls) -> str:
        return "package://iai_offis_g1_description/urdf/offis_unitree_g1.urdf"

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "pelvis"

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
