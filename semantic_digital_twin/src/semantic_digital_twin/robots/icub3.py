from __future__ import annotations

from abc import ABC

import numpy as np
from dataclasses import dataclass
from enum import StrEnum
from typing import Self, Union, List

from krrood.ormatic.utils import classproperty
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
    HasSensors,
    TGenericFingerOtherThanThumb,
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


class ICub3Joint(StrEnum):
    """
    Names of the ICub3's commandable connections, as spelled in its URDF.

    Members are usable wherever a connection name is expected, so a configuration keyed
    by them stays a plain mapping of names to positions.
    """

    TORSO_YAW = "torso_yaw"
    TORSO_ROLL = "torso_roll"
    TORSO_PITCH = "torso_pitch"

    NECK_YAW = "neck_yaw"
    NECK_ROLL = "neck_roll"
    NECK_PITCH = "neck_pitch"
    EYES_TILT = "eyes_tilt"
    LEFT_EYE_PAN = "l_eye_pan_joint"
    RIGHT_EYE_PAN = "r_eye_pan_joint"

    LEFT_SHOULDER_PITCH = "l_shoulder_pitch"
    LEFT_SHOULDER_ROLL = "l_shoulder_roll"
    LEFT_SHOULDER_YAW = "l_shoulder_yaw"
    LEFT_ELBOW = "l_elbow"
    LEFT_WRIST_PROSUPINATION = "l_wrist_prosup"
    LEFT_WRIST_PITCH = "l_wrist_pitch"
    LEFT_WRIST_YAW = "l_wrist_yaw"

    LEFT_THUMB_0 = "l_hand_thumb_0_joint"
    LEFT_THUMB_1 = "l_hand_thumb_1_joint"
    LEFT_THUMB_2 = "l_hand_thumb_2_joint"
    LEFT_THUMB_3 = "l_hand_thumb_3_joint"
    LEFT_INDEX_0 = "l_hand_index_0_joint"
    LEFT_INDEX_1 = "l_hand_index_1_joint"
    LEFT_INDEX_2 = "l_hand_index_2_joint"
    LEFT_INDEX_3 = "l_hand_index_3_joint"
    LEFT_MIDDLE_0 = "l_hand_middle_0_joint"
    LEFT_MIDDLE_1 = "l_hand_middle_1_joint"
    LEFT_MIDDLE_2 = "l_hand_middle_2_joint"
    LEFT_MIDDLE_3 = "l_hand_middle_3_joint"
    LEFT_RING_0 = "l_hand_ring_0_joint"
    LEFT_RING_1 = "l_hand_ring_1_joint"
    LEFT_RING_2 = "l_hand_ring_2_joint"
    LEFT_RING_3 = "l_hand_ring_3_joint"
    LEFT_LITTLE_0 = "l_hand_little_0_joint"
    LEFT_LITTLE_1 = "l_hand_little_1_joint"
    LEFT_LITTLE_2 = "l_hand_little_2_joint"
    LEFT_LITTLE_3 = "l_hand_little_3_joint"

    LEFT_HIP_PITCH = "l_hip_pitch"
    LEFT_HIP_ROLL = "l_hip_roll"
    LEFT_HIP_YAW = "l_hip_yaw"
    LEFT_KNEE = "l_knee"
    LEFT_ANKLE_PITCH = "l_ankle_pitch"
    LEFT_ANKLE_ROLL = "l_ankle_roll"

    RIGHT_SHOULDER_PITCH = "r_shoulder_pitch"
    RIGHT_SHOULDER_ROLL = "r_shoulder_roll"
    RIGHT_SHOULDER_YAW = "r_shoulder_yaw"
    RIGHT_ELBOW = "r_elbow"
    RIGHT_WRIST_PROSUPINATION = "r_wrist_prosup"
    RIGHT_WRIST_PITCH = "r_wrist_pitch"
    RIGHT_WRIST_YAW = "r_wrist_yaw"

    RIGHT_THUMB_0 = "r_hand_thumb_0_joint"
    RIGHT_THUMB_1 = "r_hand_thumb_1_joint"
    RIGHT_THUMB_2 = "r_hand_thumb_2_joint"
    RIGHT_THUMB_3 = "r_hand_thumb_3_joint"
    RIGHT_INDEX_0 = "r_hand_index_0_joint"
    RIGHT_INDEX_1 = "r_hand_index_1_joint"
    RIGHT_INDEX_2 = "r_hand_index_2_joint"
    RIGHT_INDEX_3 = "r_hand_index_3_joint"
    RIGHT_MIDDLE_0 = "r_hand_middle_0_joint"
    RIGHT_MIDDLE_1 = "r_hand_middle_1_joint"
    RIGHT_MIDDLE_2 = "r_hand_middle_2_joint"
    RIGHT_MIDDLE_3 = "r_hand_middle_3_joint"
    RIGHT_RING_0 = "r_hand_ring_0_joint"
    RIGHT_RING_1 = "r_hand_ring_1_joint"
    RIGHT_RING_2 = "r_hand_ring_2_joint"
    RIGHT_RING_3 = "r_hand_ring_3_joint"
    RIGHT_LITTLE_0 = "r_hand_little_0_joint"
    RIGHT_LITTLE_1 = "r_hand_little_1_joint"
    RIGHT_LITTLE_2 = "r_hand_little_2_joint"
    RIGHT_LITTLE_3 = "r_hand_little_3_joint"

    RIGHT_HIP_PITCH = "r_hip_pitch"
    RIGHT_HIP_ROLL = "r_hip_roll"
    RIGHT_HIP_YAW = "r_hip_yaw"
    RIGHT_KNEE = "r_knee"
    RIGHT_ANKLE_PITCH = "r_ankle_pitch"
    RIGHT_ANKLE_ROLL = "r_ankle_roll"


@dataclass(eq=False)
class ICub3LeftThumb(Finger):

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
                robot_root, "l_hand_thumb_0"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "l_hand_thumb_tip"
            ),
        )


@dataclass(eq=False)
class ICub3LeftIndexFinger(Finger):

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
                robot_root, "l_hand_index_0"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "l_hand_index_tip"
            ),
        )


@dataclass(eq=False)
class ICub3LeftMiddleFinger(Finger):

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
                robot_root, "l_hand_middle_0"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "l_hand_middle_tip"
            ),
        )


@dataclass(eq=False)
class ICub3LeftRingFinger(Finger):

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
                robot_root, "l_hand_ring_0"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "l_hand_ring_tip"
            ),
        )


@dataclass(eq=False)
class ICub3LeftLittleFinger(Finger):

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
                robot_root, "l_hand_little_0"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "l_hand_little_tip"
            ),
        )


@dataclass(eq=False)
class ICub3RightThumb(Finger):

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
                robot_root, "r_hand_thumb_0"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "r_hand_thumb_tip"
            ),
        )


@dataclass(eq=False)
class ICub3RightIndexFinger(Finger):

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
                robot_root, "r_hand_index_0"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "r_hand_index_tip"
            ),
        )


@dataclass(eq=False)
class ICub3RightMiddleFinger(Finger):

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
                robot_root, "r_hand_middle_0"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "r_hand_middle_tip"
            ),
        )


@dataclass(eq=False)
class ICub3RightRingFinger(Finger):

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
                robot_root, "r_hand_ring_0"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "r_hand_ring_tip"
            ),
        )


@dataclass(eq=False)
class ICub3RightLittleFinger(Finger):

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
                robot_root, "r_hand_little_0"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "r_hand_little_tip"
            ),
        )


@dataclass(eq=False)
class ICub3LeftHand(
    EndEffector,
    HasFingers[
        ICub3LeftThumb,
        ICub3LeftIndexFinger,
        ICub3LeftMiddleFinger,
        ICub3LeftRingFinger,
        ICub3LeftLittleFinger,
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

        return [gripper_open, gripper_close]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "l_hand"),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "l_gripper_tool_frame"
            ),
            front_facing_orientation=Quaternion(0.5, 0.5, 0.5, 0.5),
        )


@dataclass(eq=False)
class ICub3RightHand(
    EndEffector,
    HasFingers[
        ICub3RightThumb,
        ICub3RightIndexFinger,
        ICub3RightMiddleFinger,
        ICub3RightRingFinger,
        ICub3RightLittleFinger,
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

        return [gripper_open, gripper_close]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "r_hand"),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "r_gripper_tool_frame"
            ),
            front_facing_orientation=Quaternion(0.5, 0.5, 0.5, 0.5),
        )


@dataclass(eq=False)
class ICub3LeftArm(Arm[ICub3LeftHand]):

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
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "root_link"),
            tip=robot_root._world.get_body_in_branch_by_name(robot_root, "l_hand"),
        )


@dataclass(eq=False)
class ICub3RightArm(Arm[ICub3RightHand]):

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
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "root_link"),
            tip=robot_root._world.get_body_in_branch_by_name(robot_root, "r_hand"),
        )


@dataclass(eq=False)
class ICub3Camera(Camera):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "head"),
            forward_facing_axis=Vector3.Z(),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=0.75049,
            maximal_height=0.99483,
            default_camera=True,
        )

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []


@dataclass(eq=False)
class ICub3Neck(Neck[ICub3Camera]):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "chest"),
            tip=robot_root._world.get_body_in_branch_by_name(robot_root, "head"),
        )


@dataclass(eq=False)
class ICub3Torso(
    Torso, HasLeftRightArm[ICub3LeftArm, ICub3RightArm], HasNeck[ICub3Neck]
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
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "root_link"),
            tip=robot_root._world.get_body_in_branch_by_name(robot_root, "chest"),
        )


@dataclass(eq=False)
class ICub3MobileBase(MobileBase[OmniDrive]):

    @classproperty
    def forward_axis(cls) -> Vector3:
        return Vector3.X()

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "l_hip_1"),
        )

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []


@dataclass(eq=False)
class ICub3(AbstractRobot, HasTorso[ICub3Torso], HasMobileBase[ICub3MobileBase]):
    """
    The ICub3 robot built by the Istituto Italiano di Tecnologia.

    https://ami.iit.it/telexistence
    """

    def _setup_collision_rules(self):
        pass

    @classmethod
    def get_ros_file_path(cls) -> str:
        return (
            "package://iai_icub_description/robots/iCubGazeboV3_visuomanip/iCub3.urdf"
        )

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "base_footprint"
