from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import StrEnum
from typing import Self, Union, List

from krrood.ormatic.utils import classproperty
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
    TorsoState,
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


class JustinJoint(StrEnum):
    """
    Names of the Justin's commandable connections, as spelled in its URDF.

    Members are usable wherever a connection name is expected, so a configuration keyed by
    them stays a plain mapping of names to positions.

    ..note:: Each finger's distal joint follows its middle joint and is left out.
    """

    TORSO_1 = "torso1_joint"
    TORSO_2 = "torso2_joint"
    TORSO_3 = "torso3_joint"
    TORSO_4 = "torso4_joint"
    HEAD_1 = "head1_joint"
    HEAD_2 = "head2_joint"

    LEFT_ARM_1 = "left_arm1_joint"
    LEFT_ARM_2 = "left_arm2_joint"
    LEFT_ARM_3 = "left_arm3_joint"
    LEFT_ARM_4 = "left_arm4_joint"
    LEFT_ARM_5 = "left_arm5_joint"
    LEFT_ARM_6 = "left_arm6_joint"
    LEFT_ARM_7 = "left_arm7_joint"
    LEFT_THUMB_1 = "left_1thumb1_joint"
    LEFT_THUMB_2 = "left_1thumb2_joint"
    LEFT_THUMB_3 = "left_1thumb3_joint"
    LEFT_INDEX_1 = "left_2tip1_joint"
    LEFT_INDEX_2 = "left_2tip2_joint"
    LEFT_INDEX_3 = "left_2tip3_joint"
    LEFT_MIDDLE_1 = "left_3middle1_joint"
    LEFT_MIDDLE_2 = "left_3middle2_joint"
    LEFT_MIDDLE_3 = "left_3middle3_joint"
    LEFT_RING_1 = "left_4ring1_joint"
    LEFT_RING_2 = "left_4ring2_joint"
    LEFT_RING_3 = "left_4ring3_joint"

    RIGHT_ARM_1 = "right_arm1_joint"
    RIGHT_ARM_2 = "right_arm2_joint"
    RIGHT_ARM_3 = "right_arm3_joint"
    RIGHT_ARM_4 = "right_arm4_joint"
    RIGHT_ARM_5 = "right_arm5_joint"
    RIGHT_ARM_6 = "right_arm6_joint"
    RIGHT_ARM_7 = "right_arm7_joint"
    RIGHT_THUMB_1 = "right_1thumb1_joint"
    RIGHT_THUMB_2 = "right_1thumb2_joint"
    RIGHT_THUMB_3 = "right_1thumb3_joint"
    RIGHT_INDEX_1 = "right_2tip1_joint"
    RIGHT_INDEX_2 = "right_2tip2_joint"
    RIGHT_INDEX_3 = "right_2tip3_joint"
    RIGHT_MIDDLE_1 = "right_3middle1_joint"
    RIGHT_MIDDLE_2 = "right_3middle2_joint"
    RIGHT_MIDDLE_3 = "right_3middle3_joint"
    RIGHT_RING_1 = "right_4ring1_joint"
    RIGHT_RING_2 = "right_4ring2_joint"
    RIGHT_RING_3 = "right_4ring3_joint"


@dataclass(eq=False)
class JustinLeftThumb(Finger):

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
                robot_root, "left_1thumb_base"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_1thumb4"
            ),
        )


@dataclass(eq=False)
class JustinLeftIndexFinger(Finger):

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
                robot_root, "left_2tip_base"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(robot_root, "left_2tip4"),
        )


@dataclass(eq=False)
class JustinLeftMiddleFinger(Finger):

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
                robot_root, "left_3middle_base"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_3middle4"
            ),
        )


@dataclass(eq=False)
class JustinLeftRingFinger(Finger):

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
                robot_root, "left_4ring_base"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(robot_root, "left_4ring4"),
        )


@dataclass(eq=False)
class JustinRightThumb(Finger):

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
                robot_root, "right_1thumb_base"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_1thumb4"
            ),
        )


@dataclass(eq=False)
class JustinRightIndexFinger(Finger):

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
                robot_root, "right_2tip_base"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(robot_root, "right_2tip4"),
        )


@dataclass(eq=False)
class JustinRightMiddleFinger(Finger):

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
                robot_root, "right_3middle_base"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_3middle4"
            ),
        )


@dataclass(eq=False)
class JustinRightRingFinger(Finger):

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
                robot_root, "right_4ring_base"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_4ring4"
            ),
        )


@dataclass(eq=False)
class JustinLeftHand(
    EndEffector,
    HasFingers[
        JustinLeftThumb,
        JustinLeftIndexFinger,
        JustinLeftMiddleFinger,
        JustinLeftRingFinger,
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
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "left_arm7"),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "l_gripper_tool_frame"
            ),
            front_facing_orientation=Quaternion(0.707, -0.707, 0.707, -0.707),
        )


@dataclass(eq=False)
class JustinRightHand(
    EndEffector,
    HasFingers[
        JustinRightThumb,
        JustinRightRingFinger,
        JustinRightIndexFinger,
        JustinRightMiddleFinger,
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
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "right_arm7"),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "r_gripper_tool_frame"
            ),
            front_facing_orientation=Quaternion(0.707, 0.707, 0.707, 0.707),
        )


@dataclass(eq=False)
class JustinLeftArm(Arm[JustinLeftHand]):

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
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "base_link"),
            tip=robot_root._world.get_body_in_branch_by_name(robot_root, "left_arm7"),
        )


@dataclass(eq=False)
class JustinRightArm(Arm[JustinRightHand]):

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
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "base_link"),
            tip=robot_root._world.get_body_in_branch_by_name(robot_root, "right_arm7"),
        )


@dataclass(eq=False)
class JustinCamera(Camera):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "head2"),
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
class JustinNeck(Neck[JustinCamera]):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "torso4"),
            tip=robot_root._world.get_body_in_branch_by_name(robot_root, "head2"),
        )


@dataclass(eq=False)
class JustinTorso(
    Torso, HasLeftRightArm[JustinLeftArm, JustinRightArm], HasNeck[JustinNeck]
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
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "base_link"),
            tip=robot_root._world.get_body_in_branch_by_name(robot_root, "torso4"),
        )


@dataclass(eq=False)
class JustinMobileBase(MobileBase[OmniDrive], HasTorso[JustinTorso]):

    @classproperty
    def forward_axis(cls) -> Vector3:
        return Vector3.X()

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "base_link"),
        )

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []


@dataclass(eq=False)
class Justin(AbstractRobot, HasMobileBase[JustinMobileBase]):
    """
    The Justin robot built by the DLR.

    https://www.dlr.de/en/rm/research/robotic-systems/humanoids/rollin-justin
    """

    def _setup_collision_rules(self):
        pass

    @classmethod
    def get_ros_file_path(cls) -> str:
        return "package://iai_dlr_rollin_justin/urdf/rollin_justin.urdf"

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "base_footprint"
