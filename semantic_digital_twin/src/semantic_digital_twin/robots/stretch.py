from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from importlib.resources import files
from pathlib import Path

from typing_extensions import Self, List

from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidExternalCollisions,
    SelfCollisionMatrixRule,
)
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
    TorsoState,
)
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import (
    HasNeck,
    HasOneArm,
    HasTorso,
    HasMobileBase,
    HasTwoFingers,
)
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
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.world_description.connections import DifferentialDrive
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


class StretchJoint(StrEnum):
    """
    Names of the Stretch's commandable connections, as spelled in its URDF.

    Members are usable wherever a connection name is expected, so a configuration keyed by
    them stays a plain mapping of names to positions.

    ..note:: The two drive wheels are members because the standalone interface registers
        them alongside the arm and head joints.
    """

    LIFT = "joint_lift"
    ARM_L0 = "joint_arm_l0"
    ARM_L1 = "joint_arm_l1"
    ARM_L2 = "joint_arm_l2"
    ARM_L3 = "joint_arm_l3"
    WRIST_YAW = "joint_wrist_yaw"
    WRIST_PITCH = "joint_wrist_pitch"
    WRIST_ROLL = "joint_wrist_roll"

    GRIPPER_LEFT_FINGER = "joint_gripper_finger_left"
    GRIPPER_RIGHT_FINGER = "joint_gripper_finger_right"

    HEAD_PAN = "joint_head_pan"
    HEAD_TILT = "joint_head_tilt"

    LEFT_WHEEL = "joint_left_wheel"
    RIGHT_WHEEL = "joint_right_wheel"


@dataclass(eq=False)
class StretchLeftFinger(Finger):

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
                robot_root, "link_gripper_finger_left"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "link_gripper_fingertip_left"
            ),
        )


@dataclass(eq=False)
class StretchRightFinger(Finger):

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
                robot_root, "link_gripper_finger_right"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "link_gripper_fingertip_right"
            ),
        )


@dataclass(eq=False)
class StretchGripper(EndEffector, HasTwoFingers[StretchLeftFinger, StretchRightFinger]):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        gripper_joints = self.active_connections

        gripper_open = JointState.from_mapping(
            name=PrefixedName("gripper_open", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.109, 0.109])),
            state_type=GripperState.OPEN,
        )

        gripper_close = JointState.from_mapping(
            name=PrefixedName("gripper_close", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.0, 0.0])),
            state_type=GripperState.CLOSE,
        )

        return [gripper_open, gripper_close]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "link_straight_gripper"
            ),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "link_grasp_center"
            ),
            front_facing_orientation=Quaternion(0, 0, 0, 1),
        )


@dataclass(eq=False)
class StretchArm(Arm[StretchGripper]):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        arm_park = JointState.from_mapping(
            name=PrefixedName("arm_park", prefix=self.name.name),
            mapping={self._world.get_connection_by_name(StretchJoint.LIFT): 0.5},
            state_type=StaticJointState.PARK,
        )

        return [arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "link_lift"),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "link_wrist_roll"
            ),
        )


@dataclass(eq=False)
class StretchCameraColor(Camera):

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
                robot_root, "camera_color_optical_frame"
            ),
            forward_facing_axis=Vector3.Z(),
            minimal_height=1.322,
            maximal_height=1.322,
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            default_camera=True,
        )


@dataclass(eq=False)
class StretchCameraDepth(Camera):

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
                robot_root, "camera_depth_optical_frame"
            ),
            forward_facing_axis=Vector3.Z(),
            minimal_height=1.307,
            maximal_height=1.307,
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
        )


@dataclass(eq=False)
class StretchCameraInfra1(Camera):

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
                robot_root, "camera_infra1_optical_frame"
            ),
            forward_facing_axis=Vector3.Z(),
            minimal_height=1.307,
            maximal_height=1.307,
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
        )


@dataclass(eq=False)
class StretchCameraInfra2(Camera):

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
                robot_root, "camera_infra2_optical_frame"
            ),
            forward_facing_axis=Vector3.Z(),
            minimal_height=1.257,
            maximal_height=1.257,
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
        )


@dataclass(eq=False)
class StretchNeck(
    Neck[
        StretchCameraColor,
        StretchCameraDepth,
        StretchCameraInfra1,
        StretchCameraInfra2,
    ],
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
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "link_head"),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "link_head_tilt"
            ),
        )


@dataclass(eq=False)
class StretchTorso(Torso, HasNeck[StretchNeck], HasOneArm[StretchArm]):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        torso_joint = self.active_connections
        torso_low = JointState.from_mapping(
            name=PrefixedName("torso_low", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.0])),
            state_type=TorsoState.LOW,
        )

        torso_mid = JointState.from_mapping(
            name=PrefixedName("torso_mid", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.5])),
            state_type=TorsoState.MID,
        )

        torso_high = JointState.from_mapping(
            name=PrefixedName("torso_high", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [1])),
            state_type=TorsoState.HIGH,
        )

        return [torso_low, torso_mid, torso_high]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "link_mast"),
            tip=robot_root._world.get_body_in_branch_by_name(robot_root, "link_lift"),
        )


@dataclass(eq=False)
class StretchMobileBase(MobileBase[DifferentialDrive], HasTorso[StretchTorso]):

    full_body_controlled: bool = field(default=True, kw_only=True)
    forward_axis: Vector3 = field(default_factory=Vector3.NEGATIVE_Y)

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "base_link"),
        )


@dataclass(eq=False)
class Stretch(AbstractRobot, HasMobileBase[StretchMobileBase]):
    """
    The Stretch 2 robot by Hello Robot.

    https://teal-blue-zpt3.squarespace.com/stretch-2
    """

    @classmethod
    def get_ros_file_path(cls) -> str:
        return "package://stretch_description/urdf/stretch_from_our_robot.urdf"

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "base_link"

    def _setup_collision_rules(self):
        srdf_path = os.path.join(
            Path(files("semantic_digital_twin")).parent.parent,
            "resources",
            "collision_configs",
            "stretch.srdf",
        )
        self._world.collision_manager.add_ignore_collision_rule(
            SelfCollisionMatrixRule.from_collision_srdf(srdf_path, self._world)
        )
        self._world.collision_manager.add_default_rule(
            AvoidExternalCollisions(
                buffer_zone_distance=0.05, violated_distance=0.0, robot=self
            )
        )

    def _setup_velocity_limits(self):
        vel_limits = defaultdict(lambda: 0.1)
        vel_limits[
            self._world.get_connection_by_name(StretchJoint.GRIPPER_LEFT_FINGER)
        ] = 0.01
        vel_limits[
            self._world.get_connection_by_name(StretchJoint.GRIPPER_RIGHT_FINGER)
        ] = 0.01
        vel_limits[self._world.get_connection_by_name(StretchJoint.WRIST_YAW)] = 0.4
        vel_limits[self._world.get_connection_by_name(StretchJoint.HEAD_TILT)] = 0.5
        vel_limits[self._world.get_connection_by_name(StretchJoint.HEAD_PAN)] = 0.5
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)
