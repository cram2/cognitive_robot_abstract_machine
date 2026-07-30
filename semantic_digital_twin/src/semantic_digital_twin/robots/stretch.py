from __future__ import annotations

import os
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing_extensions import Self, Union, List

from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidExternalCollisions,
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
    HasNeck,
    HasOneArm,
    HasTorso,
    HasMobileBase,
    HasTwoFingers,
    HasEndEffector,
    HasSensors,
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
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


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
            mapping={self._world.get_connection_by_name("joint_lift"): 0.5},
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
class StretchMobileBase(MobileBase, HasTorso[StretchTorso]):

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
        vel_limits[self._world.get_connection_by_name("joint_gripper_finger_left")] = (
            0.01
        )
        vel_limits[self._world.get_connection_by_name("joint_gripper_finger_right")] = (
            0.01
        )
        vel_limits[self._world.get_connection_by_name("joint_wrist_yaw")] = 0.4
        vel_limits[self._world.get_connection_by_name("joint_head_tilt")] = 0.5
        vel_limits[self._world.get_connection_by_name("joint_head_pan")] = 0.5
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)
