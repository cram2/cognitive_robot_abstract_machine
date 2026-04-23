from __future__ import annotations

import os
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Self

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
    HasCameras,
    HasNeck,
    HasOneArm,
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
class StretchFinger(Finger, ABC):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass


@dataclass(eq=False)
class StretchLeftFinger(StretchFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(
                robot_root, "link_gripper_finger_left"
            ),
            tip=world.get_body_in_branch_by_name(
                robot_root, "link_gripper_fingertip_left"
            ),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class StretchRightFinger(StretchFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(
                robot_root, "link_gripper_finger_right"
            ),
            tip=world.get_body_in_branch_by_name(
                robot_root, "link_gripper_fingertip_right"
            ),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class StretchGripper(ParallelGripper):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
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

        self.add_joint_state(gripper_open)
        self.add_joint_state(gripper_close)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        gripper = cls(
            root=world.get_body_in_branch_by_name(robot_root, "link_straight_gripper"),
            tool_frame=world.get_body_in_branch_by_name(
                robot_root, "link_grasp_center"
            ),
            front_facing_orientation=Quaternion(0, 0, 0, 1),
        )
        world.add_semantic_annotation(gripper)
        return gripper

    def setup_finger_semantic_annotations(self):
        thumb = StretchLeftFinger.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_thumb(thumb)
        finger = (
            StretchRightFinger.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_finger(finger)


@dataclass(eq=False)
class StretchArm(Arm, HasParallelGripper):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        arm_park = JointState.from_mapping(
            name=PrefixedName("arm_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    self.active_connections,
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
            root=world.get_body_in_branch_by_name(robot_root, "link_mast"),
            tip=world.get_body_in_branch_by_name(robot_root, "link_wrist_roll"),
        )
        world.add_semantic_annotation(arm)
        return arm

    def setup_end_effector_semantic_annotation(self):
        gripper = StretchGripper.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_end_effector(gripper)
        gripper.setup_finger_semantic_annotations()


@dataclass(eq=False)
class StretchCameraColor(Camera):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        camera = cls(
            root=world.get_body_in_branch_by_name(
                robot_root, "camera_color_optical_frame"
            ),
            forward_facing_axis=Vector3.Z(),
            minimal_height=1.322,
            maximal_height=1.322,
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            default_camera=True,
        )
        world.add_semantic_annotation(camera)
        return camera


@dataclass(eq=False)
class StretchCameraDepth(Camera):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        camera = cls(
            root=world.get_body_in_branch_by_name(
                robot_root, "camera_depth_optical_frame"
            ),
            forward_facing_axis=Vector3.Z(),
            minimal_height=1.307,
            maximal_height=1.307,
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
        )
        world.add_semantic_annotation(camera)
        return camera


@dataclass(eq=False)
class StretchCameraInfra1(Camera):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        camera = cls(
            root=world.get_body_in_branch_by_name(
                robot_root, "camera_infra1_optical_frame"
            ),
            forward_facing_axis=Vector3.Z(),
            minimal_height=1.307,
            maximal_height=1.307,
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
        )
        world.add_semantic_annotation(camera)
        return camera


@dataclass(eq=False)
class StretchCameraInfra2(Camera):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        camera = cls(
            root=world.get_body_in_branch_by_name(
                robot_root, "camera_infra2_optical_frame"
            ),
            forward_facing_axis=Vector3.Z(),
            minimal_height=1.257,
            maximal_height=1.257,
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
        )
        world.add_semantic_annotation(camera)
        return camera


@dataclass(eq=False)
class StretchNeck(Neck, HasCameras):

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
            root=world.get_body_in_branch_by_name(robot_root, "link_head"),
            tip=world.get_body_in_branch_by_name(robot_root, "link_head_tilt"),
        )
        world.add_semantic_annotation(neck)
        return neck

    def setup_sensor_semantic_annotations(self):
        self.add_sensor(
            StretchCameraColor.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_sensor(
            StretchCameraDepth.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_sensor(
            StretchCameraInfra1.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_sensor(
            StretchCameraInfra2.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )


@dataclass(eq=False)
class StretchTorso(Torso, HasNeck, HasOneArm):

    def setup_neck_semantic_annotation(self):
        neck = StretchNeck.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        neck.setup_sensor_semantic_annotations()
        self.add_neck(neck)

    def setup_arm_semantic_annotations(self):
        arm = StretchArm.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_arm(arm)
        arm.setup_end_effector_semantic_annotation()

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        torso_joint = self.active_connections
        torso_low = JointState.from_mapping(
            name=PrefixedName("torso_low", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.0])),
            state_type=TorsoState.LOW,
        )

        torso_mid = JointState.from_mapping(
            name=PrefixedName("torso_mid", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.55])),
            state_type=TorsoState.MID,
        )

        torso_high = JointState.from_mapping(
            name=PrefixedName("torso_high", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [1.1])),
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
            root=world.get_body_in_branch_by_name(robot_root, "link_mast"),
            tip=world.get_body_in_branch_by_name(robot_root, "link_lift"),
        )
        world.add_semantic_annotation(torso)
        return torso


@dataclass(eq=False)
class StretchMobileBase(MobileBase, HasTorso):

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
            root=world.get_body_in_branch_by_name(robot_root, "base_link"),
        )
        world.add_semantic_annotation(mobile_base)
        return mobile_base

    def setup_torso_semantic_annotation(self):
        torso = StretchTorso.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        torso.setup_arm_semantic_annotations()
        torso.setup_neck_semantic_annotation()
        self.add_torso(torso)


@dataclass(eq=False)
class Stretch(AbstractRobot, HasMobileBase):

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "base_link"

    def setup_mobile_base_semantic_annotation(self):
        mobile_base = (
            StretchMobileBase.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        mobile_base.setup_torso_semantic_annotation()
        self.add_mobile_base(mobile_base)

    def setup_robot_part_semantic_annotations(self):
        self.setup_mobile_base_semantic_annotation()

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
        vel_limits = defaultdict(lambda: 1.0)
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)
