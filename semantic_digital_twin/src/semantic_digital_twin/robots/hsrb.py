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
class HSRBFinger(Finger, ABC):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass


@dataclass(eq=False)
class HSRBLeftFinger(HSRBFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "hand_l_proximal_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "hand_l_distal_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class HSRBRightFinger(HSRBFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "hand_r_proximal_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "hand_r_distal_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class HSRBGripper(ParallelGripper):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        world = self._world
        gripper_joints = [
            world.get_connection_by_name("hand_l_proximal_joint"),
            world.get_connection_by_name("hand_r_proximal_joint"),
            world.get_connection_by_name("hand_motor_joint"),
        ]

        gripper_open = JointState.from_mapping(
            name=PrefixedName("gripper_open", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.6, 0.6, 1.2])),
            state_type=GripperState.OPEN,
        )

        gripper_close = JointState.from_mapping(
            name=PrefixedName("gripper_close", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [-0.1, -0.1, -0.2])),
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
            root=world.get_body_in_branch_by_name(robot_root, "hand_palm_link"),
            tool_frame=world.get_body_in_branch_by_name(
                robot_root, "hand_gripper_tool_frame"
            ),
            front_facing_orientation=Quaternion(
                -0.70710678,
                0.0,
                -0.70710678,
                0.0,
            ),
        )
        world.add_semantic_annotation(gripper)
        return gripper

    def setup_finger_semantic_annotations(self):
        thumb = HSRBLeftFinger.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_thumb(thumb)
        finger = HSRBRightFinger.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_finger(finger)


@dataclass(eq=False)
class HSRBHandCamera(Camera):

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
            root=world.get_body_in_branch_by_name(robot_root, "hand_camera_frame"),
            forward_facing_axis=Vector3.Z(),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=0.75049,
            maximal_height=0.99483,
        )
        world.add_semantic_annotation(camera)
        return camera


@dataclass(eq=False)
class HSRBArm(Arm, HasParallelGripper, HasCameras):

    def setup_hardware_interfaces(self):
        controlled_joints = [
            "arm_flex_joint",
            "arm_lift_joint",
            "arm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
        ]
        for joint_name in controlled_joints:
            connection = self._world.get_connection_by_name(joint_name)
            connection.has_hardware_interface = True

    def setup_joint_states(self):
        arm_park = JointState.from_mapping(
            name=PrefixedName("arm_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    self.active_connections,
                    [0.0, 1.5, -1.85, 0.0],
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
            root=world.get_body_in_branch_by_name(robot_root, "arm_lift_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "hand_palm_link"),
        )
        world.add_semantic_annotation(arm)
        return arm

    def setup_end_effector_semantic_annotation(self):
        gripper = HSRBGripper.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_end_effector(gripper)
        gripper.setup_finger_semantic_annotations()

    def setup_sensor_semantic_annotations(self):
        camera = HSRBHandCamera.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_sensor(camera)


@dataclass(eq=False)
class HSRBHeadCenterCamera(Camera):

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
                robot_root, "head_center_camera_frame"
            ),
            forward_facing_axis=Vector3.Z(),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=0.75049,
            maximal_height=0.99483,
            default_camera=True,
        )
        world.add_semantic_annotation(camera)
        return camera


@dataclass(eq=False)
class HSRBHeadLeftCamera(Camera):

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
                robot_root, "head_l_stereo_camera_link"
            ),
            forward_facing_axis=Vector3.Z(),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=0.75049,
            maximal_height=0.99483,
        )
        world.add_semantic_annotation(camera)
        return camera


@dataclass(eq=False)
class HSRBHeadRightCamera(Camera):

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
                robot_root, "head_r_stereo_camera_link"
            ),
            forward_facing_axis=Vector3.Z(),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=0.75049,
            maximal_height=0.99483,
        )
        world.add_semantic_annotation(camera)
        return camera


@dataclass(eq=False)
class HSRBHeadRGBDCamera(Camera):

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
            root=world.get_body_in_branch_by_name(robot_root, "head_rgbd_sensor_link"),
            forward_facing_axis=Vector3.Z(),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=0.75049,
            maximal_height=0.99483,
            default_camera=True,
        )
        world.add_semantic_annotation(camera)
        return camera


@dataclass(eq=False)
class HSRBNeck(Neck, HasCameras):

    def setup_hardware_interfaces(self):
        controlled_joints = ["head_pan_joint", "head_tilt_joint"]
        for joint_name in controlled_joints:
            connection = self._world.get_connection_by_name(joint_name)
            connection.has_hardware_interface = True

    def setup_joint_states(self):
        pass

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        neck = cls(
            root=world.get_body_in_branch_by_name(robot_root, "head_pan_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "head_tilt_link"),
        )
        world.add_semantic_annotation(neck)
        return neck

    def setup_sensor_semantic_annotations(self):
        self.add_sensor(
            HSRBHeadCenterCamera.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_sensor(
            HSRBHeadLeftCamera.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_sensor(
            HSRBHeadRightCamera.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_sensor(
            HSRBHeadRGBDCamera.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )


@dataclass(eq=False)
class HSRBTorso(Torso, HasOneArm, HasNeck):

    def setup_arm_semantic_annotations(self):
        arm = HSRBArm.setup_default_configuration_in_world_below_robot_root(self.root)
        self.add_arm(arm)
        arm.setup_end_effector_semantic_annotation()
        arm.setup_sensor_semantic_annotations()

    def setup_neck_semantic_annotation(self):
        neck = HSRBNeck.setup_default_configuration_in_world_below_robot_root(self.root)
        neck.setup_sensor_semantic_annotations()
        self.add_neck(neck)

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
            mapping=dict(zip(torso_joint, [0.345 / 2])),
            state_type=TorsoState.MID,
        )

        torso_high = JointState.from_mapping(
            name=PrefixedName("torso_high", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.345])),
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
            root=world.get_body_in_branch_by_name(robot_root, "base_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "torso_lift_link"),
        )
        world.add_semantic_annotation(torso)
        return torso


@dataclass(eq=False)
class HSRBMobileBase(MobileBase, HasTorso):

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
        torso = HSRBTorso.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_torso(torso)
        torso.setup_arm_semantic_annotations()
        torso.setup_neck_semantic_annotation()


@dataclass(eq=False)
class HSRB(AbstractRobot, HasMobileBase):

    @classmethod
    def get_ros_file_path(cls) -> str:
        return "package://hsr_description/robots/hsrb4s.urdf.xacro"

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "base_footprint"

    def setup_mobile_base_semantic_annotation(self):
        mobile_base = (
            HSRBMobileBase.setup_default_configuration_in_world_below_robot_root(
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
            "hsrb.srdf",
        )
        self._world.collision_manager.add_ignore_collision_rule(
            SelfCollisionMatrixRule.from_collision_srdf(srdf_path, self._world)
        )
        self._world.collision_manager.extend_default_rules(
            [
                AvoidExternalCollisions(
                    buffer_zone_distance=0.05, violated_distance=0.0, robot=self
                ),
                AvoidExternalCollisions(
                    buffer_zone_distance=0.1,
                    violated_distance=0.03,
                    robot=self,
                    body_subset={
                        self._world.get_body_in_branch_by_name(self.root, "base_link")
                    },
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
                    2,
                    bodies={
                        self._world.get_body_in_branch_by_name(self.root, "base_link")
                    },
                ),
                MaxAvoidedCollisionsOverride(
                    4,
                    bodies=set(
                        self._world.get_direct_child_bodies_with_collision(
                            self._world.get_body_in_branch_by_name(
                                self.root, "wrist_roll_link"
                            )
                        )
                    ),
                ),
            ]
        )

    def _setup_velocity_limits(self):
        vel_limits = defaultdict(lambda: 1.0)
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)
