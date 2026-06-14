from __future__ import annotations

import os
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Self, Union, List

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
    HasNeck,
    HasOneArm,
    HasTorso,
    HasMobileBase,
    HasTwoFingers,
    HasSensors,
    HasArms,
    HasLeftRightArm,
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
from semantic_digital_twin.world_description.connections import ActiveConnection
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@dataclass(eq=False)
class HSRBLeftFinger(Finger):

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
                robot_root, "hand_l_proximal_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "hand_l_distal_link"
            ),
        )


@dataclass(eq=False)
class HSRBRightFinger(Finger):

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
                robot_root, "hand_r_proximal_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "hand_r_distal_link"
            ),
        )


@dataclass(eq=False)
class HSRBGripper(EndEffector, HasTwoFingers[HSRBLeftFinger, HSRBRightFinger]):

    def setup_hardware_interfaces(self):
        return

    def setup_joint_states(self) -> List[JointState]:
        world = self._world
        gripper_joints = [
            world.get_connection_by_name("hand_l_proximal_joint"),
            world.get_connection_by_name("hand_r_proximal_joint"),
            world.get_connection_by_name("hand_motor_joint"),
        ]

        gripper_open = JointState.from_mapping(
            name=PrefixedName("gripper_open", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.3, 0.3, 0.3])),
            state_type=GripperState.OPEN,
        )

        gripper_close = JointState.from_mapping(
            name=PrefixedName("gripper_close", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.0, 0.0, 0.0])),
            state_type=GripperState.CLOSE,
        )

        return [gripper_open, gripper_close]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "hand_palm_link"
            ),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "hand_gripper_tool_frame"
            ),
            front_facing_orientation=Quaternion(
                -0.70710678,
                0.0,
                -0.70710678,
                0.0,
            ),
        )


@dataclass(eq=False)
class HSRBHandCamera(Camera):

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
                robot_root, "hand_camera_frame"
            ),
            forward_facing_axis=Vector3.Z(),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=0.75049,
            maximal_height=0.99483,
        )


@dataclass(eq=False)
class HSRBArm(Arm[HSRBGripper], HasSensors[HSRBHandCamera]):

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

    def setup_joint_states(self) -> List[JointState]:
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

        return [arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_lift_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "hand_palm_link"
            ),
        )


@dataclass(eq=False)
class HSRBHeadCenterCamera(Camera):

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
                robot_root, "head_center_camera_frame"
            ),
            forward_facing_axis=Vector3.Z(),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=0.75049,
            maximal_height=0.99483,
            default_camera=True,
        )


@dataclass(eq=False)
class HSRBHeadLeftCamera(Camera):

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
                robot_root, "head_l_stereo_camera_link"
            ),
            forward_facing_axis=Vector3.Z(),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=0.75049,
            maximal_height=0.99483,
        )


@dataclass(eq=False)
class HSRBHeadRightCamera(Camera):

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
                robot_root, "head_r_stereo_camera_link"
            ),
            forward_facing_axis=Vector3.Z(),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=0.75049,
            maximal_height=0.99483,
        )


@dataclass(eq=False)
class HSRBHeadRGBDCamera(Camera):

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
                robot_root, "head_rgbd_sensor_link"
            ),
            forward_facing_axis=Vector3.Z(),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=0.75049,
            maximal_height=0.99483,
            default_camera=True,
        )


@dataclass(eq=False)
class HSRBNeck(
    Neck[
        HSRBHeadCenterCamera,
        HSRBHeadLeftCamera,
        HSRBHeadRightCamera,
        HSRBHeadRGBDCamera,
    ],
):

    def setup_hardware_interfaces(self):
        controlled_joints = ["head_pan_joint", "head_tilt_joint"]
        for joint_name in controlled_joints:
            connection = self._world.get_connection_by_name(joint_name)
            connection.has_hardware_interface = True

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "head_pan_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "head_tilt_link"
            ),
        )


@dataclass(eq=False)
class HSRBTorso(Torso, HasOneArm[HSRBArm], HasNeck[HSRBNeck]):

    def setup_hardware_interfaces(self):
        return

    def setup_joint_states(self) -> List[JointState]:
        torso_joint = self.active_connections
        torso_low = JointState.from_mapping(
            name=PrefixedName("torso_low", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.0])),
            state_type=TorsoState.LOW,
        )

        torso_mid = JointState.from_mapping(
            name=PrefixedName("torso_mid", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.32 / 2])),
            state_type=TorsoState.MID,
        )

        torso_high = JointState.from_mapping(
            name=PrefixedName("torso_high", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.32])),
            state_type=TorsoState.HIGH,
        )

        return [torso_low, torso_mid, torso_high]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "base_link"),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "torso_lift_link"
            ),
        )


@dataclass(eq=False)
class HSRBMobileBase(MobileBase, HasTorso[HSRBTorso]):

    full_body_controlled: bool = field(default=True, kw_only=True)

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
class HSRB(AbstractRobot, HasMobileBase[HSRBMobileBase]):
    """
    The HSRB Robot built by Toyota. https://robotsguide.com/robots/hsr
    """

    @classmethod
    def get_ros_file_path(cls) -> str:
        return "package://hsr_description/robots/hsrb4s.urdf.xacro"

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "base_footprint"

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
        self._world.collision_manager.add_default_rule(
            AvoidExternalCollisions(
                buffer_zone_distance=0.05, violated_distance=0.0, robot=self
            )
        )

        self._world.collision_manager.add_default_rule(
            AvoidExternalCollisions(
                buffer_zone_distance=0.1,
                violated_distance=0.03,
                robot=self,
                body_subset={
                    self._world.get_body_in_branch_by_name(self.root, "base_link")
                },
            )
        )
        self._world.collision_manager.add_default_rule(
            AvoidSelfCollisions(
                buffer_zone_distance=0.03,
                violated_distance=0.0,
                robot=self,
            )
        )

        self._world.collision_manager.max_avoided_bodies_rules.append(
            MaxAvoidedCollisionsOverride(
                2,
                bodies={self._world.get_body_in_branch_by_name(self.root, "base_link")},
            )
        )
        self._world.collision_manager.max_avoided_bodies_rules.append(
            MaxAvoidedCollisionsOverride(
                4,
                bodies=set(
                    self._world.get_direct_child_bodies_with_collision(
                        self._world.get_body_in_branch_by_name(
                            self.root, "wrist_roll_link"
                        )
                    )
                ),
            )
        )

    def _setup_velocity_limits(self):
        vel_limits = defaultdict(lambda: 1.0)
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)

    @property
    def end_effector(self) -> HSRBGripper:
        return self.mobile_base.torso.arm.end_effector
