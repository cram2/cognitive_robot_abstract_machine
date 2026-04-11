from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Self

from semantic_digital_twin.collision_checking.collision_matrix import (
    MaxAvoidedCollisionsOverride,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    SelfCollisionMatrixRule,
    AvoidExternalCollisions,
    AvoidSelfCollisions,
)
from semantic_digital_twin.datastructures.definitions import (
    StaticJointState,
    GripperState,
    TorsoState,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import (
    Finger,
    Arm,
    Camera,
    FieldOfView,
    Torso,
    MobileBase,
    HumanoidGripper,
)
from semantic_digital_twin.robots.abstract_robot import (
    SpecifiesLeftRightArm,
    AbstractRobot,
    HasTorso,
    HasMobileBase,
)
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection,
    FixedConnection,
)


@dataclass(eq=False)
class UnitreeG1(AbstractRobot, SpecifiesLeftRightArm, HasTorso, HasMobileBase):
    """
    A class representing the Unitree G1 robot by the Unitree Robotics team.
    """

    @classmethod
    def _get_robot_root_body(cls, world: World) -> Self:
        return world.get_body_by_name("pelvis")

    def _setup_arm_semantic_annotations(self):
        # Create left arm
        left_gripper_thumb = Finger.create_and_add_to_world(
            name=PrefixedName("left_gripper_thumb", prefix=self.name.name),
            root_name="left_hand_thumb_0_link",
            tip_name="left_hand_thumb_2_link",
            world=self._world,
        )

        left_gripper_middle_finger = Finger.create_and_add_to_world(
            name=PrefixedName("left_gripper_middle_finger", prefix=self.name.name),
            root_name="left_hand_middle_0_link",
            tip_name="left_hand_middle_1_link",
            world=self._world,
        )

        left_gripper_index_finger = Finger.create_and_add_to_world(
            name=PrefixedName("left_index_finger", prefix=self.name.name),
            root_name="left_hand_index_0_link",
            tip_name="left_hand_index_1_link",
            world=self._world,
        )

        left_gripper = HumanoidGripper.create_and_add_to_world(
            name=PrefixedName("left_gripper", prefix=self.name.name),
            root_name="left_hand_palm_link",
            tool_frame_name="left_hand_tool_frame",
            front_facing_orientation=Quaternion(),
            thumb=left_gripper_thumb,
            fingers=[left_gripper_middle_finger, left_gripper_index_finger],
            world=self._world,
        )
        left_arm = Arm.create_and_add_to_world(
            name=PrefixedName("left_arm", prefix=self.name.name),
            root_name="left_shoulder_pitch_link",
            tip_name="left_wrist_yaw_link",
            manipulator=left_gripper,
            world=self._world,
        )

        self.add_arm(left_arm)

        # Create right arm
        right_gripper_thumb = Finger.create_and_add_to_world(
            name=PrefixedName("right_gripper_thumb", prefix=self.name.name),
            root_name="right_hand_thumb_0_link",
            tip_name="right_hand_thumb_2_link",
            world=self._world,
        )

        right_gripper_middle_finger = Finger.create_and_add_to_world(
            name=PrefixedName("right_gripper_middle_finger", prefix=self.name.name),
            root_name="right_hand_middle_0_link",
            tip_name="right_hand_middle_1_link",
            world=self._world,
        )

        right_gripper_index_finger = Finger.create_and_add_to_world(
            name=PrefixedName("right_index_finger", prefix=self.name.name),
            root_name="right_hand_index_0_link",
            tip_name="right_hand_index_1_link",
        )

        right_gripper = HumanoidGripper.create_and_add_to_world(
            name=PrefixedName("right_gripper", prefix=self.name.name),
            root_name="right_hand_palm_link",
            tool_frame_name="right_hand_tool_frame",
            front_facing_orientation=Quaternion(),
            thumb=right_gripper_thumb,
            fingers=[right_gripper_middle_finger, right_gripper_index_finger],
            world=self._world,
        )

        right_arm = Arm.create_and_add_to_world(
            name=PrefixedName("right_arm", prefix=self.name.name),
            root_name="right_shoulder_pitch_link",
            tip_name="right_wrist_yaw_link",
            manipulator=right_gripper,
            world=self._world,
        )

        self.add_arm(right_arm)

    def _setup_arm_hardware_interfaces(self):
        for arm in self.arms:
            for connection in arm.active_connections:
                connection.has_hardware_interface = True

    def _setup_arm_joint_state(self):
        right_arm_park = JointState.from_mapping(
            name=PrefixedName("right_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    [
                        c
                        for c in self.right_arm.connections
                        if type(c) != FixedConnection
                    ],
                    [0, 0, 0, 0, 0, 0, 0],
                )
            ),
            state_type=StaticJointState.PARK,
        )

        self.right_arm.add_joint_state(right_arm_park)

        left_arm_park = JointState.from_mapping(
            name=PrefixedName("left_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    [
                        c
                        for c in self.left_arm.connections
                        if type(c) != FixedConnection
                    ],
                    [0, 0, 0, 0, 0, 0, 0],
                )
            ),
            state_type=StaticJointState.PARK,
        )

        self.left_arm.add_joint_state(left_arm_park)

        left_gripper_joints = [
            c
            for c in self.left_arm.manipulator.connections
            if type(c) != FixedConnection
        ]

        left_gripper_open = JointState.from_mapping(
            name=PrefixedName("left_gripper_open", prefix=self.name.name),
            mapping=dict(zip(left_gripper_joints, [0, 0, 0, 0.0, 0.0, 0.0, 0.0])),
            state_type=GripperState.OPEN,
        )

        left_gripper_close = JointState.from_mapping(
            name=PrefixedName("left_gripper_close", prefix=self.name.name),
            mapping=dict(
                zip(
                    left_gripper_joints,
                    [
                        -1.57079632,
                        -1.74532925,
                        -1.57079632,
                        -1.74532925,
                        -1.04719755,
                        -0.61086523,
                        0.0,
                    ],
                )
            ),
            state_type=GripperState.CLOSE,
        )

        self.left_arm.manipulator.add_joint_state(left_gripper_close)
        self.left_arm.manipulator.add_joint_state(left_gripper_open)

        right_gripper_joints = [
            c
            for c in self.right_arm.manipulator.connections
            if type(c) != FixedConnection
        ]

        right_gripper_open = JointState.from_mapping(
            name=PrefixedName("right_gripper_open", prefix=self.name.name),
            mapping=dict(zip(right_gripper_joints, [0, 0, 0, 0.0, 0.0, 0.0, 0.0])),
            state_type=GripperState.OPEN,
        )

        right_gripper_close = JointState.from_mapping(
            name=PrefixedName("right_gripper_close", prefix=self.name.name),
            mapping=dict(
                zip(
                    right_gripper_joints,
                    [
                        1.57079632,
                        1.74532925,
                        1.57079632,
                        1.74532925,
                        1.04719755,
                        0.61086523,
                        0.0,
                    ],
                )
            ),
            state_type=GripperState.CLOSE,
        )

        self.right_arm.manipulator.add_joint_state(right_gripper_close)
        self.right_arm.manipulator.add_joint_state(right_gripper_open)

    def _setup_torso_semantic_annotations(self):
        # Create camera and neck
        camera = Camera.create_and_add_to_world(
            name=PrefixedName("d435_link", prefix=self.name.name),
            root_name="d435_link",
            forward_facing_axis=Vector3(0, 0, 1),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=1.27,
            maximal_height=1.60,
            world=self._world,
            default_camera=True,
        )

        # Create torso
        torso = Torso.create_and_add_to_world(
            name=PrefixedName("torso", prefix=self.name.name),
            root_name="pelvis",
            tip_name="torso_link",
            world=self._world,
            sensors=[camera],
        )
        self.add_torso(torso)

    def _setup_torso_hardware_interfaces(self):
        for connection in self.torso.active_connections:
            connection.has_hardware_interface = True

    def _setup_torso_joint_state(self):
        torso_low = JointState.from_mapping(
            name=PrefixedName("torso_low", prefix=self.name.name),
            mapping=dict(),
            state_type=TorsoState.LOW,
        )

        torso_mid = JointState.from_mapping(
            name=PrefixedName("torso_mid", prefix=self.name.name),
            mapping=dict(),
            state_type=TorsoState.MID,
        )

        torso_high = JointState.from_mapping(
            name=PrefixedName("torso_high", prefix=self.name.name),
            mapping=dict(),
            state_type=TorsoState.HIGH,
        )

        self.torso.add_joint_state(torso_low)
        self.torso.add_joint_state(torso_mid)
        self.torso.add_joint_state(torso_high)

    def _setup_mobile_base_semantic_annotations(self):
        # Create the robot base
        base = MobileBase.create_and_add_to_world(
            name=PrefixedName("base", prefix=self.name.name),
            root_name="pelvis",
            world=self._world,
            main_axis=Vector3.X(),
            full_body_controlled=False,
        )

        self.add_mobile_base(base)

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
                    buffer_zone_distance=0.1, violated_distance=0.0, robot=self
                ),
                AvoidExternalCollisions(
                    buffer_zone_distance=0.05,
                    violated_distance=0.0,
                    robot=self,
                    body_subset=self.left_arm.bodies_with_collision
                    + self.right_arm.bodies_with_collision,
                ),
                AvoidExternalCollisions(
                    buffer_zone_distance=0.2,
                    violated_distance=0.05,
                    robot=self,
                    body_subset={self._world.get_body_by_name("pelvis")},
                ),
                AvoidSelfCollisions(
                    buffer_zone_distance=0.05, violated_distance=0.0, robot=self
                ),
            ]
        )

        self._world.collision_manager.extend_max_avoided_bodies_rules(
            [
                MaxAvoidedCollisionsOverride(
                    2, bodies={self._world.get_body_by_name("pelvis")}
                ),
                MaxAvoidedCollisionsOverride(
                    4,
                    bodies=set(
                        self._world.get_direct_child_bodies_with_collision(
                            self._world.get_body_by_name("left_wrist_yaw_link")
                        )
                    )
                    | set(
                        self._world.get_direct_child_bodies_with_collision(
                            self._world.get_body_by_name("right_wrist_yaw_link")
                        )
                    ),
                ),
            ]
        )

    def _setup_velocity_limits(self):
        vel_limits = defaultdict(
            lambda: 1.0,
        )
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)

    def _setup_other_hardware_interfaces(self):
        pass
