from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Self

from pkg_resources import resource_filename

from .robot_mixins import HasNeck, SpecifiesLeftRightArm
from ..collision_checking.collision_matrix import MaxAvoidedCollisionsOverride
from ..collision_checking.collision_rules import (
    SelfCollisionMatrixRule,
    AvoidAllCollisions,
    AvoidExternalCollisions,
    AvoidSelfCollisions,
)
from ..datastructures.definitions import StaticJointState, GripperState, TorsoState
from ..datastructures.joint_state import JointState
from ..datastructures.prefixed_name import PrefixedName
from ..robots.abstract_robot import (
    Neck,
    Finger,
    ParallelGripper,
    Arm,
    Camera,
    FieldOfView,
    Torso,
    AbstractRobot,
    Base,
)
from ..spatial_types import Quaternion, Vector3
from ..world import World
from ..world_description.connections import ActiveConnection, FixedConnection


@dataclass(eq=False)
class PR2(AbstractRobot, SpecifiesLeftRightArm, HasNeck):
    """
    Represents the Personal Robot 2 (PR2), which was originally created by Willow Garage.
    The PR2 robot consists of two arms, each with a parallel gripper, a head with a camera, and a prismatic torso
    """

    @classmethod
    def _init_empty_robot(cls, world: World) -> Self:
        return cls(
            name=PrefixedName(name="pr2", prefix=world.name),
            root=world.get_body_by_name("base_footprint"),
            _world=world,
        )

    def _setup_collision_rules(self):
        """
        Loads the SRDF file for the PR2 robot, if it exists.
        """
        srdf_path = os.path.join(
            resource_filename("semantic_digital_twin", "../../"),
            "resources",
            "collision_configs",
            "pr2.srdf",
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
                    body_subset={self._world.get_body_by_name("base_link")},
                ),
                AvoidSelfCollisions(
                    buffer_zone_distance=0.05, violated_distance=0.0, robot=self
                ),
            ]
        )

        self._world.collision_manager.extend_max_avoided_bodies_rules(
            [
                MaxAvoidedCollisionsOverride(
                    2, bodies={self._world.get_body_by_name("base_link")}
                ),
                MaxAvoidedCollisionsOverride(
                    4,
                    bodies=set(
                        self._world.get_direct_child_bodies_with_collision(
                            self._world.get_body_by_name("r_wrist_roll_link")
                        )
                    )
                    | set(
                        self._world.get_direct_child_bodies_with_collision(
                            self._world.get_body_by_name("l_wrist_roll_link")
                        )
                    ),
                ),
            ]
        )

    def _setup_semantic_annotations(self):
        # Create left arm
        left_gripper_thumb = Finger(
            name=PrefixedName("left_gripper_thumb", prefix=self.name.name),
            root=self._world.get_body_by_name("l_gripper_l_finger_link"),
            tip=self._world.get_body_by_name("l_gripper_l_finger_tip_link"),
            _world=self._world,
        )

        left_gripper_finger = Finger(
            name=PrefixedName("left_gripper_finger", prefix=self.name.name),
            root=self._world.get_body_by_name("l_gripper_r_finger_link"),
            tip=self._world.get_body_by_name("l_gripper_r_finger_tip_link"),
            _world=self._world,
        )

        left_gripper = ParallelGripper(
            name=PrefixedName("left_gripper", prefix=self.name.name),
            root=self._world.get_body_by_name("l_gripper_palm_link"),
            tool_frame=self._world.get_body_by_name("l_gripper_tool_frame"),
            front_facing_orientation=Quaternion(0, 0, 0, 1),
            front_facing_axis=Vector3(1, 0, 0),
            thumb=left_gripper_thumb,
            finger=left_gripper_finger,
            _world=self._world,
        )
        left_arm = Arm(
            name=PrefixedName("left_arm", prefix=self.name.name),
            root=self._world.get_body_by_name("torso_lift_link"),
            tip=self._world.get_body_by_name("l_wrist_roll_link"),
            manipulator=left_gripper,
            _world=self._world,
        )

        self.add_arm(left_arm)

        # Create right arm
        right_gripper_thumb = Finger(
            name=PrefixedName("right_gripper_thumb", prefix=self.name.name),
            root=self._world.get_body_by_name("r_gripper_l_finger_link"),
            tip=self._world.get_body_by_name("r_gripper_l_finger_tip_link"),
            _world=self._world,
        )
        right_gripper_finger = Finger(
            name=PrefixedName("right_gripper_finger", prefix=self.name.name),
            root=self._world.get_body_by_name("r_gripper_r_finger_link"),
            tip=self._world.get_body_by_name("r_gripper_r_finger_tip_link"),
            _world=self._world,
        )
        right_gripper = ParallelGripper(
            name=PrefixedName("right_gripper", prefix=self.name.name),
            root=self._world.get_body_by_name("r_gripper_palm_link"),
            tool_frame=self._world.get_body_by_name("r_gripper_tool_frame"),
            front_facing_orientation=Quaternion(0, 0, 0, 1),
            front_facing_axis=Vector3(1, 0, 0),
            thumb=right_gripper_thumb,
            finger=right_gripper_finger,
            _world=self._world,
        )
        right_arm = Arm(
            name=PrefixedName("right_arm", prefix=self.name.name),
            root=self._world.get_body_by_name("torso_lift_link"),
            tip=self._world.get_body_by_name("r_wrist_roll_link"),
            manipulator=right_gripper,
            _world=self._world,
        )

        self.add_arm(right_arm)

        # Create camera and neck
        camera = Camera(
            name=PrefixedName("wide_stereo_optical_frame", prefix=self.name.name),
            root=self._world.get_body_by_name("wide_stereo_optical_frame"),
            forward_facing_axis=Vector3(0, 0, 1),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=1.27,
            maximal_height=1.60,
            _world=self._world,
        )

        neck = Neck(
            name=PrefixedName("neck", prefix=self.name.name),
            sensors=[camera],
            root=self._world.get_body_by_name("head_pan_link"),
            tip=self._world.get_body_by_name("head_tilt_link"),
            pitch_body=self._world.get_body_by_name("head_tilt_link"),
            yaw_body=self._world.get_body_by_name("head_pan_link"),
            _world=self._world,
        )
        self.add_neck(neck)

        # Create torso
        torso = Torso(
            name=PrefixedName("torso", prefix=self.name.name),
            root=self._world.get_body_by_name("torso_lift_link"),
            tip=self._world.get_body_by_name("torso_lift_link"),
            _world=self._world,
        )
        self.add_torso(torso)

        # Create the robot base
        base = Base(
            name=PrefixedName("base", prefix=self.name.name),
            root=self._world.get_body_by_name("base_link"),
            tip=self._world.get_body_by_name("base_link"),
            _world=self._world,
        )

        self.add_base(base)

    def _setup_joint_states(self):
        right_arm_park = JointState.from_mapping(
            name=PrefixedName("right_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    [
                        c
                        for c in self.right_arm.connections
                        if type(c) != FixedConnection
                    ],
                    [-1.712, -0.256, -1.463, -2.12, 1.766, -0.07, 0.051],
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
                    [
                        1.712,
                        -0.264,
                        1.38,
                        -2.12,
                        16.996 + 3.14159,
                        -0.073,
                        0.0,
                    ],
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
            mapping=dict(zip(left_gripper_joints, [0.548, 0.548])),
            state_type=GripperState.OPEN,
        )

        left_gripper_close = JointState.from_mapping(
            name=PrefixedName("left_gripper_close", prefix=self.name.name),
            mapping=dict(zip(left_gripper_joints, [0.0, 0.0])),
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
            mapping=dict(zip(right_gripper_joints, [0.548, 0.548])),
            state_type=GripperState.OPEN,
        )

        right_gripper_close = JointState.from_mapping(
            name=PrefixedName("right_gripper_close", prefix=self.name.name),
            mapping=dict(zip(right_gripper_joints, [0.0, 0.0])),
            state_type=GripperState.CLOSE,
        )

        self.right_arm.manipulator.add_joint_state(right_gripper_close)
        self.right_arm.manipulator.add_joint_state(right_gripper_open)

        torso_joint = [self._world.get_connection_by_name("torso_lift_joint")]

        torso_low = JointState.from_mapping(
            name=PrefixedName("torso_low", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.0])),
            state_type=TorsoState.LOW,
        )

        torso_mid = JointState.from_mapping(
            name=PrefixedName("torso_mid", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.15])),
            state_type=TorsoState.MID,
        )

        torso_high = JointState.from_mapping(
            name=PrefixedName("torso_high", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.3])),
            state_type=TorsoState.HIGH,
        )

        self.torso.add_joint_state(torso_low)
        self.torso.add_joint_state(torso_mid)
        self.torso.add_joint_state(torso_high)

    def _setup_velocity_limits(self):
        vel_limits = defaultdict(
            lambda: 1.0,
            {
                self._world.get_connection_by_name("head_tilt_joint"): 3.5,
                self._world.get_connection_by_name("r_shoulder_pan_joint"): 0.15,
                self._world.get_connection_by_name("l_shoulder_pan_joint"): 0.15,
                self._world.get_connection_by_name("r_shoulder_lift_joint"): 0.2,
                self._world.get_connection_by_name("l_shoulder_lift_joint"): 0.2,
            },
        )
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)

    def _setup_hardware_interfaces(self):
        controlled_joints = [
            "torso_lift_joint",
            "head_pan_joint",
            "head_tilt_joint",
            "r_shoulder_pan_joint",
            "r_shoulder_lift_joint",
            "r_upper_arm_roll_joint",
            "r_forearm_roll_joint",
            "r_elbow_flex_joint",
            "r_wrist_flex_joint",
            "r_wrist_roll_joint",
            "l_shoulder_pan_joint",
            "l_shoulder_lift_joint",
            "l_upper_arm_roll_joint",
            "l_forearm_roll_joint",
            "l_elbow_flex_joint",
            "l_wrist_flex_joint",
            "l_wrist_roll_joint",
        ]
        for joint_name in controlled_joints:
            connection: ActiveConnection = self._world.get_connection_by_name(
                joint_name
            )
            connection.has_hardware_interface = True
