from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Self

from typing_extensions import List

from giskardpy.model.collision_matrix_manager import CollisionRequest
from .robot_mixins import HasNeck, SpecifiesLeftRightArm
from ..datastructures.definitions import StaticJointState, GripperState, TorsoState
from ..collision_checking.collision_detector import CollisionCheck
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
from ..world_description.world_entity import CollisionCheckingConfig


@dataclass(eq=False)
class PR2(AbstractRobot, SpecifiesLeftRightArm, HasNeck):
    """
    Represents the Personal Robot 2 (PR2), which was originally created by Willow Garage.
    The PR2 robot consists of two arms, each with a parallel gripper, a head with a camera, and a prismatic torso
    """

    def setup_collision_config(self):
        """
        Loads the SRDF file for the PR2 robot, if it exists.
        """
        srdf_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "..",
            "resources",
            "collision_configs",
            "pr2.srdf",
        )
        self._world.load_collision_srdf(srdf_path)

        frozen_joints = ["r_gripper_l_finger_joint", "l_gripper_l_finger_joint"]
        for joint_name in frozen_joints:
            c: ActiveConnection = self._world.get_connection_by_name(joint_name)
            c.frozen_for_collision_avoidance = True

        for body in self.bodies_with_collisions:
            collision_config = CollisionCheckingConfig(
                buffer_zone_distance=0.1, violated_distance=0.0
            )
            body.set_static_collision_config(collision_config)

        for joint_name in ["r_wrist_roll_joint", "l_wrist_roll_joint"]:
            connection: ActiveConnection = self._world.get_connection_by_name(
                joint_name
            )
            collision_config = CollisionCheckingConfig(
                buffer_zone_distance=0.05, violated_distance=0.0, max_avoided_bodies=4
            )
            connection.set_static_collision_config_for_direct_child_bodies(
                collision_config
            )

        for joint_name in ["r_wrist_flex_joint", "l_wrist_flex_joint"]:
            connection: ActiveConnection = self._world.get_connection_by_name(
                joint_name
            )
            collision_config = CollisionCheckingConfig(
                buffer_zone_distance=0.05, violated_distance=0.0, max_avoided_bodies=2
            )
            connection.set_static_collision_config_for_direct_child_bodies(
                collision_config
            )
        for joint_name in ["r_elbow_flex_joint", "l_elbow_flex_joint"]:
            connection: ActiveConnection = self._world.get_connection_by_name(
                joint_name
            )
            collision_config = CollisionCheckingConfig(
                buffer_zone_distance=0.05, violated_distance=0.0, max_avoided_bodies=1
            )
            connection.set_static_collision_config_for_direct_child_bodies(
                collision_config
            )
        for joint_name in ["r_forearm_roll_joint", "l_forearm_roll_joint"]:
            connection: ActiveConnection = self._world.get_connection_by_name(
                joint_name
            )
            collision_config = CollisionCheckingConfig(
                buffer_zone_distance=0.025, violated_distance=0.0, max_avoided_bodies=1
            )
            connection.set_static_collision_config_for_direct_child_bodies(
                collision_config
            )

        collision_config = CollisionCheckingConfig(
            buffer_zone_distance=0.2, violated_distance=0.1, max_avoided_bodies=2
        )
        self.drive.set_static_collision_config_for_direct_child_bodies(collision_config)

    @classmethod
    def from_world(cls, world: World) -> Self:
        """
        Creates a PR2 robot view from the given world.

        :param world: The world from which to create the robot view.

        :return: A PR2 robot view.
        """

        with world.modify_world():
            robot = cls(
                name=PrefixedName(name="pr2", prefix=world.name),
                root=world.get_body_by_name("base_footprint"),
                _world=world,
            )

            # Create left arm
            left_gripper_thumb = Finger(
                name=PrefixedName("left_gripper_thumb", prefix=robot.name.name),
                root=world.get_body_by_name("l_gripper_l_finger_link"),
                tip=world.get_body_by_name("l_gripper_l_finger_tip_link"),
                _world=world,
            )

            left_gripper_finger = Finger(
                name=PrefixedName("left_gripper_finger", prefix=robot.name.name),
                root=world.get_body_by_name("l_gripper_r_finger_link"),
                tip=world.get_body_by_name("l_gripper_r_finger_tip_link"),
                _world=world,
            )

            left_gripper = ParallelGripper(
                name=PrefixedName("left_gripper", prefix=robot.name.name),
                root=world.get_body_by_name("l_gripper_palm_link"),
                tool_frame=world.get_body_by_name("l_gripper_tool_frame"),
                front_facing_orientation=Quaternion(0, 0, 0, 1),
                front_facing_axis=Vector3(1, 0, 0),
                thumb=left_gripper_thumb,
                finger=left_gripper_finger,
                _world=world,
            )
            left_arm = Arm(
                name=PrefixedName("left_arm", prefix=robot.name.name),
                root=world.get_body_by_name("torso_lift_link"),
                tip=world.get_body_by_name("l_wrist_roll_link"),
                manipulator=left_gripper,
                _world=world,
            )

            robot.add_arm(left_arm)

            # Create right arm
            right_gripper_thumb = Finger(
                name=PrefixedName("right_gripper_thumb", prefix=robot.name.name),
                root=world.get_body_by_name("r_gripper_l_finger_link"),
                tip=world.get_body_by_name("r_gripper_l_finger_tip_link"),
                _world=world,
            )
            right_gripper_finger = Finger(
                name=PrefixedName("right_gripper_finger", prefix=robot.name.name),
                root=world.get_body_by_name("r_gripper_r_finger_link"),
                tip=world.get_body_by_name("r_gripper_r_finger_tip_link"),
                _world=world,
            )
            right_gripper = ParallelGripper(
                name=PrefixedName("right_gripper", prefix=robot.name.name),
                root=world.get_body_by_name("r_gripper_palm_link"),
                tool_frame=world.get_body_by_name("r_gripper_tool_frame"),
                front_facing_orientation=Quaternion(0, 0, 0, 1),
                front_facing_axis=Vector3(1, 0, 0),
                thumb=right_gripper_thumb,
                finger=right_gripper_finger,
                _world=world,
            )
            right_arm = Arm(
                name=PrefixedName("right_arm", prefix=robot.name.name),
                root=world.get_body_by_name("torso_lift_link"),
                tip=world.get_body_by_name("r_wrist_roll_link"),
                manipulator=right_gripper,
                _world=world,
            )

            robot.add_arm(right_arm)

            # Create camera and neck
            camera = Camera(
                name=PrefixedName("wide_stereo_optical_frame", prefix=robot.name.name),
                root=world.get_body_by_name("wide_stereo_optical_frame"),
                forward_facing_axis=Vector3(0, 0, 1),
                field_of_view=FieldOfView(
                    horizontal_angle=0.99483, vertical_angle=0.75049
                ),
                minimal_height=1.27,
                maximal_height=1.60,
                _world=world,
            )

            neck = Neck(
                name=PrefixedName("neck", prefix=robot.name.name),
                sensors=[camera],
                root=world.get_body_by_name("head_pan_link"),
                tip=world.get_body_by_name("head_tilt_link"),
                pitch_body=world.get_body_by_name("head_tilt_link"),
                yaw_body=world.get_body_by_name("head_pan_link"),
                _world=world,
            )
            robot.add_neck(neck)

            # Create torso
            torso = Torso(
                name=PrefixedName("torso", prefix=robot.name.name),
                root=world.get_body_by_name("torso_lift_link"),
                tip=world.get_body_by_name("torso_lift_link"),
                _world=world,
            )
            robot.add_torso(torso)

            # Create states
            right_arm_park = JointState.from_mapping(
                name=PrefixedName("right_park", prefix=robot.name.name),
                mapping=dict(
                    zip(
                        [
                            c
                            for c in right_arm.connections
                            if type(c) != FixedConnection
                        ],
                        [-1.712, -0.256, -1.463, -2.12, 1.766, -0.07, 0.051],
                    )
                ),
                state_type=StaticJointState.PARK,
            )

            right_arm.add_joint_state(right_arm_park)

            left_arm_park = JointState.from_mapping(
                name=PrefixedName("left_park", prefix=robot.name.name),
                mapping=dict(
                    zip(
                        [c for c in left_arm.connections if type(c) != FixedConnection],
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

            left_arm.add_joint_state(left_arm_park)

            left_gripper_joints = [
                c for c in left_gripper.connections if type(c) != FixedConnection
            ]

            left_gripper_open = JointState.from_mapping(
                name=PrefixedName("left_gripper_open", prefix=robot.name.name),
                mapping=dict(zip(left_gripper_joints, [0.548, 0.548])),
                state_type=GripperState.OPEN,
            )

            left_gripper_close = JointState.from_mapping(
                name=PrefixedName("left_gripper_close", prefix=robot.name.name),
                mapping=dict(zip(left_gripper_joints, [0.0, 0.0])),
                state_type=GripperState.CLOSE,
            )

            left_gripper.add_joint_state(left_gripper_close)
            left_gripper.add_joint_state(left_gripper_open)

            right_gripper_joints = [
                c for c in right_gripper.connections if type(c) != FixedConnection
            ]

            right_gripper_open = JointState.from_mapping(
                name=PrefixedName("right_gripper_open", prefix=robot.name.name),
                mapping=dict(zip(right_gripper_joints, [0.548, 0.548])),
                state_type=GripperState.OPEN,
            )

            right_gripper_close = JointState.from_mapping(
                name=PrefixedName("right_gripper_close", prefix=robot.name.name),
                mapping=dict(zip(right_gripper_joints, [0.0, 0.0])),
                state_type=GripperState.CLOSE,
            )

            right_gripper.add_joint_state(right_gripper_close)
            right_gripper.add_joint_state(right_gripper_open)

            torso_joint = [world.get_connection_by_name("torso_lift_joint")]

            torso_low = JointState.from_mapping(
                name=PrefixedName("torso_low", prefix=robot.name.name),
                mapping=dict(zip(torso_joint, [0.0])),
                state_type=TorsoState.LOW,
            )

            torso_mid = JointState.from_mapping(
                name=PrefixedName("torso_mid", prefix=robot.name.name),
                mapping=dict(zip(torso_joint, [0.15])),
                state_type=TorsoState.MID,
            )

            torso_high = JointState.from_mapping(
                name=PrefixedName("torso_high", prefix=robot.name.name),
                mapping=dict(zip(torso_joint, [0.3])),
                state_type=TorsoState.HIGH,
            )

            torso.add_joint_state(torso_low)
            torso.add_joint_state(torso_mid)
            torso.add_joint_state(torso_high)

            # Create the robot base
            base = Base(
                name=PrefixedName("base", prefix=robot.name.name),
                root=world.get_body_by_name("base_link"),
                tip=world.get_body_by_name("base_link"),
                _world=world,
            )

            robot.add_base(base)

            world.add_semantic_annotation(robot)

            vel_limits = defaultdict(
                lambda: 1.0,
                {
                    world.get_connection_by_name("head_tilt_joint"): 3.5,
                    world.get_connection_by_name("r_shoulder_pan_joint"): 0.15,
                    world.get_connection_by_name("l_shoulder_pan_joint"): 0.15,
                    world.get_connection_by_name("r_shoulder_lift_joint"): 0.2,
                    world.get_connection_by_name("l_shoulder_lift_joint"): 0.2,
                },
            )
            robot.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)

        return robot
