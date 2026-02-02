from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass

from pkg_resources import resource_filename

from .robot_mixins import HasNeck, SpecifiesLeftRightArm
from ..collision_checking.collision_matrix import MaxAvoidedCollisionsOverride
from ..collision_checking.collision_rules import (
    SelfCollisionMatrixRule,
    AvoidAllCollisions,
)
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
from ..world_description.connections import ActiveConnection


@dataclass(eq=False)
class PR2(AbstractRobot, SpecifiesLeftRightArm, HasNeck):
    """
    Represents the Personal Robot 2 (PR2), which was originally created by Willow Garage.
    The PR2 robot consists of two arms, each with a parallel gripper, a head with a camera, and a prismatic torso
    """

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
        self.high_priority_collision_rules.append(
            SelfCollisionMatrixRule.from_collision_srdf(srdf_path, self._world)
        )

        self.default_collision_rules.append(
            AvoidAllCollisions(
                buffer_zone_distance=0.1,
                violated_distance=0.0,
                bodies=self.bodies_with_collisions,
            )
        )

        self.default_collision_rules.append(
            AvoidAllCollisions(
                buffer_zone_distance=0.05,
                violated_distance=0.0,
                bodies=[self.left_arm.bodies_with_collisions],
            )
        )
        self.default_collision_rules.append(
            AvoidAllCollisions(
                buffer_zone_distance=0.05,
                violated_distance=0.0,
                bodies=[self.right_arm.bodies_with_collisions],
            )
        )
        self.default_collision_rules.append(
            AvoidAllCollisions(
                buffer_zone_distance=0.2,
                violated_distance=0.05,
                bodies=[self._world.get_body_by_name("base_link")],
            )
        )
        self.max_avoided_bodies_rules.append(
            MaxAvoidedCollisionsOverride(
                2, bodies={self._world.get_body_by_name("base_link")}
            )
        )
        self.max_avoided_bodies_rules.append(
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
            )
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
            sensors={camera},
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

        self._world.add_semantic_annotation(self)

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
