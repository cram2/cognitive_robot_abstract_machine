import os
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Self

from semantic_digital_twin.robots.robot_parts import (
    MobileBase,
    Torso,
    Arm,
    Neck,
    ParallelGripper,
    Finger,
    Camera,
)
from semantic_digital_twin.robots.abstract_robot import (
    HasMobileBase,
    AbstractRobot,
    HasTorso,
    HasLeftRightArm,
    HasNeck,
    HasParallelGripper,
    HasCameras,
)

from semantic_digital_twin.collision_checking.collision_matrix import (
    MaxAvoidedCollisionsOverride,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    SelfCollisionMatrixRule,
    AvoidExternalCollisions,
    AvoidSelfCollisions,
)
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(eq=False)
class PR2KinectV1(Camera):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        camera = cls(root=world.get_body_by_name("wide_stereo_optical_frame"))
        world.add_semantic_annotation(camera)
        return camera


@dataclass(eq=False)
class PR2Finger(Finger, ABC): ...


@dataclass(eq=False)
class PR2RightGripperLeftFinger(PR2Finger):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("r_gripper_l_finger_link"),
            tip=world.get_body_by_name("r_gripper_l_finger_tip_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class PR2RightGripperRightFinger(PR2Finger):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("r_gripper_r_finger_link"),
            tip=world.get_body_by_name("r_gripper_r_finger_tip_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class PR2LeftGripperLeftFinger(PR2Finger):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("l_gripper_l_finger_link"),
            tip=world.get_body_by_name("l_gripper_l_finger_tip_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class PR2LeftGripperRightFinger(PR2Finger):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("l_gripper_r_finger_link"),
            tip=world.get_body_by_name("l_gripper_r_finger_tip_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class PR2RightGripper(ParallelGripper):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        right_gripper = cls(
            root=world.get_body_by_name("r_gripper_palm_link"),
            tool_frame=world.get_body_by_name("r_gripper_tool_frame"),
            front_facing_orientation=Quaternion(0, 0, 0, 1),
        )
        world.add_semantic_annotation(right_gripper)
        return right_gripper

    def setup_finger_semantic_annotations(self):
        thumb = PR2RightGripperLeftFinger.setup_default_configuration_in_world(
            self._world
        )
        self.add_thumb(thumb)
        finger = PR2RightGripperRightFinger.setup_default_configuration_in_world(
            self._world
        )
        self.add_finger(finger)


@dataclass(eq=False)
class PR2LeftGripper(ParallelGripper):
    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        left_gripper = cls(
            root=world.get_body_by_name("l_gripper_palm_link"),
            tool_frame=world.get_body_by_name("l_gripper_tool_frame"),
            front_facing_orientation=Quaternion(0, 0, 0, 1),
        )
        world.add_semantic_annotation(left_gripper)
        return left_gripper

    def setup_finger_semantic_annotations(self):
        thumb = PR2LeftGripperRightFinger.setup_default_configuration_in_world(
            self._world
        )
        self.add_thumb(thumb)
        finger = PR2LeftGripperLeftFinger.setup_default_configuration_in_world(
            self._world
        )
        self.add_finger(finger)


@dataclass(eq=False)
class PR2Neck(Neck, HasCameras):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        neck = cls(
            root=world.get_body_by_name("torso_lift_link"),
            tip=world.get_body_by_name("head_tilt_link"),
        )
        world.add_semantic_annotation(neck)
        return neck

    def setup_sensor_semantic_annotations(self):
        neck = PR2KinectV1.setup_default_configuration_in_world(self._world)
        self.add_sensor(neck)


@dataclass(eq=False)
class PR2LeftArm(Arm, HasParallelGripper):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        left_arm = cls(
            root=world.get_body_by_name("torso_lift_link"),
            tip=world.get_body_by_name("l_wrist_roll_link"),
        )
        world.add_semantic_annotation(left_arm)
        return left_arm

    def setup_end_effector_semantic_annotation(self):
        gripper = PR2LeftGripper.setup_default_configuration_in_world(self._world)
        self.add_end_effector(gripper)
        gripper.setup_finger_semantic_annotations()


@dataclass(eq=False)
class PR2RightArm(Arm, HasParallelGripper):
    @classmethod
    def setup_default_configuration_in_world(cls, world: World) -> Self:
        right_arm = cls(
            root=world.get_body_by_name("torso_lift_link"),
            tip=world.get_body_by_name("r_wrist_roll_link"),
        )
        world.add_semantic_annotation(right_arm)
        return right_arm

    def setup_end_effector_semantic_annotation(self):
        gripper = PR2RightGripper.setup_default_configuration_in_world(self._world)
        self.add_end_effector(gripper)
        gripper.setup_finger_semantic_annotations()


@dataclass(eq=False)
class PR2Torso(Torso, HasLeftRightArm, HasNeck):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World) -> Self:
        torso = cls(
            root=world.get_body_by_name("base_link"),
            tip=world.get_body_by_name("torso_lift_link"),
        )
        world.add_semantic_annotation(torso)
        return torso

    def setup_arm_semantic_annotations(self):
        left_arm = PR2LeftArm.setup_default_configuration_in_world(self._world)
        self.add_arm(left_arm)
        left_arm.setup_end_effector_semantic_annotation()

        right_arm = PR2RightArm.setup_default_configuration_in_world(self._world)
        self.add_arm(right_arm)
        right_arm.setup_end_effector_semantic_annotation()

    def setup_neck_semantic_annotations(self):
        neck = PR2Neck.setup_default_configuration_in_world(self._world)
        self.add_neck(neck)


@dataclass(eq=False)
class PR2MobileBase(MobileBase, HasTorso):

    def setup_default_torso_semantic_annotation(self):
        torso = PR2Torso.setup_default_configuration_in_world(self._world)
        torso.setup_arm_semantic_annotations()
        torso.setup_neck_semantic_annotations()
        self.add_torso(torso)

    @classmethod
    def setup_default_configuration_in_world(cls, world: World) -> Self:
        self = cls(
            root=world.get_body_by_name("base_link"),
        )
        world.add_semantic_annotation(self)
        return self


@dataclass(eq=False)
class PR2(AbstractRobot, HasMobileBase):

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "base_footprint"

    def setup_mobile_base_semantic_annotation(self):
        mobile_base = PR2MobileBase.setup_default_configuration_in_world(self._world)
        mobile_base.setup_default_torso_semantic_annotation()
        self.add_mobile_base(mobile_base)

    def setup_robot_part_semantic_annotations(self):
        self.setup_mobile_base_semantic_annotation()

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

    def _setup_collision_rules(self):
        """
        Loads the SRDF file for the PR2 robot, if it exists.
        """
        srdf_path = os.path.join(
            Path(files("semantic_digital_twin")).parent.parent,
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
                    body_subset=set(self.left_arm.bodies_with_collision)
                    | set(self.right_arm.bodies_with_collision),
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
