import os
from collections import defaultdict
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

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
    def setup_default_configuration(cls, world: World):
        return cls(root=world.get_body_by_name("wide_stereo_optical_frame"))


@dataclass(eq=False)
class PR2Finger(Finger):

    @classmethod
    def setup_default_configuration(cls, world: World):
        raise NotImplementedError

    @classmethod
    def setup_right_gripper_left_finger_configuration(cls, world: World):
        return cls(
            root=world.get_body_by_name("r_gripper_l_finger_link"),
            tip=world.get_body_by_name("r_gripper_l_finger_tip_link"),
        )

    @classmethod
    def setup_right_gripper_right_finger_configuration(cls, world: World):
        return cls(
            root=world.get_body_by_name("r_gripper_r_finger_link"),
            tip=world.get_body_by_name("r_gripper_r_finger_tip_link"),
        )

    @classmethod
    def setup_left_gripper_left_finger_configuration(cls, world: World):
        return cls(
            root=world.get_body_by_name("l_gripper_l_finger_link"),
            tip=world.get_body_by_name("l_gripper_l_finger_tip_link"),
        )

    @classmethod
    def setup_left_gripper_right_finger_configuration(cls, world: World):
        return cls(
            root=world.get_body_by_name("l_gripper_r_finger_link"),
            tip=world.get_body_by_name("l_gripper_r_finger_tip_link"),
        )


@dataclass(eq=False)
class PR2Gripper(ParallelGripper):

    @classmethod
    def setup_default_configuration(cls, world: World):
        raise NotImplementedError

    @classmethod
    def setup_right_gripper_configuration(cls, world: World):
        return cls(
            root=world.get_body_by_name("l_gripper_palm_link"),
            tool_frame=world.get_body_by_name("l_gripper_tool_frame"),
            front_facing_orientation=Quaternion(0, 0, 0, 1),
            thumb=PR2Finger.setup_right_gripper_right_finger_configuration(world),
            fingers=[PR2Finger.setup_right_gripper_left_finger_configuration(world)],
        )

    @classmethod
    def setup_left_gripper_configuration(cls, world: World):
        return cls(
            root=world.get_body_by_name("r_gripper_palm_link"),
            tool_frame=world.get_body_by_name("r_gripper_tool_frame"),
            front_facing_orientation=Quaternion(0, 0, 0, 1),
            thumb=PR2Finger.setup_left_gripper_right_finger_configuration(world),
            fingers=[PR2Finger.setup_left_gripper_left_finger_configuration(world)],
        )

    def _setup_finger_semantic_annotations(self, world: World):
        pass


@dataclass(eq=False)
class PR2Neck(Neck, HasCameras):

    @classmethod
    def setup_default_configuration(cls, world: World):
        return cls(
            root=world.get_body_by_name("torso_lift_link"),
            tip=world.get_body_by_name("head_tilt_link"),
            sensors=[PR2KinectV1.setup_default_configuration(world)],
        )

    def _setup_sensors_semantic_annotations(self, world: World):
        pass


@dataclass(eq=False)
class PR2Arm(Arm, HasParallelGripper):

    @classmethod
    def setup_default_configuration(cls, world: World):
        raise NotImplementedError

    @classmethod
    def setup_left_arm_configuration(cls, world: World):
        return cls(
            root=world.get_body_by_name("torso_lift_link"),
            tip=world.get_body_by_name("l_wrist_roll_link"),
            end_effector=PR2Gripper.setup_left_gripper_configuration(world),
        )

    @classmethod
    def setup_right_arm_configuration(cls, world: World):
        self = cls(
            root=world.get_body_by_name("torso_lift_link"),
            tip=world.get_body_by_name("r_wrist_roll_link"),
        )
        self._setup_specifications(world)

    def _setup_end_effector_semantic_annotations(self, world: World):
        PR2Gripper.setup_right_gripper_configuration(world)


@dataclass(eq=False)
class PR2Torso(Torso, HasLeftRightArm, HasNeck):

    @classmethod
    def setup_default_configuration(cls, world: World):
        return cls(
            root=world.get_body_by_name("base_link"),
            tip=world.get_body_by_name("torso_lift_link"),
        )

    def _setup_arm_semantic_annotations(self, world: World):
        return [
            PR2Arm.setup_left_arm_configuration(self._world),
            PR2Arm.setup_right_arm_configuration(self._world),
        ]

    def _setup_neck_semantic_annotations(self, world: World):
        return PR2Neck.setup_default_configuration(self._world)


@dataclass(eq=False)
class PR2MobileBase(MobileBase, HasTorso):

    @classmethod
    def _setup_default_torso_semantic_annotation(cls, world: World) -> Torso:
        return PR2Torso.setup_default_configuration(world)

    @classmethod
    def setup_default_configuration(cls, world: World):

        return cls(
            root=world.get_body_by_name("base_link"),
            torso=cls._setup_default_torso_semantic_annotation(world),
        )


@dataclass(eq=False)
class PR2(AbstractRobot, HasMobileBase):

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "base_footprint"

    def _setup_mobile_base_semantic_annotation(self, world: World) -> MobileBase:
        return PR2MobileBase.setup_default_configuration(world)

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
