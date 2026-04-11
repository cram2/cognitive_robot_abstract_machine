import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Self

from importlib.resources import files
from pathlib import Path

from semantic_digital_twin.robots.robot_parts import (
    Arm,
    Finger,
    ParallelGripper,
    Camera,
    Torso,
    MobileBase,
    FieldOfView,
)
from semantic_digital_twin.robots.abstract_robot import (
    HasArms,
    AbstractRobot,
    HasOneArm,
    HasTorso,
    HasMobileBase,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    SelfCollisionMatrixRule,
    AvoidExternalCollisions,
)
from semantic_digital_twin.datastructures.definitions import (
    StaticJointState,
    GripperState,
    TorsoState,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Quaternion
from semantic_digital_twin.spatial_types.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection


@dataclass(eq=False)
class Stretch(AbstractRobot, HasOneArm, HasTorso, HasMobileBase):
    """
    Class that describes the Stretch Robot.
    """

    @classmethod
    def _get_robot_root_body(cls, world: World) -> Self:
        return world.get_body_by_name("base_link")

    def _setup_arm_semantic_annotations(self):
        # Create arm
        gripper_thumb = Finger.create_and_add_to_world(
            name=PrefixedName("gripper_thumb", prefix=self.name.name),
            root_name="link_gripper_finger_left",
            tip_name="link_gripper_fingertip_left",
            world=self._world,
        )

        gripper_finger = Finger.create_and_add_to_world(
            name=PrefixedName("gripper_finger", prefix=self.name.name),
            root_name="link_gripper_finger_right",
            tip_name="link_gripper_fingertip_right",
            world=self._world,
        )

        gripper = ParallelGripper.create_and_add_to_world(
            name=PrefixedName("gripper", prefix=self.name.name),
            root_name="link_straight_gripper",
            tool_frame_name="link_grasp_center",
            front_facing_orientation=Quaternion(0, 0, 0, 1),
            thumb=gripper_thumb,
            finger=gripper_finger,
            world=self._world,
        )

        arm = Arm.create_and_add_to_world(
            name=PrefixedName("arm", prefix=self.name.name),
            root_name="link_mast",
            tip_name="link_wrist_roll",
            manipulator=gripper,
            world=self._world,
        )

        self.add_arm(arm)

    def _setup_arm_hardware_interfaces(self):
        pass

    def _setup_arm_joint_state(self):
        arm_park = JointState.from_mapping(
            name=PrefixedName("arm_park", prefix=self.name.name),
            mapping={self._world.get_connection_by_name("joint_lift"): 0.5},
            state_type=StaticJointState.PARK,
        )

        self.arm.add_joint_state(arm_park)

        gripper_joints = [
            self._world.get_connection_by_name("joint_gripper_finger_left"),
            self._world.get_connection_by_name("joint_gripper_finger_right"),
        ]

        gripper_open = JointState.from_mapping(
            name=PrefixedName("gripper_open", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.59, 0.59])),
            state_type=GripperState.OPEN,
        )

        gripper_close = JointState.from_mapping(
            name=PrefixedName("gripper_close", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.0, 0.0])),
            state_type=GripperState.CLOSE,
        )

        self.arm.manipulator.add_joint_state(gripper_open)
        self.arm.manipulator.add_joint_state(gripper_close)

    def _setup_torso_semantic_annotations(self):

        # I just took the default FoV of the PR2 for now, as there were none before
        field_of_view = FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049)

        # Create camera and neck
        camera_color = Camera.create_and_add_to_world(
            name=PrefixedName("camera_color_optical_frame", prefix=self.name.name),
            root_name="camera_color_optical_frame",
            forward_facing_axis=Vector3(0, 0, 1),
            minimal_height=1.322,
            maximal_height=1.322,
            world=self._world,
            field_of_view=field_of_view,
            default_camera=True,
        )

        camera_depth = Camera.create_and_add_to_world(
            name=PrefixedName("camera_depth_optical_frame", prefix=self.name.name),
            root_name="camera_depth_optical_frame",
            forward_facing_axis=Vector3(0, 0, 1),
            minimal_height=1.307,
            maximal_height=1.307,
            world=self._world,
            field_of_view=field_of_view,
            default_camera=False,
        )

        camera_infra1 = Camera.create_and_add_to_world(
            name=PrefixedName("camera_infra1_optical_frame", prefix=self.name.name),
            root_name="camera_infra1_optical_frame",
            forward_facing_axis=Vector3(0, 0, 1),
            minimal_height=1.307,
            maximal_height=1.307,
            world=self._world,
            field_of_view=field_of_view,
            default_camera=False,
        )

        camera_infra2 = Camera.create_and_add_to_world(
            name=PrefixedName("camera_infra2_optical_frame", prefix=self.name.name),
            root_name="camera_infra2_optical_frame",
            forward_facing_axis=Vector3(0, 0, 1),
            minimal_height=1.257,
            maximal_height=1.257,
            world=self._world,
            field_of_view=field_of_view,
            default_camera=False,
        )

        # Create torso
        torso = Torso.create_and_add_to_world(
            name=PrefixedName("torso", prefix=self.name.name),
            root_name="link_mast",
            tip_name="link_lift",
            world=self._world,
            sensors=[camera_color, camera_depth, camera_infra1, camera_infra2],
        )
        self.add_torso(torso)

    def _setup_torso_hardware_interfaces(self):
        pass

    def _setup_torso_joint_state(self):
        torso_joint = [self._world.get_connection_by_name("joint_lift")]

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
            mapping=dict(zip(torso_joint, [1.0])),
            state_type=TorsoState.HIGH,
        )

        self.torso.add_joint_state(torso_low)
        self.torso.add_joint_state(torso_mid)
        self.torso.add_joint_state(torso_high)

    def _setup_mobile_base_semantic_annotations(self):
        base = MobileBase.create_and_add_to_world(
            name=PrefixedName("base", prefix=self.name.name),
            root_name="base_link",
            world=self._world,
            main_axis=Vector3(0, -1, 0, self._world.get_body_by_name("base_link")),
            full_body_controlled=True,
        )

        self.add_mobile_base(base)

    def _setup_collision_rules(self):
        srdf_path = os.path.join(
            Path(files("semantic_digital_twin")).parent.parent,
            "resources",
            "collision_configs",
            "stretch.srdf",
        )
        self._world.collision_manager.ignore_collision_rules.append(
            SelfCollisionMatrixRule.from_collision_srdf(srdf_path, self._world)
        )
        self._world.collision_manager.add_default_rule(
            AvoidExternalCollisions(
                buffer_zone_distance=0.05,
                violated_distance=0.0,
                robot=self,
            )
        )

    def _setup_velocity_limits(self):
        vel_limits = defaultdict(lambda: 1)
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)

    def _setup_other_hardware_interfaces(self):
        controlled_joints = [
            "joint_gripper_finger_left",
            "joint_gripper_finger_right",
            "joint_right_wheel",
            "joint_left_wheel",
            "joint_lift",
            "joint_arm_l3",
            "joint_arm_l2",
            "joint_arm_l1",
            "joint_arm_l0",
            "joint_wrist_yaw",
            "joint_head_pan",
            "joint_head_tilt",
        ]
        for joint_name in controlled_joints:
            connection: ActiveConnection = self._world.get_connection_by_name(
                joint_name
            )
            connection.has_hardware_interface = True
