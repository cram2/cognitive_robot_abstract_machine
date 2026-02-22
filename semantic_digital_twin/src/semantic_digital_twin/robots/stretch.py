import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Self

from pkg_resources import resource_filename

from .abstract_robot import (
    AbstractRobot,
    Arm,
    Neck,
    Finger,
    ParallelGripper,
    Camera,
    Torso,
    Base,
)
from .robot_mixins import HasNeck, HasArms
from ..collision_checking.collision_rules import (
    SelfCollisionMatrixRule,
    AvoidExternalCollisions,
)
from ..datastructures.definitions import StaticJointState, GripperState, TorsoState
from ..datastructures.joint_state import JointState
from ..datastructures.prefixed_name import PrefixedName
from ..spatial_types import Quaternion
from ..spatial_types.spatial_types import Vector3
from ..world import World
from ..world_description.connections import ActiveConnection


@dataclass(eq=False)
class Stretch(AbstractRobot, HasArms, HasNeck):
    """
    Class that describes the Stretch Robot.
    """

    @property
    def arm(self) -> Arm:
        return self.arms[0]

    @classmethod
    def _init_empty_robot(cls, world: World) -> Self:
        return cls(
            name=PrefixedName("stretch", prefix=world.name),
            root=world.get_body_by_name("base_link"),
            _world=world,
        )

    def _setup_semantic_annotations(self):
        # Create arm
        gripper_thumb = Finger(
            name=PrefixedName("gripper_thumb", prefix=self.name.name),
            root=self._world.get_body_by_name("link_gripper_finger_left"),
            tip=self._world.get_body_by_name("link_gripper_fingertip_left"),
            _world=self._world,
        )

        gripper_finger = Finger(
            name=PrefixedName("gripper_finger", prefix=self.name.name),
            root=self._world.get_body_by_name("link_gripper_finger_right"),
            tip=self._world.get_body_by_name("link_gripper_fingertip_right"),
            _world=self._world,
        )

        gripper = ParallelGripper(
            name=PrefixedName("gripper", prefix=self.name.name),
            root=self._world.get_body_by_name("link_straight_gripper"),
            tool_frame=self._world.get_body_by_name("link_grasp_center"),
            front_facing_orientation=Quaternion(0, 0, 0, 1),
            front_facing_axis=Vector3(1, 0, 0),
            thumb=gripper_thumb,
            finger=gripper_finger,
            _world=self._world,
        )

        arm = Arm(
            name=PrefixedName("arm", prefix=self.name.name),
            root=self._world.get_body_by_name("link_mast"),
            tip=self._world.get_body_by_name("link_wrist_roll"),
            manipulator=gripper,
            _world=self._world,
        )

        self.add_arm(arm)

        # Create camera and neck
        camera_color = Camera(
            name=PrefixedName("camera_color_optical_frame", prefix=self.name.name),
            root=self._world.get_body_by_name("camera_color_optical_frame"),
            forward_facing_axis=Vector3(0, 0, 1),
            minimal_height=1.322,
            maximal_height=1.322,
            _world=self._world,
        )

        camera_depth = Camera(
            name=PrefixedName("camera_depth_optical_frame", prefix=self.name.name),
            root=self._world.get_body_by_name("camera_depth_optical_frame"),
            forward_facing_axis=Vector3(0, 0, 1),
            minimal_height=1.307,
            maximal_height=1.307,
            _world=self._world,
        )

        camera_infra1 = Camera(
            name=PrefixedName("camera_infra1_optical_frame", prefix=self.name.name),
            root=self._world.get_body_by_name("camera_infra1_optical_frame"),
            forward_facing_axis=Vector3(0, 0, 1),
            minimal_height=1.307,
            maximal_height=1.307,
            _world=self._world,
        )

        camera_infra2 = Camera(
            name=PrefixedName("camera_infra2_optical_frame", prefix=self.name.name),
            root=self._world.get_body_by_name("camera_infra2_optical_frame"),
            forward_facing_axis=Vector3(0, 0, 1),
            minimal_height=1.257,
            maximal_height=1.257,
            _world=self._world,
        )

        neck = Neck(
            name=PrefixedName("neck", prefix=self.name.name),
            sensors=[camera_color, camera_depth, camera_infra1, camera_infra2],
            root=self._world.get_body_by_name("link_head"),
            tip=self._world.get_body_by_name("link_head_tilt"),
            pitch_body=self._world.get_body_by_name("link_head_tilt"),
            yaw_body=self._world.get_body_by_name("link_head_pan"),
            _world=self._world,
        )
        self.add_neck(neck)

        # Create torso
        torso = Torso(
            name=PrefixedName("torso", prefix=self.name.name),
            root=self._world.get_body_by_name("link_mast"),
            tip=self._world.get_body_by_name("link_lift"),
            _world=self._world,
        )
        self.add_torso(torso)

        base = Base(
            name=PrefixedName("base", prefix=self.name.name),
            root=self._world.get_body_by_name("base_link"),
            tip=self._world.get_body_by_name("base_link"),
            _world=self._world,
            main_axis=Vector3(0, -1, 0, self._world.get_body_by_name("base_link")),
        )

        self.add_base(base)

    def _setup_collision_rules(self):
        srdf_path = os.path.join(
            resource_filename("semantic_digital_twin", "../../"),
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

    def _setup_hardware_interfaces(self):
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

    def _setup_joint_states(self):
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
        self.full_body_controlled = True
