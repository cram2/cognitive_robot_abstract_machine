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
    FieldOfView,
    Base,
)
from .robot_mixins import HasNeck, SpecifiesLeftRightArm
from ..collision_checking.collision_matrix import MaxAvoidedCollisionsOverride
from ..collision_checking.collision_rules import (
    SelfCollisionMatrixRule,
    AvoidExternalCollisions,
    AvoidSelfCollisions,
)
from ..datastructures.definitions import StaticJointState, GripperState, TorsoState
from ..datastructures.joint_state import JointState
from ..datastructures.prefixed_name import PrefixedName
from ..spatial_types import Quaternion
from ..spatial_types.spatial_types import Vector3
from ..world import World
from ..world_description.connections import FixedConnection, ActiveConnection


@dataclass(eq=False)
class Tiago(AbstractRobot, SpecifiesLeftRightArm, HasNeck):
    """
    Class that describes the Take It And Go Robot (TIAGo).
    """

    @classmethod
    def _init_empty_robot(cls, world: World) -> Self:
        return cls(
            name=PrefixedName("tiago", prefix=world.name),
            root=world.get_body_by_name("base_footprint"),
            _world=world,
            full_body_controlled=False,
        )

    def _setup_semantic_annotations(self):

        # Create left arm
        left_gripper_thumb = Finger(
            name=PrefixedName("left_gripper_thumb", prefix=self.name.name),
            root=self._world.get_body_by_name("gripper_left_base_link"),
            tip=self._world.get_body_by_name("gripper_left_left_inner_finger_pad"),
            _world=self._world,
        )

        left_gripper_finger = Finger(
            name=PrefixedName("left_gripper_finger", prefix=self.name.name),
            root=self._world.get_body_by_name("gripper_left_base_link"),
            tip=self._world.get_body_by_name("gripper_left_right_inner_finger_pad"),
            _world=self._world,
        )

        left_gripper = ParallelGripper(
            name=PrefixedName("left_gripper", prefix=self.name.name),
            root=self._world.get_body_by_name("gripper_left_base_link"),
            tool_frame=self._world.get_body_by_name("gripper_left_grasping_frame"),
            front_facing_orientation=Quaternion(0, 0, 0, 1),
            front_facing_axis=Vector3(1, 0, 0),
            thumb=left_gripper_thumb,
            finger=left_gripper_finger,
            _world=self._world,
        )
        left_arm = Arm(
            name=PrefixedName("left_arm", prefix=self.name.name),
            root=self._world.get_body_by_name("torso_lift_link"),
            tip=self._world.get_body_by_name("arm_left_tool_link"),
            manipulator=left_gripper,
            _world=self._world,
        )

        self.add_arm(left_arm)

        # Create right arm
        right_gripper_thumb = Finger(
            name=PrefixedName("right_gripper_thumb", prefix=self.name.name),
            root=self._world.get_body_by_name("gripper_right_base_link"),
            tip=self._world.get_body_by_name("gripper_right_left_inner_finger_pad"),
            _world=self._world,
        )
        right_gripper_finger = Finger(
            name=PrefixedName("right_gripper_finger", prefix=self.name.name),
            root=self._world.get_body_by_name("gripper_right_base_link"),
            tip=self._world.get_body_by_name("gripper_right_right_inner_finger_pad"),
            _world=self._world,
        )
        right_gripper = ParallelGripper(
            name=PrefixedName("right_gripper", prefix=self.name.name),
            root=self._world.get_body_by_name("gripper_right_base_link"),
            tool_frame=self._world.get_body_by_name("gripper_right_grasping_frame"),
            front_facing_orientation=Quaternion(0, 0, 0, 1),
            front_facing_axis=Vector3(1, 0, 0),
            thumb=right_gripper_thumb,
            finger=right_gripper_finger,
            _world=self._world,
        )
        right_arm = Arm(
            name=PrefixedName("right_arm", prefix=self.name.name),
            root=self._world.get_body_by_name("torso_lift_link"),
            tip=self._world.get_body_by_name("arm_right_tool_link"),
            manipulator=right_gripper,
            _world=self._world,
        )

        self.add_arm(right_arm)

        # Create camera and neck
        camera = Camera(
            name=PrefixedName("head_front_camera_optical_frame", prefix=self.name.name),
            root=self._world.get_body_by_name("head_front_camera_optical_frame"),
            forward_facing_axis=Vector3(0, 0, 1),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=1.0665,
            maximal_height=1.4165,
            _world=self._world,
        )

        neck = Neck(
            name=PrefixedName("neck", prefix=self.name.name),
            sensors=[camera],
            root=self._world.get_body_by_name("torso_lift_link"),
            tip=self._world.get_body_by_name("head_2_link"),
            pitch_body=self._world.get_body_by_name("head_2_link"),
            yaw_body=self._world.get_body_by_name("head_1_link"),
            _world=self._world,
        )
        self.add_neck(neck)

        # Create torso
        torso = Torso(
            name=PrefixedName("torso", prefix=self.name.name),
            root=self._world.get_body_by_name("torso_fixed_link"),
            tip=self._world.get_body_by_name("torso_lift_link"),
            _world=self._world,
        )
        self.add_torso(torso)
        base = Base(
            name=PrefixedName("base", prefix=self.name.name),
            root=self._world.get_body_by_name("base_link"),
            tip=self._world.get_body_by_name("base_link"),
            _world=self._world,
        )

        self.add_base(base)

    def _setup_collision_rules(self):
        srdf_path = os.path.join(
            resource_filename("semantic_digital_twin", "../../"),
            "resources",
            "collision_configs",
            "tiago_dual.srdf",
        )
        self._world.collision_manager.ignore_collision_rules.append(
            SelfCollisionMatrixRule.from_collision_srdf(srdf_path, self._world)
        )
        self._world.collision_manager.add_default_rule(
            AvoidExternalCollisions(
                buffer_zone_distance=0.08,
                violated_distance=0.03,
                robot=self,
            )
        )
        self._world.collision_manager.add_default_rule(
            AvoidSelfCollisions(
                buffer_zone_distance=0.08,
                violated_distance=0.04,
                robot=self,
            )
        )

        self._world.collision_manager.add_default_rule(
            AvoidExternalCollisions(
                buffer_zone_distance=0.2,
                violated_distance=0.1,
                robot=self,
                body_subset={self._world.get_body_by_name("base_link")},
            )
        )

        self._world.collision_manager.max_avoided_bodies_rules.append(
            MaxAvoidedCollisionsOverride(
                2, bodies={self._world.get_body_by_name("base_link")}
            )
        )
        self._world.collision_manager.max_avoided_bodies_rules.append(
            MaxAvoidedCollisionsOverride(
                4,
                bodies=set(
                    self._world.get_direct_child_bodies_with_collision(
                        self._world.get_body_by_name("arm_right_7_link")
                    )
                ),
            )
        )
        self._world.collision_manager.max_avoided_bodies_rules.append(
            MaxAvoidedCollisionsOverride(
                4,
                bodies=set(
                    self._world.get_direct_child_bodies_with_collision(
                        self._world.get_body_by_name("arm_left_7_link")
                    )
                ),
            )
        )

    def _setup_velocity_limits(self):
        vel_limits = defaultdict(lambda: 1)
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)

    def _setup_hardware_interfaces(self):
        controlled_joints = [
            "torso_lift_joint",
            "head_1_joint",
            "head_2_joint",
            "arm_left_1_joint",
            "arm_left_2_joint",
            "arm_left_3_joint",
            "arm_left_4_joint",
            "arm_left_5_joint",
            "arm_left_6_joint",
            "arm_left_7_joint",
            "arm_right_1_joint",
            "arm_right_2_joint",
            "arm_right_3_joint",
            "arm_right_4_joint",
            "arm_right_5_joint",
            "arm_right_6_joint",
            "arm_right_7_joint",
            "gripper_right_finger_joint",
            "gripper_left_finger_joint",
        ]
        for joint_name in controlled_joints:
            connection: ActiveConnection = self._world.get_connection_by_name(
                joint_name
            )
            connection.has_hardware_interface = True

    def _setup_joint_states(self):
        # Create states
        left_arm_park = JointState.from_mapping(
            name=PrefixedName("left_arm_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    [
                        c
                        for c in self.left_arm.connections
                        if type(c) != FixedConnection
                    ],
                    [0.27, -1.07, 1.5, 1.96, -2.0, 1.2, 0.5],
                )
            ),
            state_type=StaticJointState.PARK,
        )

        self.left_arm.add_joint_state(left_arm_park)

        right_arm_park = JointState.from_mapping(
            name=PrefixedName("right_arm_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    [
                        c
                        for c in self.right_arm.connections
                        if type(c) != FixedConnection
                    ],
                    [0.27, -1.07, 1.5, 1.96, -2.0, 1.2, 0.5],
                )
            ),
            state_type=StaticJointState.PARK,
        )

        self.right_arm.add_joint_state(right_arm_park)

        left_gripper_joints = [
            self._world.get_connection_by_name("gripper_left_finger_joint"),
        ]

        left_gripper_open = JointState.from_mapping(
            name=PrefixedName("left_gripper_open", prefix=self.name.name),
            mapping=dict(zip(left_gripper_joints, [0.05])),
            state_type=GripperState.OPEN,
        )

        left_gripper_close = JointState.from_mapping(
            name=PrefixedName("left_gripper_close", prefix=self.name.name),
            mapping=dict(zip(left_gripper_joints, [0.75])),
            state_type=GripperState.CLOSE,
        )

        self.left_arm.manipulator.add_joint_state(left_gripper_close)
        self.left_arm.manipulator.add_joint_state(left_gripper_open)

        right_gripper_joints = [
            self._world.get_connection_by_name("gripper_right_finger_joint"),
        ]

        right_gripper_open = JointState.from_mapping(
            name=PrefixedName("right_gripper_open", prefix=self.name.name),
            mapping=dict(zip(right_gripper_joints, [0.05])),
            state_type=GripperState.OPEN,
        )

        right_gripper_close = JointState.from_mapping(
            name=PrefixedName("right_gripper_close", prefix=self.name.name),
            mapping=dict(zip(right_gripper_joints, [0.75])),
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
            mapping=dict(zip(torso_joint, [0.35])),
            state_type=TorsoState.HIGH,
        )

        self.torso.add_joint_state(torso_low)
        self.torso.add_joint_state(torso_mid)
        self.torso.add_joint_state(torso_high)


@dataclass(eq=False)
class TiagoMujoco(AbstractRobot, SpecifiesLeftRightArm):
    """
    Class that describes the Take It And Go Robot (TIAGo). This version is based on the MuJoCo model, which contains
    less bodies and connections than the URDF version, including missing some crucial links like the camera etc.
    """

    @classmethod
    def _init_empty_robot(cls, world: World) -> Self:
        return cls(
            name=PrefixedName("tiago", prefix=world.name),
            root=world.get_body_by_name("base_link"),
            _world=world,
        )

    def _setup_semantic_annotations(self):
        # Create left arm
        left_gripper_thumb = Finger(
            name=PrefixedName("left_gripper_thumb", prefix=self.name.name),
            root=self._world.get_body_by_name("arm_left_7_link"),
            tip=self._world.get_body_by_name("gripper_left_left_finger_link"),
            _world=self._world,
        )

        left_gripper_finger = Finger(
            name=PrefixedName("left_gripper_finger", prefix=self.name.name),
            root=self._world.get_body_by_name("arm_left_7_link"),
            tip=self._world.get_body_by_name("gripper_left_right_finger_link"),
            _world=self._world,
        )

        left_gripper = ParallelGripper(
            name=PrefixedName("left_gripper", prefix=self.name.name),
            root=self._world.get_body_by_name("arm_left_7_link"),
            tool_frame=self._world.get_body_by_name("arm_left_7_link"),
            front_facing_orientation=Quaternion(0, 0, 0, 1),
            front_facing_axis=Vector3(1, 0, 0),
            thumb=left_gripper_thumb,
            finger=left_gripper_finger,
            _world=self._world,
        )
        left_arm = Arm(
            name=PrefixedName("left_arm", prefix=self.name.name),
            root=self._world.get_body_by_name("torso_lift_link"),
            tip=self._world.get_body_by_name("arm_left_7_link"),
            manipulator=left_gripper,
            _world=self._world,
        )

        self.add_arm(left_arm)

        # Create right arm
        right_gripper_thumb = Finger(
            name=PrefixedName("right_gripper_thumb", prefix=self.name.name),
            root=self._world.get_body_by_name("arm_right_7_link"),
            tip=self._world.get_body_by_name("gripper_right_left_finger_link"),
            _world=self._world,
        )
        right_gripper_finger = Finger(
            name=PrefixedName("right_gripper_finger", prefix=self.name.name),
            root=self._world.get_body_by_name("arm_right_7_link"),
            tip=self._world.get_body_by_name("gripper_right_right_finger_link"),
            _world=self._world,
        )
        right_gripper = ParallelGripper(
            name=PrefixedName("right_gripper", prefix=self.name.name),
            root=self._world.get_body_by_name("arm_right_7_link"),
            tool_frame=self._world.get_body_by_name("arm_right_7_link"),
            front_facing_orientation=Quaternion(0, 0, 0, 1),
            front_facing_axis=Vector3(0, 0, 1),
            thumb=right_gripper_thumb,
            finger=right_gripper_finger,
            _world=self._world,
        )
        right_arm = Arm(
            name=PrefixedName("right_arm", prefix=self.name.name),
            root=self._world.get_body_by_name("torso_lift_link"),
            tip=self._world.get_body_by_name("arm_right_7_link"),
            manipulator=right_gripper,
            _world=self._world,
        )

        self.add_arm(right_arm)

        # Create torso
        torso = Torso(
            name=PrefixedName("torso", prefix=self.name.name),
            root=self._world.get_body_by_name("base_link"),
            tip=self._world.get_body_by_name("torso_lift_link"),
            _world=self._world,
        )
        self.add_torso(torso)
        base = Base(
            name=PrefixedName("base", prefix=self.name.name),
            root=self._world.get_body_by_name("base_link"),
            tip=self._world.get_body_by_name("base_link"),
            _world=self._world,
        )

        self.add_base(base)

    def _setup_collision_rules(self):
        pass

    def _setup_velocity_limits(self):
        pass

    def _setup_hardware_interfaces(self):
        pass

    def _setup_joint_states(self):
        # Create states
        left_arm_park = JointState.from_mapping(
            name=PrefixedName("left_arm_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    [
                        c
                        for c in self.left_arm.connections
                        if type(c) != FixedConnection
                    ],
                    [0.27, -1.07, 1.5, 1.96, -2.0, 1.2, 0.5],
                )
            ),
            state_type=StaticJointState.PARK,
        )

        self.left_arm.add_joint_state(left_arm_park)

        right_arm_park = JointState.from_mapping(
            name=PrefixedName("right_arm_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    [
                        c
                        for c in self.right_arm.connections
                        if type(c) != FixedConnection
                    ],
                    [0.27, -1.07, 1.5, 1.96, -2.0, 1.2, 0.5],
                )
            ),
            state_type=StaticJointState.PARK,
        )

        self.right_arm.add_joint_state(right_arm_park)

        left_gripper_joints = [
            self._world.get_connection_by_name("gripper_left_left_finger_joint"),
            self._world.get_connection_by_name("gripper_left_right_finger_joint"),
        ]

        left_gripper_open = JointState.from_mapping(
            name=PrefixedName("left_gripper_open", prefix=self.name.name),
            mapping=dict(zip(left_gripper_joints, [0.044, 0.044])),
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
            self._world.get_connection_by_name("gripper_right_left_finger_joint"),
            self._world.get_connection_by_name("gripper_right_right_finger_joint"),
        ]

        right_gripper_open = JointState.from_mapping(
            name=PrefixedName("right_gripper_open", prefix=self.name.name),
            mapping=dict(zip(right_gripper_joints, [0.044, 0.044])),
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
            mapping=dict(zip(torso_joint, [0.35])),
            state_type=TorsoState.HIGH,
        )

        self.torso.add_joint_state(torso_low)
        self.torso.add_joint_state(torso_mid)
        self.torso.add_joint_state(torso_high)
