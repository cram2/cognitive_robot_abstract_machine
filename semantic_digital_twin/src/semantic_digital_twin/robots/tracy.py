from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import field, dataclass
from typing import Self

from pkg_resources import resource_filename

from ..collision_checking.collision_rules import (
    SelfCollisionMatrixRule,
    AvoidAllCollisions,
    AvoidExternalCollisions,
    AvoidSelfCollisions,
)
from ..datastructures.definitions import StaticJointState, GripperState
from ..datastructures.joint_state import JointState
from ..datastructures.prefixed_name import PrefixedName
from ..robots.abstract_robot import (
    Finger,
    ParallelGripper,
    Arm,
    Camera,
    FieldOfView,
    Neck,
    AbstractRobot,
)
from .robot_mixins import HasNeck, SpecifiesLeftRightArm
from ..spatial_types import Quaternion, Vector3
from ..world import World
from ..world_description.connections import FixedConnection, ActiveConnection


@dataclass(eq=False)
class Tracy(AbstractRobot, SpecifiesLeftRightArm, HasNeck):
    """
    Represents two UR10e Arms on a table, with a pole between them holding a small camera.
     Example can be found at: https://vib.ai.uni-bremen.de/page/comingsoon/the-tracebot-laboratory/
    """

    def __hash__(self):
        return hash(
            tuple(
                [self.__class__]
                + sorted([kse.name for kse in self.kinematic_structure_entities])
            )
        )

    @classmethod
    def _init_empty_robot(cls, world: World) -> Self:
        return cls(
            name=PrefixedName(name="tracy", prefix=world.name),
            root=world.get_body_by_name("table"),
            _world=world,
        )

    def _setup_semantic_annotations(self):
        # Create left arm
        left_gripper_thumb = Finger(
            name=PrefixedName("left_gripper_thumb", prefix=self.name.name),
            root=self._world.get_body_by_name("left_robotiq_85_left_knuckle_link"),
            tip=self._world.get_body_by_name("left_robotiq_85_left_finger_tip_link"),
            _world=self._world,
        )

        left_gripper_finger = Finger(
            name=PrefixedName("left_gripper_finger", prefix=self.name.name),
            root=self._world.get_body_by_name("left_robotiq_85_right_knuckle_link"),
            tip=self._world.get_body_by_name("left_robotiq_85_right_finger_tip_link"),
            _world=self._world,
        )

        left_gripper = ParallelGripper(
            name=PrefixedName("left_gripper", prefix=self.name.name),
            root=self._world.get_body_by_name("left_robotiq_85_base_link"),
            tool_frame=self._world.get_body_by_name("l_gripper_tool_frame"),
            front_facing_orientation=Quaternion(0.5, 0.5, 0.5, 0.5),
            front_facing_axis=Vector3(0, 0, 1),
            thumb=left_gripper_thumb,
            finger=left_gripper_finger,
            _world=self._world,
        )
        left_arm = Arm(
            name=PrefixedName("left_arm", prefix=self.name.name),
            root=self._world.get_body_by_name("table"),
            tip=self._world.get_body_by_name("left_wrist_3_link"),
            manipulator=left_gripper,
            _world=self._world,
        )

        self.add_arm(left_arm)

        right_gripper_thumb = Finger(
            name=PrefixedName("right_gripper_thumb", prefix=self.name.name),
            root=self._world.get_body_by_name("right_robotiq_85_left_knuckle_link"),
            tip=self._world.get_body_by_name("right_robotiq_85_left_finger_tip_link"),
            _world=self._world,
        )
        right_gripper_finger = Finger(
            name=PrefixedName("right_gripper_finger", prefix=self.name.name),
            root=self._world.get_body_by_name("right_robotiq_85_right_knuckle_link"),
            tip=self._world.get_body_by_name("right_robotiq_85_right_finger_tip_link"),
            _world=self._world,
        )
        right_gripper = ParallelGripper(
            name=PrefixedName("right_gripper", prefix=self.name.name),
            root=self._world.get_body_by_name("right_robotiq_85_base_link"),
            tool_frame=self._world.get_body_by_name("r_gripper_tool_frame"),
            front_facing_orientation=Quaternion(0.5, 0.5, 0.5, 0.5),
            front_facing_axis=Vector3(0, 0, 1),
            thumb=right_gripper_thumb,
            finger=right_gripper_finger,
            _world=self._world,
        )
        right_arm = Arm(
            name=PrefixedName("right_arm", prefix=self.name.name),
            root=self._world.get_body_by_name("table"),
            tip=self._world.get_body_by_name("right_wrist_3_link"),
            manipulator=right_gripper,
            _world=self._world,
        )
        self.add_arm(right_arm)

        camera = Camera(
            name=PrefixedName("camera", prefix=self.name.name),
            root=self._world.get_body_by_name("camera_link"),
            forward_facing_axis=Vector3(0, 0, 1),
            field_of_view=FieldOfView(horizontal_angle=1.047, vertical_angle=0.785),
            minimal_height=0.8,
            maximal_height=1.7,
            _world=self._world,
        )

        # Probably should be classified as "Neck", as that implies that i can move.
        neck = Neck(
            name=PrefixedName("neck", prefix=self.name.name),
            sensors=[camera],
            root=self._world.get_body_by_name("camera_pole"),
            tip=self._world.get_body_by_name("camera_link"),
            _world=self._world,
        )

        self.add_kinematic_chain(neck)

    def _setup_collision_rules(self):
        srdf_path = os.path.join(
            resource_filename("semantic_digital_twin", "../../"),
            "resources",
            "collision_configs",
            "tracy.srdf",
        )
        self._world.collision_manager.ignore_collision_rules.append(
            SelfCollisionMatrixRule.from_collision_srdf(srdf_path, self._world)
        )

        self._world.collision_manager.add_default_rule(
            AvoidExternalCollisions(
                buffer_zone_distance=0.05, violated_distance=0.0, robot=self
            )
        )
        self._world.collision_manager.add_default_rule(
            AvoidSelfCollisions(
                buffer_zone_distance=0.03,
                violated_distance=0.0,
                robot=self,
            )
        )

    def _setup_velocity_limits(self):
        vel_limits = defaultdict(lambda: 0.2)
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)

    def _setup_hardware_interfaces(self):
        controlled_joints = [
            "left_shoulder_pan_joint",
            "left_shoulder_lift_joint",
            "left_elbow_joint",
            "left_wrist_1_joint",
            "left_wrist_2_joint",
            "left_wrist_3_joint",
            "right_shoulder_pan_joint",
            "right_shoulder_lift_joint",
            "right_elbow_joint",
            "right_wrist_1_joint",
            "right_wrist_2_joint",
            "right_wrist_3_joint",
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
                    [2.62, -1.035, 1.13, -0.966, -0.88, 2.07],
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
                    [3.72, -2.07, -1.17, 4.0, 0.82, 0.75],
                )
            ),
            state_type=StaticJointState.PARK,
        )

        self.right_arm.add_joint_state(right_arm_park)

        left_gripper_joints = [
            self._world.get_connection_by_name("left_robotiq_85_left_knuckle_joint"),
            self._world.get_connection_by_name("left_robotiq_85_right_knuckle_joint"),
        ]

        left_gripper_open = JointState.from_mapping(
            name=PrefixedName("left_gripper_open", prefix=self.name.name),
            mapping=dict(zip(left_gripper_joints, [0.0, 0.0])),
            state_type=GripperState.OPEN,
        )

        left_gripper_close = JointState.from_mapping(
            name=PrefixedName("left_gripper_close", prefix=self.name.name),
            mapping=dict(
                zip(
                    left_gripper_joints,
                    [
                        0.8,
                        -0.8,
                    ],
                )
            ),
            state_type=GripperState.CLOSE,
        )

        self.left_arm.manipulator.add_joint_state(left_gripper_close)
        self.left_arm.manipulator.add_joint_state(left_gripper_open)

        right_gripper_joints = [
            self._world.get_connection_by_name("right_robotiq_85_left_knuckle_joint"),
            self._world.get_connection_by_name("right_robotiq_85_right_knuckle_joint"),
        ]

        right_gripper_open = JointState.from_mapping(
            name=PrefixedName("right_gripper_open", prefix=self.name.name),
            mapping=dict(zip(right_gripper_joints, [0.0, 0.0])),
            state_type=GripperState.OPEN,
        )

        right_gripper_close = JointState.from_mapping(
            name=PrefixedName("right_gripper_close", prefix=self.name.name),
            mapping=dict(zip(right_gripper_joints, [0.8, -0.8])),
            state_type=GripperState.CLOSE,
        )

        self.right_arm.manipulator.add_joint_state(right_gripper_close)
        self.right_arm.manipulator.add_joint_state(right_gripper_open)
