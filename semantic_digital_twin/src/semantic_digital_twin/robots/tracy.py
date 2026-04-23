from __future__ import annotations

import os
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Self

from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidExternalCollisions,
    AvoidSelfCollisions,
    SelfCollisionMatrixRule,
)
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import (
    HasCameras,
    HasLeftRightArm,
    HasNeck,
    HasParallelGripper,
)
from semantic_digital_twin.robots.robot_parts import (
    AbstractRobot,
    Arm,
    Camera,
    FieldOfView,
    Finger,
    Neck,
    ParallelGripper,
)
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@dataclass(eq=False)
class TracyFinger(Finger, ABC):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass


@dataclass(eq=False)
class TracyLeftGripperLeftFinger(TracyFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(
                robot_root, "left_robotiq_85_left_knuckle_link"
            ),
            tip=world.get_body_in_branch_by_name(
                robot_root, "left_robotiq_85_left_finger_tip_link"
            ),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class TracyLeftGripperRightFinger(TracyFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(
                robot_root, "left_robotiq_85_right_knuckle_link"
            ),
            tip=world.get_body_in_branch_by_name(
                robot_root, "left_robotiq_85_right_finger_tip_link"
            ),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class TracyRightGripperLeftFinger(TracyFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(
                robot_root, "right_robotiq_85_left_knuckle_link"
            ),
            tip=world.get_body_in_branch_by_name(
                robot_root, "right_robotiq_85_left_finger_tip_link"
            ),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class TracyRightGripperRightFinger(TracyFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(
                robot_root, "right_robotiq_85_right_knuckle_link"
            ),
            tip=world.get_body_in_branch_by_name(
                robot_root, "right_robotiq_85_right_finger_tip_link"
            ),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class TracyGripper(ParallelGripper, ABC):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        gripper_joints = self.active_connections

        gripper_open = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_open", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
            state_type=GripperState.OPEN,
        )

        gripper_close = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_close", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.8, 0.8, 0.8, 0.8, 0.8, 0.8])),
            state_type=GripperState.CLOSE,
        )

        self.add_joint_state(gripper_open)
        self.add_joint_state(gripper_close)


@dataclass(eq=False)
class TracyLeftGripper(TracyGripper):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        gripper = cls(
            root=world.get_body_in_branch_by_name(
                robot_root, "left_robotiq_85_base_link"
            ),
            tool_frame=world.get_body_in_branch_by_name(
                robot_root, "l_gripper_tool_frame"
            ),
            front_facing_orientation=Quaternion(0.5, 0.5, 0.5, 0.5),
        )
        world.add_semantic_annotation(gripper)
        return gripper

    def setup_finger_semantic_annotations(self):
        self.add_thumb(
            TracyLeftGripperLeftFinger.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_finger(
            TracyLeftGripperRightFinger.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )


@dataclass(eq=False)
class TracyRightGripper(TracyGripper):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        gripper = cls(
            root=world.get_body_in_branch_by_name(
                robot_root, "right_robotiq_85_base_link"
            ),
            tool_frame=world.get_body_in_branch_by_name(
                robot_root, "r_gripper_tool_frame"
            ),
            front_facing_orientation=Quaternion(0.5, 0.5, 0.5, 0.5),
        )
        world.add_semantic_annotation(gripper)
        return gripper

    def setup_finger_semantic_annotations(self):
        self.add_thumb(
            TracyRightGripperLeftFinger.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_finger(
            TracyRightGripperRightFinger.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )


@dataclass(eq=False)
class TracyLeftArm(Arm, HasParallelGripper):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        arm_park = JointState.from_mapping(
            name=PrefixedName("left_arm_park", prefix=self.name.name),
            mapping=dict(zip(self.active_connections, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
            state_type=StaticJointState.PARK,
        )
        self.add_joint_state(arm_park)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        arm = cls(
            root=world.get_body_in_branch_by_name(robot_root, "table"),
            tip=world.get_body_in_branch_by_name(robot_root, "left_wrist_3_link"),
        )
        world.add_semantic_annotation(arm)
        return arm

    def setup_end_effector_semantic_annotation(self):
        gripper = (
            TracyLeftGripper.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_end_effector(gripper)
        gripper.setup_finger_semantic_annotations()


@dataclass(eq=False)
class TracyRightArm(Arm, HasParallelGripper):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        arm_park = JointState.from_mapping(
            name=PrefixedName("right_arm_park", prefix=self.name.name),
            mapping=dict(zip(self.active_connections, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
            state_type=StaticJointState.PARK,
        )
        self.add_joint_state(arm_park)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        arm = cls(
            root=world.get_body_in_branch_by_name(robot_root, "table"),
            tip=world.get_body_in_branch_by_name(robot_root, "right_wrist_3_link"),
        )
        world.add_semantic_annotation(arm)
        return arm

    def setup_end_effector_semantic_annotation(self):
        gripper = (
            TracyRightGripper.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_end_effector(gripper)
        gripper.setup_finger_semantic_annotations()


@dataclass(eq=False)
class TracyCamera(Camera):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        camera = cls(
            root=world.get_body_in_branch_by_name(robot_root, "camera_link"),
            forward_facing_axis=Vector3.Z(),
            field_of_view=FieldOfView(horizontal_angle=1.047, vertical_angle=0.785),
            minimal_height=0.8,
            maximal_height=1.7,
        )
        world.add_semantic_annotation(camera)
        return camera


@dataclass(eq=False)
class Tracy(AbstractRobot, HasLeftRightArm, HasCameras):

    def setup_sensor_semantic_annotations(self):
        camera = TracyCamera.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_sensor(camera)

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "table"

    def setup_arm_semantic_annotations(self):
        left_arm = TracyLeftArm.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_arm(left_arm)
        left_arm.setup_end_effector_semantic_annotation()

        right_arm = TracyRightArm.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_arm(right_arm)
        right_arm.setup_end_effector_semantic_annotation()

    def setup_robot_part_semantic_annotations(self):
        self.setup_arm_semantic_annotations()
        self.setup_sensor_semantic_annotations()

    def _setup_collision_rules(self):
        srdf_path = os.path.join(
            Path(files("semantic_digital_twin")).parent.parent,
            "resources",
            "collision_configs",
            "tracy.srdf",
        )
        self._world.collision_manager.add_ignore_collision_rule(
            SelfCollisionMatrixRule.from_collision_srdf(srdf_path, self._world)
        )

        self._world.collision_manager.extend_default_rules(
            [
                AvoidExternalCollisions(
                    buffer_zone_distance=0.05, violated_distance=0.0, robot=self
                ),
                AvoidSelfCollisions(
                    buffer_zone_distance=0.03,
                    violated_distance=0.0,
                    robot=self,
                ),
            ]
        )

    def _setup_velocity_limits(self):
        vel_limits = defaultdict(lambda: 1.0)
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)
