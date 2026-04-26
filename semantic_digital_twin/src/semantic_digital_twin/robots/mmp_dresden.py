from __future__ import annotations

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Self

from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import (
    HasCameras,
    HasOneArm,
    HasParallelGripper,
    HasTorso,
    HasMobileBase,
)
from semantic_digital_twin.robots.robot_parts import (
    AbstractRobot,
    Arm,
    Camera,
    FieldOfView,
    Finger,
    ParallelGripper,
    Torso,
    MobileBase,
)
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@dataclass(eq=False)
class MMPDresdenFinger(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass


@dataclass(eq=False)
class MMPDresdenThumb(MMPDresdenFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(
                robot_root, "arm_0_gripper_robotiq_85_left_knuckle_link"
            ),
            tip=world.get_body_in_branch_by_name(
                robot_root, "arm_0_gripper_robotiq_85_left_finger_tip_link"
            ),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class MMPDresdenIndexFinger(MMPDresdenFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(
                robot_root, "arm_0_gripper_robotiq_85_right_knuckle_link"
            ),
            tip=world.get_body_in_branch_by_name(
                robot_root, "arm_0_gripper_robotiq_85_right_finger_tip_link"
            ),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class MMPDresdenGripper(ParallelGripper):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        gripper_joints = self.active_connections

        gripper_open = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_open", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.0])),
            state_type=GripperState.OPEN,
        )

        gripper_close = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_close", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.8])),
            state_type=GripperState.CLOSE,
        )

        self.add_joint_state(gripper_open)
        self.add_joint_state(gripper_close)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        gripper = cls(
            root=world.get_body_in_branch_by_name(robot_root, "arm_0_flange"),
            tool_frame=world.get_body_in_branch_by_name(robot_root, "arm_0_tool_frame"),
            front_facing_orientation=Quaternion(0.5, 0.5, 0.5, 0.5),
        )
        world.add_semantic_annotation(gripper)
        return gripper

    def setup_finger_semantic_annotations(self):
        self.add_thumb(
            MMPDresdenThumb.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_finger(
            MMPDresdenIndexFinger.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )


@dataclass(eq=False)
class MMPDresdenCamera(Camera):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        camera = cls(
            root=world.get_body_in_branch_by_name(
                robot_root, "pan_and_tilt_camera_link"
            ),
            forward_facing_axis=Vector3.Z(),
            field_of_view=FieldOfView(horizontal_angle=1.047, vertical_angle=0.785),
            minimal_height=0.8,
            maximal_height=1.7,
            default_camera=True,
        )
        world.add_semantic_annotation(camera)
        return camera

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass


@dataclass(eq=False)
class MMPDresdenArm(Arm, HasParallelGripper):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        arm_park = JointState.from_mapping(
            name=PrefixedName("arm_park", prefix=self.name.name),
            mapping=dict(zip(self.active_connections, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
            state_type=StaticJointState.PARK,
        )
        arm_both = JointState.from_mapping(
            name=PrefixedName("arm_both", prefix=self.name.name),
            mapping=dict(
                zip(
                    self.active_connections,
                    [np.pi / 2, -(np.pi / 2), 0.0, 0.0, 0.0, np.pi / 2],
                )
            ),
            state_type=StaticJointState.PARK,
        )
        self.add_joint_state(arm_park)
        self.add_joint_state(arm_both)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        arm = cls(
            root=world.get_body_in_branch_by_name(robot_root, "hub_holder_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "arm_0_flange"),
        )
        world.add_semantic_annotation(arm)
        return arm

    def setup_end_effector_semantic_annotation(self):
        gripper = (
            MMPDresdenGripper.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_end_effector(gripper)
        gripper.setup_finger_semantic_annotations()


@dataclass(eq=False)
class MMPDresdenTorso(Torso, HasOneArm, HasCameras):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        pass

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        torso = cls(
            root=world.get_body_in_branch_by_name(robot_root, "base_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "hub_holder_link"),
        )
        world.add_semantic_annotation(torso)
        return torso

    def setup_arm_semantic_annotations(self):
        arm = MMPDresdenArm.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_arm(arm)
        arm.setup_end_effector_semantic_annotation()

    def setup_sensor_semantic_annotations(self):
        camera = MMPDresdenCamera.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_camera(camera)


@dataclass(eq=False)
class MMPDresdenMobileBase(MobileBase, HasTorso):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        mobile_base = cls(
            root=world.get_body_in_branch_by_name(robot_root, "base_link"),
        )
        world.add_semantic_annotation(mobile_base)
        return mobile_base

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass

    def setup_torso_semantic_annotation(self):
        torso = MMPDresdenTorso.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_torso(torso)
        torso.setup_arm_semantic_annotations()
        torso.setup_sensor_semantic_annotations()


@dataclass(eq=False)
class MMPDresden(AbstractRobot, HasMobileBase):

    @classmethod
    def get_ros_file_path(cls) -> str:
        return "package://iai_smart_mobility/urdf/mmp_dresden.urdf"

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "base_link"

    def setup_mobile_base_semantic_annotation(self):
        mobile_base = (
            MMPDresdenMobileBase.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_mobile_base(mobile_base)
        mobile_base.setup_torso_semantic_annotation()

    def setup_robot_part_semantic_annotations(self):
        self.setup_mobile_base_semantic_annotation()

    def _setup_velocity_limits(self):
        vel_limits = defaultdict(lambda: 1.0)
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)
