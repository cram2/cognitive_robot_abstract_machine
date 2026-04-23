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
    TorsoState,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import (
    HasCameras,
    HasLeftRightArm,
    HasMobileBase,
    HasNeck,
    HasParallelGripper,
    HasTorso,
)
from semantic_digital_twin.robots.robot_parts import (
    AbstractRobot,
    Arm,
    Camera,
    FieldOfView,
    Finger,
    MobileBase,
    Neck,
    ParallelGripper,
    Torso,
)
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@dataclass(eq=False)
class TiagoFinger(Finger, ABC):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass


@dataclass(eq=False)
class TiagoLeftThumb(TiagoFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "gripper_left_base_link"),
            tip=world.get_body_in_branch_by_name(
                robot_root, "gripper_left_left_inner_finger_pad"
            ),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class TiagoLeftIndexFinger(TiagoFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "gripper_left_base_link"),
            tip=world.get_body_in_branch_by_name(
                robot_root, "gripper_left_right_inner_finger_pad"
            ),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class TiagoRightThumb(TiagoFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(
                robot_root, "gripper_right_base_link"
            ),
            tip=world.get_body_in_branch_by_name(
                robot_root, "gripper_right_left_inner_finger_pad"
            ),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class TiagoRightIndexFinger(TiagoFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(
                robot_root, "gripper_right_base_link"
            ),
            tip=world.get_body_in_branch_by_name(
                robot_root, "gripper_right_right_inner_finger_pad"
            ),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class TiagoGripper(ParallelGripper, ABC):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        gripper_joints = self.active_connections

        gripper_open = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_open", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.045, 0.045])),
            state_type=GripperState.OPEN,
        )

        gripper_close = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_close", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.0, 0.0])),
            state_type=GripperState.CLOSE,
        )

        self.add_joint_state(gripper_open)
        self.add_joint_state(gripper_close)


@dataclass(eq=False)
class TiagoLeftHand(TiagoGripper):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        gripper = cls(
            root=world.get_body_in_branch_by_name(robot_root, "gripper_left_base_link"),
            tool_frame=world.get_body_in_branch_by_name(
                robot_root, "gripper_left_grasping_frame"
            ),
            front_facing_orientation=Quaternion(0, 0, 0, 1),
        )
        world.add_semantic_annotation(gripper)
        return gripper

    def setup_finger_semantic_annotations(self):
        self.add_thumb(
            TiagoLeftThumb.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_finger(
            TiagoLeftIndexFinger.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )


@dataclass(eq=False)
class TiagoRightHand(TiagoGripper):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        gripper = cls(
            root=world.get_body_in_branch_by_name(
                robot_root, "gripper_right_base_link"
            ),
            tool_frame=world.get_body_in_branch_by_name(
                robot_root, "gripper_right_grasping_frame"
            ),
            front_facing_orientation=Quaternion(0, 0, 0, 1),
        )
        world.add_semantic_annotation(gripper)
        return gripper

    def setup_finger_semantic_annotations(self):
        self.add_thumb(
            TiagoRightThumb.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_finger(
            TiagoRightIndexFinger.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )


@dataclass(eq=False)
class TiagoLeftArm(Arm, HasParallelGripper):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        arm_park = JointState.from_mapping(
            name=PrefixedName("left_arm_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    self.active_connections,
                    [1.57, 0.33, 1.57, 1.57, 0.0, 1.1, 0.0],
                )
            ),
            state_type=StaticJointState.PARK,
        )
        self.add_joint_state(arm_park)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        arm = cls(
            root=world.get_body_in_branch_by_name(robot_root, "torso_lift_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "arm_left_tool_link"),
        )
        world.add_semantic_annotation(arm)
        return arm

    def setup_end_effector_semantic_annotation(self):
        gripper = TiagoLeftHand.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_end_effector(gripper)
        gripper.setup_finger_semantic_annotations()


@dataclass(eq=False)
class TiagoRightArm(Arm, HasParallelGripper):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        arm_park = JointState.from_mapping(
            name=PrefixedName("right_arm_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    self.active_connections,
                    [1.57, 0.33, 1.57, 1.57, 0.0, 1.1, 0.0],
                )
            ),
            state_type=StaticJointState.PARK,
        )
        self.add_joint_state(arm_park)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        arm = cls(
            root=world.get_body_in_branch_by_name(robot_root, "torso_lift_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "arm_right_tool_link"),
        )
        world.add_semantic_annotation(arm)
        return arm

    def setup_end_effector_semantic_annotation(self):
        gripper = TiagoRightHand.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_end_effector(gripper)
        gripper.setup_finger_semantic_annotations()


@dataclass(eq=False)
class TiagoCamera(Camera):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        camera = cls(
            root=world.get_body_in_branch_by_name(
                robot_root, "head_front_camera_optical_frame"
            ),
            forward_facing_axis=Vector3.Z(),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=1.0665,
            maximal_height=1.4165,
            default_camera=True,
        )
        world.add_semantic_annotation(camera)

        return camera

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass


@dataclass(eq=False)
class TiagoNeck(Neck, HasCameras):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        pass

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        neck = cls(
            root=world.get_body_in_branch_by_name(robot_root, "torso_lift_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "head_2_link"),
        )
        world.add_semantic_annotation(neck)
        return neck

    def setup_sensor_semantic_annotations(self):
        camera = TiagoCamera.setup_default_configuration_in_world_below_robot_root(
            robot_root=self.root
        )
        self.add_sensor(camera)


@dataclass(eq=False)
class TiagoTorso(Torso, HasLeftRightArm, HasNeck):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        torso_joint = self.active_connections
        torso_low = JointState.from_mapping(
            name=PrefixedName("torso_low", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.0])),
            state_type=TorsoState.LOW,
        )

        torso_mid = JointState.from_mapping(
            name=PrefixedName("torso_mid", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.175])),
            state_type=TorsoState.MID,
        )

        torso_high = JointState.from_mapping(
            name=PrefixedName("torso_high", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.35])),
            state_type=TorsoState.HIGH,
        )

        self.add_joint_state(torso_low)
        self.add_joint_state(torso_mid)
        self.add_joint_state(torso_high)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        torso = cls(
            root=world.get_body_in_branch_by_name(robot_root, "torso_fixed_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "torso_lift_link"),
        )
        world.add_semantic_annotation(torso)
        return torso

    def setup_arm_semantic_annotations(self):
        left_arm = TiagoLeftArm.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_arm(left_arm)
        left_arm.setup_end_effector_semantic_annotation()

        right_arm = TiagoRightArm.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_arm(right_arm)
        right_arm.setup_end_effector_semantic_annotation()

    def setup_neck_semantic_annotation(self):
        neck = TiagoNeck.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        neck.setup_sensor_semantic_annotations()
        self.add_neck(neck)


@dataclass(eq=False)
class TiagoMobileBase(MobileBase, HasTorso):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        base = cls(
            root=world.get_body_in_branch_by_name(robot_root, "base_link"),
            forward_axis=Vector3.X(),
            full_body_controlled=False,
        )
        world.add_semantic_annotation(base)
        return base

    def setup_torso_semantic_annotation(self):
        torso = TiagoTorso.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        torso.setup_arm_semantic_annotations()
        torso.setup_neck_semantic_annotation()
        self.add_torso(torso)


@dataclass(eq=False)
class Tiago(AbstractRobot, HasMobileBase):

    @classmethod
    def get_ros_file_path(cls) -> str:
        return "package://iai_tiago_description/urdf/tiago_from_our_robot.urdf"

    def setup_mobile_base_semantic_annotation(self):
        mobile_base = (
            TiagoMobileBase.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_mobile_base(mobile_base)
        mobile_base.setup_torso_semantic_annotation()

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "base_footprint"

    def setup_robot_part_semantic_annotations(self):
        self.setup_mobile_base_semantic_annotation()

    def _setup_collision_rules(self):
        srdf_path = os.path.join(
            Path(files("semantic_digital_twin")).parent.parent,
            "resources",
            "collision_configs",
            "tiago.srdf",
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


@dataclass(eq=False)
class TiagoMujocoFinger(Finger, ABC):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass


@dataclass(eq=False)
class TiagoMujocoLeftThumb(TiagoMujocoFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "arm_left_7_link"),
            tip=world.get_body_in_branch_by_name(
                robot_root, "gripper_left_left_finger_link"
            ),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class TiagoMujocoLeftIndexFinger(TiagoMujocoFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "arm_left_7_link"),
            tip=world.get_body_in_branch_by_name(
                robot_root, "gripper_left_right_finger_link"
            ),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class TiagoMujocoRightThumb(TiagoMujocoFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "arm_right_7_link"),
            tip=world.get_body_in_branch_by_name(
                robot_root, "gripper_right_left_finger_link"
            ),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class TiagoMujocoRightIndexFinger(TiagoMujocoFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "arm_right_7_link"),
            tip=world.get_body_in_branch_by_name(
                robot_root, "gripper_right_right_finger_link"
            ),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class TiagoMujocoGripper(ParallelGripper, ABC):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        gripper_joints = self.active_connections
        gripper_open = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_open", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.044, 0.044])),
            state_type=GripperState.OPEN,
        )
        gripper_close = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_close", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.0, 0.0])),
            state_type=GripperState.CLOSE,
        )
        self.add_joint_state(gripper_close)
        self.add_joint_state(gripper_open)


@dataclass(eq=False)
class TiagoMujocoLeftGripper(TiagoMujocoGripper):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        gripper = cls(
            root=world.get_body_in_branch_by_name(robot_root, "arm_left_7_link"),
            tool_frame=world.get_body_in_branch_by_name(robot_root, "arm_left_7_link"),
            front_facing_orientation=Quaternion(0, 0, 0, 1),
        )
        world.add_semantic_annotation(gripper)
        return gripper

    def setup_finger_semantic_annotations(self):
        thumb = (
            TiagoMujocoLeftThumb.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_thumb(thumb)
        finger = TiagoMujocoLeftIndexFinger.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_finger(finger)


@dataclass(eq=False)
class TiagoMujocoRightGripper(TiagoMujocoGripper):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        gripper = cls(
            root=world.get_body_in_branch_by_name(robot_root, "arm_right_7_link"),
            tool_frame=world.get_body_in_branch_by_name(robot_root, "arm_right_7_link"),
            front_facing_orientation=Quaternion(0, 0, 0, 1),
        )
        world.add_semantic_annotation(gripper)
        return gripper

    def setup_finger_semantic_annotations(self):
        thumb = (
            TiagoMujocoRightThumb.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_thumb(thumb)
        finger = TiagoMujocoRightIndexFinger.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_finger(finger)


@dataclass(eq=False)
class TiagoMujocoLeftArm(Arm, HasParallelGripper):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        arm_park = JointState.from_mapping(
            name=PrefixedName("left_arm_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    self.active_connections,
                    [0.27, -1.07, 1.5, 1.96, -2.0, 1.2, 0.5],
                )
            ),
            state_type=StaticJointState.PARK,
        )
        self.add_joint_state(arm_park)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        arm = cls(
            root=world.get_body_in_branch_by_name(robot_root, "torso_lift_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "arm_left_7_link"),
        )
        world.add_semantic_annotation(arm)
        return arm

    def setup_end_effector_semantic_annotation(self):
        gripper = TiagoMujocoLeftGripper.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_end_effector(gripper)
        gripper.setup_finger_semantic_annotations()


@dataclass(eq=False)
class TiagoMujocoRightArm(Arm, HasParallelGripper):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        arm_park = JointState.from_mapping(
            name=PrefixedName("right_arm_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    self.active_connections,
                    [0.27, -1.07, 1.5, 1.96, -2.0, 1.2, 0.5],
                )
            ),
            state_type=StaticJointState.PARK,
        )
        self.add_joint_state(arm_park)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        arm = cls(
            root=world.get_body_in_branch_by_name(robot_root, "torso_lift_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "arm_right_7_link"),
        )
        world.add_semantic_annotation(arm)
        return arm

    def setup_end_effector_semantic_annotation(self):
        gripper = TiagoMujocoRightGripper.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_end_effector(gripper)
        gripper.setup_finger_semantic_annotations()


@dataclass(eq=False)
class TiagoMujocoNeck(Neck, HasCameras):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        pass

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        neck = cls(
            root=world.get_body_in_branch_by_name(robot_root, "torso_lift_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "head_2_link"),
        )
        world.add_semantic_annotation(neck)
        return neck

    def setup_sensor_semantic_annotations(self):
        camera = TiagoCamera.setup_default_configuration_in_world_below_robot_root(
            robot_root=self.root
        )
        self.add_sensor(camera)


@dataclass(eq=False)
class TiagoMujocoTorso(Torso, HasLeftRightArm, HasNeck):

    def setup_neck_semantic_annotation(self):
        neck = TiagoMujocoNeck.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        neck.setup_sensor_semantic_annotations()
        self.add_neck(neck)

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        torso_joint = self.active_connections
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
        self.add_joint_state(torso_low)
        self.add_joint_state(torso_mid)
        self.add_joint_state(torso_high)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        torso = cls(
            root=world.get_body_in_branch_by_name(robot_root, "base_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "torso_lift_link"),
        )
        world.add_semantic_annotation(torso)
        return torso

    def setup_arm_semantic_annotations(self):
        left_arm = (
            TiagoMujocoLeftArm.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_arm(left_arm)
        left_arm.setup_end_effector_semantic_annotation()

        right_arm = (
            TiagoMujocoRightArm.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_arm(right_arm)
        right_arm.setup_end_effector_semantic_annotation()


@dataclass(eq=False)
class TiagoMujocoMobileBase(MobileBase, HasTorso):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        base = cls(
            root=world.get_body_in_branch_by_name(robot_root, "base_link"),
            forward_axis=Vector3.X(),
            full_body_controlled=False,
        )
        world.add_semantic_annotation(base)
        return base

    def setup_torso_semantic_annotation(self):
        torso = TiagoMujocoTorso.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        torso.setup_arm_semantic_annotations()
        torso.setup_neck_semantic_annotation()
        self.add_torso(torso)


@dataclass(eq=False)
class TiagoMujoco(AbstractRobot, HasMobileBase):
    """
    Class that describes the Take It And Go Robot (TIAGo). This version is based on the MuJoCo model, which contains
    less bodies and connections than the URDF version, including missing some crucial links like the camera etc.
    """

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "base_link"

    def setup_mobile_base_semantic_annotation(self):
        mobile_base = (
            TiagoMujocoMobileBase.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        mobile_base.setup_torso_semantic_annotation()
        self.add_mobile_base(mobile_base)

    def setup_robot_part_semantic_annotations(self):
        self.setup_mobile_base_semantic_annotation()

    def _setup_collision_rules(self):
        pass

    def _setup_velocity_limits(self):
        pass
