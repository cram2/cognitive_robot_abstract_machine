from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Self

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
    HasNeck,
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
    Neck,
    ParallelGripper,
    Torso,
    MobileBase,
)
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@dataclass(eq=False)
class JustinFinger(Finger, ABC):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass


@dataclass(eq=False)
class JustinLeftThumb(JustinFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "left_1thumb_base"),
            tip=world.get_body_in_branch_by_name(robot_root, "left_1thumb4"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class JustinLeftIndexFinger(JustinFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "left_2tip_base"),
            tip=world.get_body_in_branch_by_name(robot_root, "left_2tip4"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class JustinLeftMiddleFinger(JustinFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "left_3middle_base"),
            tip=world.get_body_in_branch_by_name(robot_root, "left_3middle4"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class JustinLeftRingFinger(JustinFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "left_4ring_base"),
            tip=world.get_body_in_branch_by_name(robot_root, "left_4ring4"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class JustinRightThumb(JustinFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "right_1thumb_base"),
            tip=world.get_body_in_branch_by_name(robot_root, "right_1thumb4"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class JustinRightIndexFinger(JustinFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "right_2tip_base"),
            tip=world.get_body_in_branch_by_name(robot_root, "right_2tip4"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class JustinRightMiddleFinger(JustinFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "right_3middle_base"),
            tip=world.get_body_in_branch_by_name(robot_root, "right_3middle4"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class JustinRightRingFinger(JustinFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(robot_root, "right_4ring_base"),
            tip=world.get_body_in_branch_by_name(robot_root, "right_4ring4"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class JustinGripper(ParallelGripper, ABC):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        gripper_joints = self.active_connections

        gripper_open = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_open", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.0] * len(gripper_joints))),
            state_type=GripperState.OPEN,
        )

        gripper_close = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_close", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [1.0] * len(gripper_joints))),
            state_type=GripperState.CLOSE,
        )

        self.add_joint_state(gripper_open)
        self.add_joint_state(gripper_close)


@dataclass(eq=False)
class JustinLeftHand(JustinGripper):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        gripper = cls(
            root=world.get_body_in_branch_by_name(robot_root, "left_arm7"),
            tool_frame=world.get_body_in_branch_by_name(
                robot_root, "l_gripper_tool_frame"
            ),
            front_facing_orientation=Quaternion(0.707, -0.707, 0.707, -0.707),
        )
        world.add_semantic_annotation(gripper)
        return gripper

    def setup_finger_semantic_annotations(self):
        self.add_thumb(
            JustinLeftThumb.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_finger(
            JustinLeftIndexFinger.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_finger(
            JustinLeftMiddleFinger.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_finger(
            JustinLeftRingFinger.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )


@dataclass(eq=False)
class JustinRightHand(JustinGripper):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        gripper = cls(
            root=world.get_body_in_branch_by_name(robot_root, "right_arm7"),
            tool_frame=world.get_body_in_branch_by_name(
                robot_root, "r_gripper_tool_frame"
            ),
            front_facing_orientation=Quaternion(0.707, 0.707, 0.707, 0.707),
        )
        world.add_semantic_annotation(gripper)
        return gripper

    def setup_finger_semantic_annotations(self):
        self.add_thumb(
            JustinRightThumb.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_finger(
            JustinRightIndexFinger.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_finger(
            JustinRightMiddleFinger.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_finger(
            JustinRightRingFinger.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )


@dataclass(eq=False)
class JustinLeftArm(Arm, HasParallelGripper):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        arm_park = JointState.from_mapping(
            name=PrefixedName("left_arm_park", prefix=self.name.name),
            mapping=dict(
                zip(self.active_connections, [0.0] * len(self.active_connections))
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
            root=world.get_body_in_branch_by_name(robot_root, "base_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "left_arm7"),
        )
        world.add_semantic_annotation(arm)
        return arm

    def setup_end_effector_semantic_annotation(self):
        gripper = JustinLeftHand.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_end_effector(gripper)
        gripper.setup_finger_semantic_annotations()


@dataclass(eq=False)
class JustinRightArm(Arm, HasParallelGripper):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        arm_park = JointState.from_mapping(
            name=PrefixedName("right_arm_park", prefix=self.name.name),
            mapping=dict(
                zip(self.active_connections, [0.0] * len(self.active_connections))
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
            root=world.get_body_in_branch_by_name(robot_root, "base_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "right_arm7"),
        )
        world.add_semantic_annotation(arm)
        return arm

    def setup_end_effector_semantic_annotation(self):
        gripper = JustinRightHand.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_end_effector(gripper)
        gripper.setup_finger_semantic_annotations()


@dataclass(eq=False)
class JustinCamera(Camera):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        camera = cls(
            root=world.get_body_in_branch_by_name(robot_root, "head_link2"),
            forward_facing_axis=Vector3.Z(),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=0.75049,
            maximal_height=0.99483,
        )
        world.add_semantic_annotation(camera)
        return camera

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass


@dataclass(eq=False)
class JustinNeck(Neck, HasCameras):

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
            root=world.get_body_in_branch_by_name(robot_root, "torso4"),
            tip=world.get_body_in_branch_by_name(robot_root, "head2"),
        )
        world.add_semantic_annotation(neck)
        return neck

    def setup_sensor_semantic_annotations(self):
        self.add_sensor(
            JustinCamera.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )


@dataclass(eq=False)
class JustinTorso(Torso, HasLeftRightArm, HasNeck):

    def setup_arm_semantic_annotations(self):
        left_arm = JustinLeftArm.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_arm(left_arm)
        left_arm.setup_end_effector_semantic_annotation()

        right_arm = (
            JustinRightArm.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_arm(right_arm)
        right_arm.setup_end_effector_semantic_annotation()

    def setup_neck_semantic_annotation(self):
        neck = JustinNeck.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        neck.setup_sensor_semantic_annotations()
        self.add_neck(neck)

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
            tip=world.get_body_in_branch_by_name(robot_root, "torso4"),
        )
        world.add_semantic_annotation(torso)
        return torso


@dataclass(eq=False)
class JustinMobileBase(MobileBase, HasTorso):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        torso = cls(
            root=world.get_body_in_branch_by_name(robot_root, "base_link"),
        )
        world.add_semantic_annotation(torso)
        return torso

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass

    def setup_torso_semantic_annotation(self):
        torso = JustinTorso.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_torso(torso)
        torso.setup_arm_semantic_annotations()
        torso.setup_neck_semantic_annotation()


@dataclass(eq=False)
class Justin(AbstractRobot, HasMobileBase):

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "base_footprint"

    def setup_robot_part_semantic_annotations(self):
        self.setup_mobile_base_semantic_annotation()

    def setup_mobile_base_semantic_annotation(self):
        mobile_base = (
            JustinMobileBase.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_mobile_base(mobile_base)
        mobile_base.setup_torso_semantic_annotation()
