from __future__ import annotations

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
    HasOneArm,
    HasParallelGripper,
    HasTorso,
    HasNeck,
    HasCameras,
    HasMobileBase,
)
from semantic_digital_twin.robots.robot_parts import (
    AbstractRobot,
    Arm,
    Finger,
    ParallelGripper,
    Torso,
    Neck,
    Camera,
    FieldOfView,
    MobileBase,
)
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@dataclass(eq=False)
class DonbotFinger(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass


@dataclass(eq=False)
class DonbotLeftFinger(DonbotFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(
                robot_root, "gripper_gripper_left_link"
            ),
            tip=world.get_body_in_branch_by_name(
                robot_root, "gripper_finger_right_link"
            ),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class DonbotRightFinger(DonbotFinger):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(
                robot_root, "gripper_gripper_left_link"
            ),
            tip=world.get_body_in_branch_by_name(
                robot_root, "gripper_finger_left_link"
            ),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class DonbotGripper(ParallelGripper):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        world = self._world
        gripper_joints = [
            world.get_connection_by_name("gripper_joint"),
            world.get_connection_by_name("gripper_base_gripper_left_joint"),
        ]

        gripper_open = JointState.from_mapping(
            name=PrefixedName("gripper_open", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.109, -0.055])),
            state_type=GripperState.OPEN,
        )

        gripper_close = JointState.from_mapping(
            name=PrefixedName("gripper_close", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.0065, -0.0027])),
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
            root=world.get_body_in_branch_by_name(robot_root, "gripper_base_link"),
            tool_frame=world.get_body_in_branch_by_name(
                robot_root, "gripper_tool_frame"
            ),
            front_facing_orientation=Quaternion(0.707, -0.707, 0.707, -0.707),
        )
        world.add_semantic_annotation(gripper)
        return gripper

    def setup_finger_semantic_annotations(self):
        thumb = DonbotLeftFinger.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_thumb(thumb)
        finger = (
            DonbotRightFinger.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        self.add_finger(finger)


@dataclass(eq=False)
class DonbotCamera(Camera):

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
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=0.5,
            maximal_height=1.2,
        )
        world.add_semantic_annotation(camera)
        return camera


@dataclass(eq=False)
class DonbotArm(Arm, HasParallelGripper, HasCameras):

    def setup_sensor_semantic_annotations(self):
        camera = DonbotCamera.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_sensor(camera)

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        arm_park = JointState.from_mapping(
            name=PrefixedName("arm_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    self.active_connections,
                    [3.23, -1.51, -1.57, 0.0, 1.57, -1.65],
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
            root=world.get_body_in_branch_by_name(robot_root, "ur5_base_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "ur5_wrist_3_link"),
        )
        world.add_semantic_annotation(arm)
        return arm

    def setup_end_effector_semantic_annotation(self):
        gripper = DonbotGripper.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_end_effector(gripper)
        gripper.setup_finger_semantic_annotations()


@dataclass(eq=False)
class DonbotMobileBase(MobileBase, HasOneArm):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass

    def setup_arm_semantic_annotations(self):
        arm = DonbotArm.setup_default_configuration_in_world_below_robot_root(self.root)
        self.add_arm(arm)
        arm.setup_end_effector_semantic_annotation()

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        mobile_base = cls(
            root=world.get_body_in_branch_by_name(robot_root, "base_footprint"),
        )
        world.add_semantic_annotation(mobile_base)
        return mobile_base


@dataclass(eq=False)
class Donbot(AbstractRobot, HasMobileBase):

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "base_footprint"

    def setup_mobile_base_semantic_annotation(self):
        mobile_base = (
            DonbotMobileBase.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        mobile_base.setup_arm_semantic_annotations()
        self.add_mobile_base(mobile_base)

    def setup_robot_part_semantic_annotations(self):
        self.setup_mobile_base_semantic_annotation()
