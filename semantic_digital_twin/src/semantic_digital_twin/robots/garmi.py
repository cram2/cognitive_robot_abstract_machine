from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from importlib.resources import files
from pathlib import Path
from typing import List, Self

from krrood.ormatic.utils import classproperty
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
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import (
    HasLeftRightArm,
    HasMobileBase,
    HasNeck,
    HasTorso,
    HasTwoFingers,
)
from semantic_digital_twin.robots.robot_parts import (
    AbstractRobot,
    Arm,
    Camera,
    EndEffector,
    Finger,
    MobileBase,
    Neck,
    Torso,
)
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world_description.connections import OmniDrive
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


class GarmiJoint(StrEnum):
    """
    Names of GARMI's commandable connections, as spelled in its URDF.

    Members are usable wherever a connection name is expected.
    """

    LEFT_ARM_1 = "left_fr3_joint1"
    LEFT_ARM_2 = "left_fr3_joint2"
    LEFT_ARM_3 = "left_fr3_joint3"
    LEFT_ARM_4 = "left_fr3_joint4"
    LEFT_ARM_5 = "left_fr3_joint5"
    LEFT_ARM_6 = "left_fr3_joint6"
    LEFT_ARM_7 = "left_fr3_joint7"

    RIGHT_ARM_1 = "right_fr3_joint1"
    RIGHT_ARM_2 = "right_fr3_joint2"
    RIGHT_ARM_3 = "right_fr3_joint3"
    RIGHT_ARM_4 = "right_fr3_joint4"
    RIGHT_ARM_5 = "right_fr3_joint5"
    RIGHT_ARM_6 = "right_fr3_joint6"
    RIGHT_ARM_7 = "right_fr3_joint7"

    LEFT_FINGER_1 = "left_fr3_finger_joint1"
    LEFT_FINGER_2 = "left_fr3_finger_joint2"
    RIGHT_FINGER_1 = "right_fr3_finger_joint1"
    RIGHT_FINGER_2 = "right_fr3_finger_joint2"

    FRONT_LEFT_WHEEL = "front_left_wheel_joint"
    FRONT_RIGHT_WHEEL = "front_right_wheel_joint"
    REAR_LEFT_WHEEL = "rear_left_wheel_joint"
    REAR_RIGHT_WHEEL = "rear_right_wheel_joint"

    HEAD_PAN = "o1_motor_1"
    HEAD_TILT = "o1_motor_2"

    LIFT_LOWER = "lift_0_lower_joint"
    LIFT_UPPER = "lift_0_upper_joint"

    @classmethod
    def left_arm(cls) -> List[GarmiJoint]:
        """
        :return: The seven left FR3 arm joints, ordered from base to tip.
        """
        return [
            cls.LEFT_ARM_1,
            cls.LEFT_ARM_2,
            cls.LEFT_ARM_3,
            cls.LEFT_ARM_4,
            cls.LEFT_ARM_5,
            cls.LEFT_ARM_6,
            cls.LEFT_ARM_7,
        ]

    @classmethod
    def right_arm(cls) -> List[GarmiJoint]:
        """
        :return: The seven right FR3 arm joints, ordered from base to tip.
        """
        return [
            cls.RIGHT_ARM_1,
            cls.RIGHT_ARM_2,
            cls.RIGHT_ARM_3,
            cls.RIGHT_ARM_4,
            cls.RIGHT_ARM_5,
            cls.RIGHT_ARM_6,
            cls.RIGHT_ARM_7,
        ]

    @classmethod
    def left_fingers(cls) -> List[GarmiJoint]:
        """
        :return: The two finger joints of the left FR3 gripper.
        """
        return [cls.LEFT_FINGER_1, cls.LEFT_FINGER_2]

    @classmethod
    def right_fingers(cls) -> List[GarmiJoint]:
        """
        :return: The two finger joints of the right FR3 gripper.
        """
        return [cls.RIGHT_FINGER_1, cls.RIGHT_FINGER_2]

    @classmethod
    def wheels(cls) -> List[GarmiJoint]:
        """
        :return: The four mecanum wheel joints of the base.
        """
        return [
            cls.FRONT_LEFT_WHEEL,
            cls.FRONT_RIGHT_WHEEL,
            cls.REAR_LEFT_WHEEL,
            cls.REAR_RIGHT_WHEEL,
        ]

    @classmethod
    def head(cls) -> List[GarmiJoint]:
        """
        :return: The head's pan and tilt joints.
        """
        return [cls.HEAD_PAN, cls.HEAD_TILT]

    @classmethod
    def lift(cls) -> List[GarmiJoint]:
        """
        :return: The two prismatic torso lift joints.
        """
        return [cls.LIFT_LOWER, cls.LIFT_UPPER]


@dataclass(eq=False)
class GarmiCamera(Camera):
    """
    The head camera of the GARMI robot.
    """

    def setup_hardware_interfaces(self):
        """
        No hardware interface for the camera itself.
        """

    def setup_joint_states(self) -> List[JointState]:
        """
        No joint states for the camera.
        """
        return []

    @property
    def forward_facing_axis(self) -> Vector3:
        return Vector3.X(reference_frame=self.root)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        """
        Sets up the default configuration for the camera.
        """
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "head"),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=0.75049,
            maximal_height=0.99483,
            default_camera=True,
        )


@dataclass(eq=False)
class GarmiNeck(Neck[GarmiCamera]):
    """
    The pan/tilt neck of the GARMI robot.
    """

    def setup_hardware_interfaces(self):
        """
        Sets up hardware interfaces for the neck's pan and tilt joints.
        """
        for joint_name in GarmiJoint.head():
            self._world.get_connection_by_name(joint_name).has_hardware_interface = True

    def setup_joint_states(self) -> List[JointState]:
        """
        No default joint states for the neck.
        """
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        """
        Sets up the default configuration for the neck.
        """
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "neck_1"),
            tip=robot_root._world.get_body_in_branch_by_name(robot_root, "head"),
        )


@dataclass(eq=False)
class GarmiLeftGripperLeftFinger(Finger):
    """
    The left finger of the left gripper.
    """

    def setup_hardware_interfaces(self):
        """
        No separate hardware interface for the finger.
        """

    def setup_joint_states(self) -> List[JointState]:
        """
        No separate joint states for the finger.
        """
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        """
        Sets up the default configuration for the finger.
        """
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_fr3_hand"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_fr3_leftfinger"
            ),
        )


@dataclass(eq=False)
class GarmiLeftGripperRightFinger(Finger):
    """
    The right finger of the left gripper.
    """

    def setup_hardware_interfaces(self):
        """
        No separate hardware interface for the finger.
        """

    def setup_joint_states(self) -> List[JointState]:
        """
        No separate joint states for the finger.
        """
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        """
        Sets up the default configuration for the finger.
        """
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_fr3_hand"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_fr3_rightfinger"
            ),
        )


@dataclass(eq=False)
class GarmiRightGripperLeftFinger(Finger):
    """
    The left finger of the right gripper.
    """

    def setup_hardware_interfaces(self):
        """
        No separate hardware interface for the finger.
        """

    def setup_joint_states(self) -> List[JointState]:
        """
        No separate joint states for the finger.
        """
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        """
        Sets up the default configuration for the finger.
        """
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_fr3_hand"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_fr3_leftfinger"
            ),
        )


@dataclass(eq=False)
class GarmiRightGripperRightFinger(Finger):
    """
    The right finger of the right gripper.
    """

    def setup_hardware_interfaces(self):
        """
        No separate hardware interface for the finger.
        """

    def setup_joint_states(self) -> List[JointState]:
        """
        No separate joint states for the finger.
        """
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        """
        Sets up the default configuration for the finger.
        """
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_fr3_hand"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_fr3_rightfinger"
            ),
        )


@dataclass(eq=False)
class GarmiLeftGripper(
    EndEffector, HasTwoFingers[GarmiLeftGripperLeftFinger, GarmiLeftGripperRightFinger]
):
    """
    The left Franka parallel gripper.
    """

    def setup_hardware_interfaces(self):
        """
        Sets up hardware interfaces for the gripper's finger joints.
        """
        for joint_name in GarmiJoint.left_fingers():
            self._world.get_connection_by_name(joint_name).has_hardware_interface = True

    def setup_joint_states(self) -> List[JointState]:
        """
        Sets up open and close states for the gripper.
        """
        gripper_joints = self.active_connections
        gripper_open = JointState.from_mapping(
            name=PrefixedName("gripper_open", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.04, 0.04])),
            state_type=GripperState.OPEN,
        )
        gripper_close = JointState.from_mapping(
            name=PrefixedName("gripper_close", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.0, 0.0])),
            state_type=GripperState.CLOSE,
        )
        return [gripper_open, gripper_close]

    @property
    def approach_axis(self) -> Vector3:
        return Vector3.Z(reference_frame=self.tool_frame)

    @property
    def closing_axis(self) -> Vector3:
        return Vector3.Y(reference_frame=self.tool_frame)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        """
        Sets up the default configuration for the gripper.
        """
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_fr3_hand"
            ),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_fr3_hand_tcp"
            ),
        )


@dataclass(eq=False)
class GarmiRightGripper(
    EndEffector,
    HasTwoFingers[GarmiRightGripperLeftFinger, GarmiRightGripperRightFinger],
):
    """
    The right Franka parallel gripper.
    """

    def setup_hardware_interfaces(self):
        """
        Sets up hardware interfaces for the gripper's finger joints.
        """
        for joint_name in GarmiJoint.right_fingers():
            self._world.get_connection_by_name(joint_name).has_hardware_interface = True

    def setup_joint_states(self) -> List[JointState]:
        """
        Sets up open and close states for the gripper.
        """
        gripper_joints = self.active_connections
        gripper_open = JointState.from_mapping(
            name=PrefixedName("gripper_open", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.04, 0.04])),
            state_type=GripperState.OPEN,
        )
        gripper_close = JointState.from_mapping(
            name=PrefixedName("gripper_close", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.0, 0.0])),
            state_type=GripperState.CLOSE,
        )
        return [gripper_open, gripper_close]

    @property
    def approach_axis(self) -> Vector3:
        return Vector3.Z(reference_frame=self.tool_frame)

    @property
    def closing_axis(self) -> Vector3:
        return Vector3.Y(reference_frame=self.tool_frame)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        """
        Sets up the default configuration for the gripper.
        """
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_fr3_hand"
            ),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_fr3_hand_tcp"
            ),
        )


@dataclass(eq=False)
class GarmiLeftArm(Arm[GarmiLeftGripper]):
    """
    The left Franka FR3 arm.
    """

    def setup_hardware_interfaces(self):
        """
        Sets up hardware interfaces for the arm joints.
        """
        for joint_name in GarmiJoint.left_arm():
            self._world.get_connection_by_name(joint_name).has_hardware_interface = True

    def setup_joint_states(self) -> List[JointState]:
        """
        Sets up the park configuration for the arm.
        """
        park_configuration = {
            GarmiJoint.LEFT_ARM_1: 0.0,
            GarmiJoint.LEFT_ARM_2: -1.6,
            GarmiJoint.LEFT_ARM_3: -1.0,
            GarmiJoint.LEFT_ARM_4: -2.356194490192345,
            GarmiJoint.LEFT_ARM_5: 0.0,
            GarmiJoint.LEFT_ARM_6: 1.5707963267948966,
            GarmiJoint.LEFT_ARM_7: 0.7853981633974483,
        }

        arm_park = JointState.from_mapping(
            name=PrefixedName("park", prefix=self.name.name),
            mapping={
                self._world.get_connection_by_name(joint_name): position
                for joint_name, position in park_configuration.items()
            },
            state_type=StaticJointState.PARK,
        )
        return [arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        """
        Sets up the default configuration for the arm.
        """
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_mount_left_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "left_fr3_link8"
            ),
        )


@dataclass(eq=False)
class GarmiRightArm(Arm[GarmiRightGripper]):
    """
    The right Franka FR3 arm.
    """

    def setup_hardware_interfaces(self):
        """
        Sets up hardware interfaces for the arm joints.
        """
        for joint_name in GarmiJoint.right_arm():
            self._world.get_connection_by_name(joint_name).has_hardware_interface = True

    def setup_joint_states(self) -> List[JointState]:
        """
        Sets up the park configuration for the arm.
        """
        park_configuration = {
            GarmiJoint.RIGHT_ARM_1: 0.0,
            GarmiJoint.RIGHT_ARM_2: -1.6,
            GarmiJoint.RIGHT_ARM_3: 1.0,
            GarmiJoint.RIGHT_ARM_4: -2.356194490192345,
            GarmiJoint.RIGHT_ARM_5: 0.0,
            GarmiJoint.RIGHT_ARM_6: 1.5707963267948966,
            GarmiJoint.RIGHT_ARM_7: 0.7853981633974483,
        }

        arm_park = JointState.from_mapping(
            name=PrefixedName("park", prefix=self.name.name),
            mapping={
                self._world.get_connection_by_name(joint_name): position
                for joint_name, position in park_configuration.items()
            },
            state_type=StaticJointState.PARK,
        )
        return [arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        """
        Sets up the default configuration for the arm.
        """
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_mount_right_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "right_fr3_link8"
            ),
        )


@dataclass(eq=False)
class GarmiTorso(
    Torso, HasLeftRightArm[GarmiLeftArm, GarmiRightArm], HasNeck[GarmiNeck]
):
    """
    The lift torso of the GARMI robot.
    """

    def setup_hardware_interfaces(self):
        """
        Sets up hardware interfaces for the lift joints.
        """
        for joint_name in GarmiJoint.lift():
            self._world.get_connection_by_name(joint_name).has_hardware_interface = True

    def setup_joint_states(self) -> List[JointState]:
        """
        Sets up torso states (low, mid, high).
        """
        lift_joints = [
            self._world.get_connection_by_name(joint_name)
            for joint_name in GarmiJoint.lift()
        ]
        torso_states = (
            ("torso_low", [0.0, 0.0], TorsoState.LOW),
            ("torso_mid", [0.2, 0.2], TorsoState.MID),
            ("torso_high", [0.4, 0.4], TorsoState.HIGH),
        )
        return [
            JointState.from_mapping(
                name=PrefixedName(name, prefix=self.name.name),
                mapping=dict(zip(lift_joints, positions)),
                state_type=state_type,
            )
            for name, positions, state_type in torso_states
        ]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        """
        Sets up the default configuration for the torso.
        """
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "lift_0_base_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "lift_0_mount_rotated_link"
            ),
        )


@dataclass(eq=False)
class GarmiMobileBase(MobileBase[OmniDrive], HasTorso[GarmiTorso]):
    """
    The mecanum mobile base of the GARMI robot.
    """

    @classproperty
    def forward_axis(cls) -> Vector3:
        return Vector3.X()

    def setup_hardware_interfaces(self):
        """
        Sets up hardware interfaces for the wheel joints.
        """
        for joint_name in GarmiJoint.wheels():
            self._world.get_connection_by_name(joint_name).has_hardware_interface = True

    def setup_joint_states(self) -> List[JointState]:
        """
        No default joint states for the mobile base.
        """
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        """
        Sets up the default configuration for the mobile base.
        """
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "chassis_link"
            ),
            full_body_controlled=False,
        )


@dataclass(eq=False)
class Garmi(AbstractRobot, HasMobileBase[GarmiMobileBase]):
    """
    Semantic annotation for GARMI, a mobile service robot with a mecanum base, lift, two
    Franka FR3 arms, parallel grippers, and a pan/tilt head.
    """

    @classproperty
    def uses_visual_as_collision_backup(cls) -> bool:
        """
        :return: True, since GARMI's shell -- the side, front and rear covers -- is
            drawn but never described for contact, so without this the parts that bound
            its real width do not collide.
        """
        return True

    @classmethod
    def get_ros_file_path(cls) -> str:
        """
        Returns the ROS file path for the GARMI robot description.
        """
        return "package://garmi_description/urdf/garmi.urdf"

    @classmethod
    def _get_root_body_name(cls) -> str:
        """
        Returns the name of the root body for the GARMI robot.
        """
        return "base_link"

    def _setup_velocity_limits(self):
        """
        Sets up velocity limits for the robot's joints.
        """
        vel_limits = defaultdict(lambda: 0.2)
        for joint_name in GarmiJoint.wheels():
            vel_limits[self._world.get_connection_by_name(joint_name)] = 1.3
        for joint_name in GarmiJoint.head():
            vel_limits[self._world.get_connection_by_name(joint_name)] = 1.0
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)

    def _setup_collision_rules(self):
        """
        Sets up collision avoidance rules for the robot, including SRDF-based self-
        collision ignore rules.
        """
        srdf_path = os.path.join(
            Path(files("semantic_digital_twin")).parent.parent,
            "resources",
            "collision_configs",
            "garmi.srdf",
        )
        self._world.collision_manager.add_ignore_collision_rule(
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

        # The base's shell is what bumps into furniture first, so it keeps a wider berth
        # than the rest of the robot.
        self._world.collision_manager.add_default_rule(
            AvoidExternalCollisions(
                buffer_zone_distance=0.2,
                violated_distance=0.05,
                robot=self,
                body_subset={
                    self._world.get_body_in_branch_by_name(self.root, body_name)
                    for body_name in (
                        "right_side_cover_link",
                        "left_side_cover_link",
                        "front_cover_link",
                        "rear_cover_link",
                        "cover_link",
                    )
                },
            )
        )
