from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import ClassVar, List, Self

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
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


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

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        """
        Sets up the default configuration for the camera.
        """
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "head"),
            forward_facing_axis=Vector3(1, 0, 0),
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
        for joint_name in ("head_pan_joint", "head_tilt_joint"):
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
                robot_root, "arm_0_gripper_fr3_hand"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_0_gripper_fr3_leftfinger"
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
                robot_root, "arm_0_gripper_fr3_hand"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_0_gripper_fr3_rightfinger"
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
                robot_root, "arm_1_gripper_fr3_hand"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_1_gripper_fr3_leftfinger"
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
                robot_root, "arm_1_gripper_fr3_hand"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_1_gripper_fr3_rightfinger"
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
        for joint_name in (
            "arm_0_gripper_fr3_finger_joint1",
            "arm_0_gripper_fr3_finger_joint2",
        ):
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

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        """
        Sets up the default configuration for the gripper.
        """
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_0_gripper_fr3_hand"
            ),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_0_gripper_fr3_hand_tcp"
            ),
            front_facing_orientation=Quaternion(0, 0, 0, 1),
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
        for joint_name in (
            "arm_1_gripper_fr3_finger_joint1",
            "arm_1_gripper_fr3_finger_joint2",
        ):
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

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        """
        Sets up the default configuration for the gripper.
        """
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_1_gripper_fr3_hand"
            ),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_1_gripper_fr3_hand_tcp"
            ),
            front_facing_orientation=Quaternion(0, 0, 0, 1),
        )


@dataclass(eq=False)
class GarmiLeftArm(Arm[GarmiLeftGripper]):
    """
    The left Franka FR3 arm.
    """

    ARM_PARK_CONFIGURATION: ClassVar[dict[str, float]] = {
        "fr3_joint1": 0.0,
        "fr3_joint2": -0.7853981633974483,
        "fr3_joint3": 0.0,
        "fr3_joint4": -2.356194490192345,
        "fr3_joint5": 0.0,
        "fr3_joint6": 1.5707963267948966,
        "fr3_joint7": 0.7853981633974483,
    }

    def setup_hardware_interfaces(self):
        """
        Sets up hardware interfaces for the arm joints.
        """
        for joint_index in range(1, 8):
            self._world.get_connection_by_name(
                f"arm_0_fr3_joint{joint_index}"
            ).has_hardware_interface = True

    def setup_joint_states(self) -> List[JointState]:
        """
        Sets up the park configuration for the arm.
        """
        arm_park = JointState.from_mapping(
            name=PrefixedName("park", prefix=self.name.name),
            mapping={
                connection: position
                for connection in self.connections
                if not isinstance(connection, FixedConnection)
                for joint_name, position in self.ARM_PARK_CONFIGURATION.items()
                if connection.name.name.endswith(joint_name)
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
                robot_root, "arm_0_fr3_link8"
            ),
        )


@dataclass(eq=False)
class GarmiRightArm(Arm[GarmiRightGripper]):
    """
    The right Franka FR3 arm.
    """

    ARM_PARK_CONFIGURATION: ClassVar[dict[str, float]] = {
        "fr3_joint1": 0.0,
        "fr3_joint2": -0.7853981633974483,
        "fr3_joint3": 0.0,
        "fr3_joint4": -2.356194490192345,
        "fr3_joint5": 0.0,
        "fr3_joint6": 1.5707963267948966,
        "fr3_joint7": 0.7853981633974483,
    }

    def setup_hardware_interfaces(self):
        """
        Sets up hardware interfaces for the arm joints.
        """
        for joint_index in range(1, 8):
            self._world.get_connection_by_name(
                f"arm_1_fr3_joint{joint_index}"
            ).has_hardware_interface = True

    def setup_joint_states(self) -> List[JointState]:
        """
        Sets up the park configuration for the arm.
        """
        arm_park = JointState.from_mapping(
            name=PrefixedName("park", prefix=self.name.name),
            mapping={
                connection: position
                for connection in self.connections
                if not isinstance(connection, FixedConnection)
                for joint_name, position in self.ARM_PARK_CONFIGURATION.items()
                if connection.name.name.endswith(joint_name)
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
                robot_root, "arm_1_fr3_link8"
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
        for joint_name in ("lift_0_lower_joint", "lift_0_upper_joint"):
            self._world.get_connection_by_name(joint_name).has_hardware_interface = True

    def setup_joint_states(self) -> List[JointState]:
        """
        Sets up torso states (low, mid, high).
        """
        lift_joints = [
            self._world.get_connection_by_name("lift_0_lower_joint"),
            self._world.get_connection_by_name("lift_0_upper_joint"),
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
class GarmiMobileBase(MobileBase, HasTorso[GarmiTorso]):
    """
    The mecanum mobile base of the GARMI robot.
    """

    def setup_hardware_interfaces(self):
        """
        Sets up hardware interfaces for the wheel joints.
        """
        for joint_name in (
            "front_left_wheel_joint",
            "front_right_wheel_joint",
            "rear_left_wheel_joint",
            "rear_right_wheel_joint",
        ):
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
            full_body_controlled=True,
        )


@dataclass(eq=False)
class Garmi(AbstractRobot, HasMobileBase[GarmiMobileBase]):
    """
    Semantic annotation for GARMI, a mobile service robot with a mecanum base, lift, two
    Franka FR3 arms, parallel grippers, and a pan/tilt head.
    """

    @classmethod
    def get_ros_file_path(cls) -> str:
        """
        Returns the ROS file path for the GARMI robot description.
        """
        raise NotImplementedError("We dont have the ROS Package yet")

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
        for joint_name in (
            "front_left_wheel_joint",
            "front_right_wheel_joint",
            "rear_left_wheel_joint",
            "rear_right_wheel_joint",
        ):
            vel_limits[self._world.get_connection_by_name(joint_name)] = 1.3
        for joint_name in ("head_pan_joint", "head_tilt_joint"):
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
