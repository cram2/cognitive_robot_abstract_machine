from __future__ import annotations

import math
import os
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from importlib.resources import files
from pathlib import Path
from typing import Self, List

from krrood.ormatic.utils import classproperty
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.sensors.lidar import Lidar, LidarSource
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
from semantic_digital_twin.datastructures.scan_pattern import ScanPattern
from semantic_digital_twin.robots.robot_part_mixins import (
    HasLidar,
    HasLeftRightArm,
    HasMobileBase,
    HasNeck,
    HasTorso,
    HasTwoFingers,
    TGenericLeftFinger,
    TGenericRightFinger,
    HasEndEffector,
)
from semantic_digital_twin.robots.robot_parts import (
    AbstractRobot,
    AbstractRobotPart,
    Arm,
    Camera,
    Finger,
    MobileBase,
    Neck,
    Torso,
    EndEffector,
)
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world_description.connection_properties import (
    DriveServos,
    JointServo,
    ServoGains,
    VelocityServoGains,
)
from semantic_digital_twin.world_description.connections import (
    ActiveConnection,
    ActiveConnection1DOF,
    DifferentialDrive,
)
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


class TiagoJoint(StrEnum):
    """
    Names of the Tiago's commandable connections, as spelled in its URDF.

    Members are usable wherever a connection name is expected, so a configuration keyed by
    them stays a plain mapping of names to positions.

    ..note:: Connections that no controller commands, such as the base wheels and the
        gripper's coupled knuckle and inner finger joints, are left out.
    """

    TORSO_LIFT = "torso_lift_joint"
    HEAD_1 = "head_1_joint"
    HEAD_2 = "head_2_joint"

    LEFT_ARM_1 = "arm_left_1_joint"
    LEFT_ARM_2 = "arm_left_2_joint"
    LEFT_ARM_3 = "arm_left_3_joint"
    LEFT_ARM_4 = "arm_left_4_joint"
    LEFT_ARM_5 = "arm_left_5_joint"
    LEFT_ARM_6 = "arm_left_6_joint"
    LEFT_ARM_7 = "arm_left_7_joint"
    LEFT_GRIPPER_FINGER = "gripper_left_finger_joint"

    RIGHT_ARM_1 = "arm_right_1_joint"
    RIGHT_ARM_2 = "arm_right_2_joint"
    RIGHT_ARM_3 = "arm_right_3_joint"
    RIGHT_ARM_4 = "arm_right_4_joint"
    RIGHT_ARM_5 = "arm_right_5_joint"
    RIGHT_ARM_6 = "arm_right_6_joint"
    RIGHT_ARM_7 = "arm_right_7_joint"
    RIGHT_GRIPPER_FINGER = "gripper_right_finger_joint"


@dataclass(eq=False)
class TiagoLeftThumb(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_left_base_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_left_left_inner_finger_pad"
            ),
        )


@dataclass(eq=False)
class TiagoLeftIndexFinger(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_left_base_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_left_right_inner_finger_pad"
            ),
        )


@dataclass(eq=False)
class TiagoRightThumb(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_right_base_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_right_left_inner_finger_pad"
            ),
        )


@dataclass(eq=False)
class TiagoRightIndexFinger(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_right_base_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_right_right_inner_finger_pad"
            ),
        )


@dataclass(eq=False)
class TiagoLeftGripper(
    EndEffector, HasTwoFingers[TiagoLeftThumb, TiagoLeftIndexFinger]
):

    def setup_hardware_interfaces(self):
        controlled_joints = [
            TiagoJoint.LEFT_GRIPPER_FINGER,
        ]
        for joint_name in controlled_joints:
            connection: ActiveConnection = self._world.get_connection_by_name(
                joint_name
            )
            connection.has_hardware_interface = True

    def setup_joint_states(self) -> List[JointState]:
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

        return [gripper_open, gripper_close]

    @property
    def approach_axis(self) -> Vector3:
        return Vector3.X(reference_frame=self.tool_frame)

    @property
    def closing_axis(self) -> Vector3:
        return Vector3.Y(reference_frame=self.tool_frame)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_left_base_link"
            ),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_left_grasping_frame"
            ),
        )


@dataclass(eq=False)
class TiagoRightGripper(
    EndEffector, HasTwoFingers[TiagoRightThumb, TiagoRightIndexFinger]
):

    def setup_hardware_interfaces(self):
        controlled_joints = [
            TiagoJoint.RIGHT_GRIPPER_FINGER,
        ]
        for joint_name in controlled_joints:
            connection: ActiveConnection = self._world.get_connection_by_name(
                joint_name
            )
            connection.has_hardware_interface = True

    def setup_joint_states(self) -> List[JointState]:
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

        return [gripper_open, gripper_close]

    @property
    def approach_axis(self) -> Vector3:
        return Vector3.X(reference_frame=self.tool_frame)

    @property
    def closing_axis(self) -> Vector3:
        return Vector3.Y(reference_frame=self.tool_frame)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_right_base_link"
            ),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_right_grasping_frame"
            ),
        )


@dataclass(eq=False)
class TiagoLeftArm(Arm[TiagoLeftGripper]):

    def setup_hardware_interfaces(self):
        controlled_joints = [
            TiagoJoint.LEFT_ARM_1,
            TiagoJoint.LEFT_ARM_2,
            TiagoJoint.LEFT_ARM_3,
            TiagoJoint.LEFT_ARM_4,
            TiagoJoint.LEFT_ARM_5,
            TiagoJoint.LEFT_ARM_6,
            TiagoJoint.LEFT_ARM_7,
        ]
        for joint_name in controlled_joints:
            connection: ActiveConnection = self._world.get_connection_by_name(
                joint_name
            )
            connection.has_hardware_interface = True

    def setup_joint_states(self) -> List[JointState]:
        arm_park = JointState.from_mapping(
            name=PrefixedName("left_arm_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    self.active_connections,
                    [0, -0.8, 1.57, 1.57, -2.0, 1.1, 0.0],
                )
            ),
            state_type=StaticJointState.PARK,
        )
        return [arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "torso_lift_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_left_tool_link"
            ),
        )


@dataclass(eq=False)
class TiagoRightArm(Arm[TiagoRightGripper]):

    def setup_hardware_interfaces(self):
        controlled_joints = [
            TiagoJoint.RIGHT_ARM_1,
            TiagoJoint.RIGHT_ARM_2,
            TiagoJoint.RIGHT_ARM_3,
            TiagoJoint.RIGHT_ARM_4,
            TiagoJoint.RIGHT_ARM_5,
            TiagoJoint.RIGHT_ARM_6,
            TiagoJoint.RIGHT_ARM_7,
        ]
        for joint_name in controlled_joints:
            connection: ActiveConnection = self._world.get_connection_by_name(
                joint_name
            )
            connection.has_hardware_interface = True

    def setup_joint_states(self) -> List[JointState]:
        arm_park = JointState.from_mapping(
            name=PrefixedName("right_arm_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    self.active_connections,
                    [0, -0.8, 1.57, 1.57, -2.0, 1.1, 0.0],
                )
            ),
            state_type=StaticJointState.PARK,
        )
        return [arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "torso_lift_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_right_tool_link"
            ),
        )


@dataclass(eq=False)
class TiagoCamera(Camera):

    @property
    def forward_facing_axis(self) -> Vector3:
        return Vector3.Z(reference_frame=self.root)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "head_front_camera_optical_frame"
            ),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=1.0665,
            maximal_height=1.4165,
            default_camera=True,
        )

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []


@dataclass(eq=False)
class TiagoNeck(Neck[TiagoCamera]):

    def setup_hardware_interfaces(self):
        controlled_joints = [
            TiagoJoint.HEAD_1,
            TiagoJoint.HEAD_2,
        ]
        for joint_name in controlled_joints:
            connection: ActiveConnection = self._world.get_connection_by_name(
                joint_name
            )
            connection.has_hardware_interface = True

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "torso_lift_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(robot_root, "head_2_link"),
        )


@dataclass(eq=False)
class TiagoTorso(
    Torso, HasLeftRightArm[TiagoLeftArm, TiagoRightArm], HasNeck[TiagoNeck]
):

    def setup_hardware_interfaces(self):
        controlled_joints = [
            TiagoJoint.TORSO_LIFT,
        ]
        for joint_name in controlled_joints:
            connection: ActiveConnection = self._world.get_connection_by_name(
                joint_name
            )
            connection.has_hardware_interface = True

    def setup_joint_states(self) -> List[JointState]:
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
            mapping=dict(zip(torso_joint, [0.34])),
            state_type=TorsoState.HIGH,
        )

        return [torso_low, torso_mid, torso_high]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "torso_fixed_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "torso_lift_link"
            ),
        )


@dataclass(eq=False)
class TiagoBaseLidar(Lidar):
    """
    The SICK TIM551 scanner sweeping the floor in front of the Tiago's base.

    ..note:: The description's own beam count is not a whole number, so the scanner's
        angular resolution of a third of a degree gives the angle between two beams.
    """

    @classmethod
    def with_source(
        cls, robot_root: KinematicStructureEntity, source: LidarSource
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "base_laser_link"
            ),
            scan_pattern=ScanPattern(
                minimum_angle=-1.658133,
                maximum_angle=1.66347956,
                angle_increment=0.00581718,
                minimum_range=0.05,
                maximum_range=10.0,
            ),
            source=source,
        )


@dataclass(eq=False)
class TiagoMobileBase(
    MobileBase[DifferentialDrive],
    HasTorso[TiagoTorso],
    HasLidar[TiagoBaseLidar],
):

    @classproperty
    def forward_axis(cls) -> Vector3:
        return Vector3.X()

    full_body_controlled: bool = field(default=True, kw_only=True)

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "base_link"),
        )


@dataclass(eq=False)
class Tiago(AbstractRobot, HasMobileBase[TiagoMobileBase]):
    """
    The Tiago++ robot by PAL Robotics with updated Robotiq grippers.

    https://pal-robotics.com/blog/tiago-bi-manual-robot-research/
    """

    @classmethod
    def get_ros_file_path(cls) -> str:
        return "package://iai_tiago_description/urdf/tiago_from_our_robot.urdf"

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "base_footprint"

    def _setup_collision_rules(self):
        srdf_path = os.path.join(
            Path(files("semantic_digital_twin")).parent.parent,
            "resources",
            "collision_configs",
            "tiago_from_our_robot.srdf",
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


# %% names in MuJoCo's own model of the robot


class TiagoMujocoFingerJoint(StrEnum):
    """
    Names of the sliding finger joints of MuJoCo's own model of the Tiago, which has
    them in place of the URDF's linkage; its other joints are named as in
    :class:`TiagoJoint`.
    """

    LEFT_GRIPPER_LEFT_FINGER = "gripper_left_left_finger_joint"
    LEFT_GRIPPER_RIGHT_FINGER = "gripper_left_right_finger_joint"
    RIGHT_GRIPPER_LEFT_FINGER = "gripper_right_left_finger_joint"
    RIGHT_GRIPPER_RIGHT_FINGER = "gripper_right_right_finger_joint"


class TiagoMujocoBody(StrEnum):
    """
    Names of the bodies of MuJoCo's own model of the Tiago that its parts are rooted in
    or end at.
    """

    BASE = "base_link"
    TORSO_LIFT = "torso_lift_link"
    HEAD = "head_2_link"

    LEFT_WRIST = "arm_left_7_link"
    LEFT_GRIPPER_GRASPING_FRAME = "gripper_left_grasping_frame"
    LEFT_GRIPPER_LEFT_FINGER_PAD = "gripper_left_left_finger_pad"
    LEFT_GRIPPER_RIGHT_FINGER_PAD = "gripper_left_right_finger_pad"

    RIGHT_WRIST = "arm_right_7_link"
    RIGHT_GRIPPER_GRASPING_FRAME = "gripper_right_grasping_frame"
    RIGHT_GRIPPER_LEFT_FINGER_PAD = "gripper_right_left_finger_pad"
    RIGHT_GRIPPER_RIGHT_FINGER_PAD = "gripper_right_right_finger_pad"


# %% what MuJoCo's own model of the robot drives its joints with


@dataclass(eq=False)
class ServoedByItsMujocoModel(AbstractRobotPart, ABC):
    """
    A part of the MuJoCo Tiago whose joints are driven in a physical simulation by the
    servos of :attr:`servo_of_each_joint`.
    """

    @classproperty
    def servo_of_each_joint(cls) -> dict[str, JointServo]:
        """
        What drives every servoed joint of the MuJoCo description, by the name that
        description gives it: the stiffness and force limit of the position actuators
        of ``tiago_dual_position.xml``, and the damping MuJoCo works out for the damping
        ratio of 1 they state.
        """
        return {
            TiagoJoint.TORSO_LIFT: JointServo(
                gains=ServoGains(stiffness=50000.0, damping=2361.08, torque_limit=500.0)
            ),
            TiagoJoint.HEAD_1: JointServo(
                gains=ServoGains(stiffness=20.0, damping=0.83, torque_limit=5.197)
            ),
            TiagoJoint.HEAD_2: JointServo(
                gains=ServoGains(stiffness=20.0, damping=0.87, torque_limit=5.197)
            ),
            TiagoJoint.LEFT_ARM_1: JointServo(
                gains=ServoGains(stiffness=250.0, damping=48.53, torque_limit=43.0)
            ),
            TiagoJoint.LEFT_ARM_2: JointServo(
                gains=ServoGains(stiffness=1500.0, damping=94.28, torque_limit=43.0)
            ),
            TiagoJoint.LEFT_ARM_3: JointServo(
                gains=ServoGains(stiffness=600.0, damping=4.36, torque_limit=26.0)
            ),
            TiagoJoint.LEFT_ARM_4: JointServo(
                gains=ServoGains(stiffness=1000.0, damping=35.38, torque_limit=26.0)
            ),
            TiagoJoint.LEFT_ARM_5: JointServo(
                gains=ServoGains(stiffness=80.0, damping=0.97, torque_limit=3.0)
            ),
            TiagoJoint.LEFT_ARM_6: JointServo(
                gains=ServoGains(stiffness=80.0, damping=2.57, torque_limit=6.6)
            ),
            TiagoJoint.LEFT_ARM_7: JointServo(
                gains=ServoGains(stiffness=80.0, damping=0.54, torque_limit=6.6)
            ),
            TiagoJoint.RIGHT_ARM_1: JointServo(
                gains=ServoGains(stiffness=250.0, damping=45.14, torque_limit=43.0)
            ),
            TiagoJoint.RIGHT_ARM_2: JointServo(
                gains=ServoGains(stiffness=1500.0, damping=87.68, torque_limit=43.0)
            ),
            TiagoJoint.RIGHT_ARM_3: JointServo(
                gains=ServoGains(stiffness=600.0, damping=3.90, torque_limit=26.0)
            ),
            TiagoJoint.RIGHT_ARM_4: JointServo(
                gains=ServoGains(stiffness=1000.0, damping=33.51, torque_limit=26.0)
            ),
            TiagoJoint.RIGHT_ARM_5: JointServo(
                gains=ServoGains(stiffness=80.0, damping=0.66, torque_limit=3.0)
            ),
            TiagoJoint.RIGHT_ARM_6: JointServo(
                gains=ServoGains(stiffness=80.0, damping=2.56, torque_limit=6.6)
            ),
            TiagoJoint.RIGHT_ARM_7: JointServo(
                gains=ServoGains(stiffness=80.0, damping=0.53, torque_limit=6.6)
            ),
            TiagoMujocoFingerJoint.LEFT_GRIPPER_LEFT_FINGER: JointServo(
                gains=ServoGains(stiffness=500.0, damping=14.63, torque_limit=5.197)
            ),
            TiagoMujocoFingerJoint.LEFT_GRIPPER_RIGHT_FINGER: JointServo(
                gains=ServoGains(stiffness=500.0, damping=14.63, torque_limit=5.197)
            ),
            TiagoMujocoFingerJoint.RIGHT_GRIPPER_LEFT_FINGER: JointServo(
                gains=ServoGains(stiffness=500.0, damping=14.63, torque_limit=5.197)
            ),
            TiagoMujocoFingerJoint.RIGHT_GRIPPER_RIGHT_FINGER: JointServo(
                gains=ServoGains(stiffness=500.0, damping=14.63, torque_limit=5.197)
            ),
        }

    def _setup_servos(self) -> None:
        for connection in self.active_connections:
            if not isinstance(connection, ActiveConnection1DOF):
                continue
            self._declare_servo(
                connection, self.servo_of_each_joint[connection.name.name]
            )
        self._compensate_gravity()


@dataclass(eq=False)
class TiagoMujocoLeftThumb(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, TiagoMujocoBody.LEFT_WRIST
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, TiagoMujocoBody.LEFT_GRIPPER_LEFT_FINGER_PAD
            ),
        )


@dataclass(eq=False)
class TiagoMujocoLeftIndexFinger(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, TiagoMujocoBody.LEFT_WRIST
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, TiagoMujocoBody.LEFT_GRIPPER_RIGHT_FINGER_PAD
            ),
        )


@dataclass(eq=False)
class TiagoMujocoRightThumb(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, TiagoMujocoBody.RIGHT_WRIST
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, TiagoMujocoBody.RIGHT_GRIPPER_LEFT_FINGER_PAD
            ),
        )


@dataclass(eq=False)
class TiagoMujocoRightIndexFinger(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, TiagoMujocoBody.RIGHT_WRIST
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, TiagoMujocoBody.RIGHT_GRIPPER_RIGHT_FINGER_PAD
            ),
        )


@dataclass(eq=False)
class TiagoMujocoLeftGripper(
    EndEffector,
    HasTwoFingers[TiagoMujocoLeftThumb, TiagoMujocoLeftIndexFinger],
    ServoedByItsMujocoModel,
):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        """
        The hand opens and closes to the ends of its fingers' travel.
        """
        gripper_joints = self.active_connections
        gripper_open = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_open", prefix=self.name.name),
            mapping={
                connection: connection.raw_dof.limits.upper.position
                for connection in gripper_joints
            },
            state_type=GripperState.OPEN,
        )
        gripper_close = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_close", prefix=self.name.name),
            mapping={
                connection: connection.raw_dof.limits.lower.position
                for connection in gripper_joints
            },
            state_type=GripperState.CLOSE,
        )
        return [gripper_close, gripper_open]

    @property
    def approach_axis(self) -> Vector3:
        return Vector3.X(reference_frame=self.tool_frame)

    @property
    def closing_axis(self) -> Vector3:
        return Vector3.Y(reference_frame=self.tool_frame)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, TiagoMujocoBody.LEFT_WRIST
            ),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, TiagoMujocoBody.LEFT_GRIPPER_GRASPING_FRAME
            ),
        )


@dataclass(eq=False)
class TiagoMujocoRightGripper(
    EndEffector,
    HasTwoFingers[TiagoMujocoRightThumb, TiagoMujocoRightIndexFinger],
    ServoedByItsMujocoModel,
):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        """
        The hand opens and closes to the ends of its fingers' travel.
        """
        gripper_joints = self.active_connections
        gripper_open = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_open", prefix=self.name.name),
            mapping={
                connection: connection.raw_dof.limits.upper.position
                for connection in gripper_joints
            },
            state_type=GripperState.OPEN,
        )
        gripper_close = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_close", prefix=self.name.name),
            mapping={
                connection: connection.raw_dof.limits.lower.position
                for connection in gripper_joints
            },
            state_type=GripperState.CLOSE,
        )
        return [gripper_close, gripper_open]

    @property
    def approach_axis(self) -> Vector3:
        return Vector3.X(reference_frame=self.tool_frame)

    @property
    def closing_axis(self) -> Vector3:
        return Vector3.Y(reference_frame=self.tool_frame)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, TiagoMujocoBody.RIGHT_WRIST
            ),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, TiagoMujocoBody.RIGHT_GRIPPER_GRASPING_FRAME
            ),
        )


@dataclass(eq=False)
class TiagoMujocoLeftArm(Arm[TiagoMujocoLeftGripper], ServoedByItsMujocoModel):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
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
        return [arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, TiagoMujocoBody.TORSO_LIFT
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, TiagoMujocoBody.LEFT_WRIST
            ),
        )


@dataclass(eq=False)
class TiagoMujocoRightArm(Arm[TiagoMujocoRightGripper], ServoedByItsMujocoModel):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
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
        return [arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, TiagoMujocoBody.TORSO_LIFT
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, TiagoMujocoBody.RIGHT_WRIST
            ),
        )


@dataclass(eq=False)
class TiagoMujocoCamera(Camera):
    """
    Tiago's head camera, rooted in the head, since the MuJoCo model has no optical
    frame: the field of view is the camera's, while its origin sits a few centimetres
    behind and below the lens.
    """

    @property
    def forward_facing_axis(self) -> Vector3:
        return Vector3.X(reference_frame=self.root)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, TiagoMujocoBody.HEAD
            ),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=1.0665,
            maximal_height=1.4165,
            default_camera=True,
        )

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []


@dataclass(eq=False)
class TiagoMujocoNeck(Neck[TiagoMujocoCamera], ServoedByItsMujocoModel):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, TiagoMujocoBody.TORSO_LIFT
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, TiagoMujocoBody.HEAD
            ),
        )


@dataclass(eq=False)
class TiagoMujocoTorso(
    Torso,
    HasLeftRightArm[TiagoMujocoLeftArm, TiagoMujocoRightArm],
    HasNeck[TiagoMujocoNeck],
    ServoedByItsMujocoModel,
):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
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
        return [torso_low, torso_mid, torso_high]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, TiagoMujocoBody.BASE
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, TiagoMujocoBody.TORSO_LIFT
            ),
        )


@dataclass(eq=False)
class TiagoMujocoMobileBase(MobileBase[DifferentialDrive], HasTorso[TiagoMujocoTorso]):

    @classproperty
    def forward_axis(cls) -> Vector3:
        return Vector3.X()

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @property
    def drive_servos(self) -> DriveServos:
        """
        Servos tuned for the base, which carries the whole robot, to follow its
        commanded velocity; they are not taken from the real robot.
        """
        return DriveServos(
            translation_gains=VelocityServoGains(damping=20000.0, torque_limit=10000.0),
            rotation_gains=VelocityServoGains(damping=5000.0, torque_limit=5000.0),
        )

    def _setup_servos(self) -> None:
        """
        Declare the drive's servos, see :meth:`MobileBase._setup_servos`, and let a
        physical simulation carry the weight of every body standing on the base.
        """
        super()._setup_servos()
        self._compensate_gravity()

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, TiagoMujocoBody.BASE
            ),
            full_body_controlled=False,
        )


@dataclass(eq=False)
class TiagoMujoco(AbstractRobot, HasMobileBase[TiagoMujocoMobileBase]):
    """
    Class that describes the Take It And Go Robot (TIAGo).

    Described by MuJoCo's own model of the robot, whose gripper is two sliding fingers,
    rather than by the URDF, whose gripper is a linkage. The model carries fewer bodies
    than the URDF, the camera's optical frame among those it leaves out.
    """

    @classmethod
    def get_ros_file_path(cls) -> str:
        return "package://iai_tiago_description/mjcf/tiago_dual.xml"

    @classmethod
    def from_description(cls) -> Self:
        """
        :return: This robot, read from its MJCF without the world body MJCF puts above
            every robot's root.
        """
        world = MJCFParser(cls.get_ros_file_path()).parse()
        with world.modify_world():
            world.remove_kinematic_structure_entity(world.root)
        return cls.from_world(world)

    @classmethod
    def _get_root_body_name(cls) -> str:
        return TiagoMujocoBody.BASE

    def _setup_collision_rules(self):
        """
        Avoid collisions with everything but the robot itself, whose description states
        which of its own bodies may touch.
        """
        self._world.collision_manager.extend_default_rules(
            [
                AvoidExternalCollisions(
                    buffer_zone_distance=0.05, violated_distance=0.0, robot=self
                ),
            ]
        )

    def _setup_velocity_limits(self):
        self.tighten_dof_velocity_limits_of_1dof_connections(
            new_limits=defaultdict(lambda: 1.0)
        )
