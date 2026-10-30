from __future__ import annotations

# %% Imports
import os
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from importlib.resources import files
from pathlib import Path
from typing_extensions import Self

from krrood.ormatic.utils import classproperty
from semantic_digital_twin.adapters.sensors.lidar import Lidar, LidarSource
from semantic_digital_twin.collision_checking.collision_matrix import (
    MaxAvoidedCollisionsOverride,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidExternalCollisions,
    AvoidSelfCollisions,
    SelfCollisionMatrixRule,
)
from semantic_digital_twin.datastructures.definitions import (
    JointName,
    GripperState,
    StaticJointState,
    TorsoState,
)
from semantic_digital_twin.datastructures.camera_model import FieldOfViewCameraModel
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.scan_pattern import ScanPattern
from semantic_digital_twin.robots.robot_part_mixins import (
    HasNeck,
    HasLidar,
    HasOneArm,
    HasTorso,
    HasMobileBase,
    HasTwoFingers,
    HasSensors,
)
from semantic_digital_twin.robots.robot_parts import (
    AbstractRobot,
    Arm,
    RobotCamera,
    Finger,
    Neck,
    Torso,
    MobileBase,
    EndEffector,
    ServoedRobotPart,
)
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world_description.connection_properties import (
    DriveServos,
    JointDrive,
    ServoGains,
    VelocityServoGains,
)
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    OmniDrive,
)
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% Robot parts


class HSRBTopic(StrEnum):
    """
    Topics the HSRB publishes the state of its parts on, where it does not follow the
    conventional name.
    """

    ODOMETRY = "laser_odom"
    """
    The topic the mobile base reports its odometry on.
    """


class HSRBJoint(JointName):
    """
    Names of the HSRB's commandable connections, as spelled in its URDF.

    Members are usable wherever a connection name is expected, so a configuration keyed by
    them stays a plain mapping of names to positions.

    ..note:: Connections that no controller commands, such as the drive and passive base
        wheels and the hand's distal joints, are left out. The hand's spring joints are
        commanded only in simulation.
    """

    TORSO_LIFT = "torso_lift_joint"
    HEAD_PAN = "head_pan_joint"
    HEAD_TILT = "head_tilt_joint"

    ARM_LIFT = "arm_lift_joint"
    ARM_FLEX = "arm_flex_joint"
    ARM_ROLL = "arm_roll_joint"
    WRIST_FLEX = "wrist_flex_joint"
    WRIST_ROLL = "wrist_roll_joint"

    HAND_MOTOR = "hand_motor_joint"
    HAND_LEFT_PROXIMAL = "hand_l_proximal_joint"
    HAND_RIGHT_PROXIMAL = "hand_r_proximal_joint"
    HAND_LEFT_SPRING_PROXIMAL = "hand_l_spring_proximal_joint"
    HAND_RIGHT_SPRING_PROXIMAL = "hand_r_spring_proximal_joint"


@dataclass(eq=False)
class HSRBLeftFinger(Finger):
    def setup_hardware_interfaces(self) -> None:
        pass

    def setup_joint_states(self) -> list[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "hand_l_proximal_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "hand_l_distal_link"
            ),
        )


@dataclass(eq=False)
class HSRBRightFinger(Finger):
    def setup_hardware_interfaces(self) -> None:
        pass

    def setup_joint_states(self) -> list[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "hand_r_proximal_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "hand_r_distal_link"
            ),
        )


@dataclass(eq=False)
class HSRBGripper(
    EndEffector, HasTwoFingers[HSRBLeftFinger, HSRBRightFinger], ServoedRobotPart
):
    @property
    def drive_of_each_joint(self) -> dict[JointName, JointDrive]:
        """
        The hand's motor, which opens and closes both fingers, and the spring of each
        finger, which holds its fingertip straight until what the hand grips presses it
        back.
        """
        spring = JointDrive(
            gains=ServoGains(stiffness=10.0, damping=1.0, torque_limit=10.0)
        )
        return {
            HSRBJoint.HAND_MOTOR: JointDrive(
                gains=ServoGains(stiffness=50.0, damping=5.0, torque_limit=100.0)
            ),
            HSRBJoint.HAND_LEFT_SPRING_PROXIMAL: spring,
            HSRBJoint.HAND_RIGHT_SPRING_PROXIMAL: spring,
        }

    def setup_hardware_interfaces(self) -> None:
        return

    def setup_joint_states(self) -> list[JointState]:
        """
        The gripper stands open with its motor and both finger springs at their upper
        limits, and closed with its motor at zero and the springs relaxed.
        """
        world = self._world
        motor: ActiveConnection1DOF = world.get_connection_by_name(HSRBJoint.HAND_MOTOR)
        springs: list[ActiveConnection1DOF] = [
            world.get_connection_by_name(HSRBJoint.HAND_LEFT_SPRING_PROXIMAL),
            world.get_connection_by_name(HSRBJoint.HAND_RIGHT_SPRING_PROXIMAL),
        ]

        gripper_open = JointState.from_mapping(
            name=PrefixedName("gripper_open", prefix=self.name.name),
            mapping={
                connection: connection.raw_dof.limits.upper.position
                for connection in [motor, *springs]
            },
            state_type=GripperState.OPEN,
        )

        gripper_close = JointState.from_mapping(
            name=PrefixedName("gripper_close", prefix=self.name.name),
            mapping={
                motor: 0.0,
                **{spring: spring.raw_dof.limits.lower.position for spring in springs},
            },
            state_type=GripperState.CLOSE,
        )

        return [gripper_open, gripper_close]

    @property
    def approach_axis(self) -> Vector3:
        return Vector3.Z(reference_frame=self.tool_frame)

    @property
    def closing_axis(self) -> Vector3:
        return Vector3.NEGATIVE_Y(reference_frame=self.tool_frame)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "hand_palm_link"
            ),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "hand_gripper_tool_frame"
            ),
        )


@dataclass(eq=False)
class HSRBHandCamera(RobotCamera[FieldOfViewCameraModel]):
    def setup_hardware_interfaces(self) -> None:
        pass

    def setup_joint_states(self) -> list[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "hand_camera_frame"
            ),
            camera_model=FieldOfViewCameraModel(
                angular_extent=FieldOfView(
                    horizontal_angle=0.99483, vertical_angle=0.75049
                )
            ),
            minimal_height=0.75049,
            maximal_height=0.99483,
        )


@dataclass(eq=False)
class HSRBArm(Arm[HSRBGripper], HasSensors[HSRBHandCamera], ServoedRobotPart):
    @property
    def drive_of_each_joint(self) -> dict[JointName, JointDrive]:
        wrist = JointDrive(
            gains=ServoGains(stiffness=500.0, damping=50.0, torque_limit=100.0),
            armature=0.1,
        )
        return {
            HSRBJoint.ARM_FLEX: JointDrive(
                gains=ServoGains(stiffness=2000.0, damping=200.0, torque_limit=100.0),
                armature=0.1,
            ),
            HSRBJoint.ARM_ROLL: wrist,
            HSRBJoint.WRIST_FLEX: wrist,
            HSRBJoint.WRIST_ROLL: wrist,
        }

    def setup_hardware_interfaces(self) -> None:
        controlled_joints = [
            HSRBJoint.ARM_FLEX,
            HSRBJoint.ARM_LIFT,
            HSRBJoint.ARM_ROLL,
            HSRBJoint.WRIST_FLEX,
            HSRBJoint.WRIST_ROLL,
        ]
        for joint_name in controlled_joints:
            connection = self._world.get_connection_by_name(joint_name)
            connection.has_hardware_interface = True

    def setup_joint_states(self) -> list[JointState]:
        arm_park = JointState.from_mapping(
            name=PrefixedName("arm_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    self.active_connections,
                    [0.0, 1.5, -1.85, 0.0],
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
                robot_root, "arm_lift_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "hand_palm_link"
            ),
        )


@dataclass(eq=False)
class HSRBHeadCenterCamera(RobotCamera[FieldOfViewCameraModel]):
    def setup_hardware_interfaces(self) -> None:
        pass

    def setup_joint_states(self) -> list[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "head_center_camera_frame"
            ),
            camera_model=FieldOfViewCameraModel(
                angular_extent=FieldOfView(
                    horizontal_angle=0.99483, vertical_angle=0.75049
                )
            ),
            minimal_height=0.75049,
            maximal_height=0.99483,
            default_camera=True,
        )


@dataclass(eq=False)
class HSRBHeadLeftCamera(RobotCamera[FieldOfViewCameraModel]):
    def setup_hardware_interfaces(self) -> None:
        pass

    def setup_joint_states(self) -> list[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "head_l_stereo_camera_link"
            ),
            camera_model=FieldOfViewCameraModel(
                angular_extent=FieldOfView(
                    horizontal_angle=0.99483, vertical_angle=0.75049
                )
            ),
            minimal_height=0.75049,
            maximal_height=0.99483,
        )


@dataclass(eq=False)
class HSRBHeadRightCamera(RobotCamera[FieldOfViewCameraModel]):
    def setup_hardware_interfaces(self) -> None:
        pass

    def setup_joint_states(self) -> list[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "head_r_stereo_camera_link"
            ),
            camera_model=FieldOfViewCameraModel(
                angular_extent=FieldOfView(
                    horizontal_angle=0.99483, vertical_angle=0.75049
                )
            ),
            minimal_height=0.75049,
            maximal_height=0.99483,
        )


@dataclass(eq=False)
class HSRBHeadRGBDCamera(RobotCamera[FieldOfViewCameraModel]):
    def setup_hardware_interfaces(self) -> None:
        pass

    def setup_joint_states(self) -> list[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "head_rgbd_sensor_link"
            ),
            camera_model=FieldOfViewCameraModel(
                angular_extent=FieldOfView(
                    horizontal_angle=0.99483, vertical_angle=0.75049
                )
            ),
            minimal_height=0.75049,
            maximal_height=0.99483,
            default_camera=True,
        )


@dataclass(eq=False)
class HSRBNeck(
    Neck[
        HSRBHeadCenterCamera,
        HSRBHeadLeftCamera,
        HSRBHeadRightCamera,
        HSRBHeadRGBDCamera,
    ],
    ServoedRobotPart,
):
    @property
    def drive_of_each_joint(self) -> dict[JointName, JointDrive]:
        head = JointDrive(
            gains=ServoGains(stiffness=300.0, damping=30.0, torque_limit=100.0),
            armature=0.1,
        )
        return {HSRBJoint.HEAD_PAN: head, HSRBJoint.HEAD_TILT: head}

    def setup_hardware_interfaces(self) -> None:
        controlled_joints = [HSRBJoint.HEAD_PAN, HSRBJoint.HEAD_TILT]
        for joint_name in controlled_joints:
            connection = self._world.get_connection_by_name(joint_name)
            connection.has_hardware_interface = True

    def setup_joint_states(self) -> list[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "head_pan_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "head_tilt_link"
            ),
        )


@dataclass(eq=False)
class HSRBTorso(Torso, HasOneArm[HSRBArm], HasNeck[HSRBNeck], ServoedRobotPart):
    @property
    def drive_of_each_joint(self) -> dict[JointName, JointDrive]:
        """
        The arm lift, which the torso lift follows at half its travel.
        """
        return {
            HSRBJoint.ARM_LIFT: JointDrive(
                gains=ServoGains(stiffness=10000.0, damping=1000.0, torque_limit=300.0),
                armature=0.1,
            )
        }

    def setup_hardware_interfaces(self) -> None:
        return

    def setup_joint_states(self) -> list[JointState]:
        torso_joint = self.active_connections
        torso_low = JointState.from_mapping(
            name=PrefixedName("torso_low", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.0])),
            state_type=TorsoState.LOW,
        )

        torso_mid = JointState.from_mapping(
            name=PrefixedName("torso_mid", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.32 / 2])),
            state_type=TorsoState.MID,
        )

        torso_high = JointState.from_mapping(
            name=PrefixedName("torso_high", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.32])),
            state_type=TorsoState.HIGH,
        )

        return [torso_low, torso_mid, torso_high]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "base_link"),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "torso_lift_link"
            ),
        )


@dataclass(eq=False)
class HSRBBaseLidar(Lidar):
    """
    The Hokuyo scanner sweeping the floor around the HSRB's base.
    """

    @classmethod
    def with_source(
        cls, robot_root: KinematicStructureEntity, source: LidarSource
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "base_range_sensor_link"
            ),
            scan_pattern=ScanPattern(
                minimum_angle=-2.098758,
                maximum_angle=2.098758,
                angle_increment=0.0043633,
                minimum_range=0.012,
                maximum_range=60.0,
            ),
            source=source,
        )


@dataclass(eq=False)
class HSRBMobileBase(
    MobileBase[OmniDrive], HasTorso[HSRBTorso], HasLidar[HSRBBaseLidar]
):
    @classproperty
    def topic_name(cls) -> str:
        """
        The topic the HSRB publishes the pose of its base on.
        """
        return HSRBTopic.ODOMETRY

    @classproperty
    def forward_axis(cls) -> Vector3:
        return Vector3.X()

    full_body_controlled: bool = field(default=True, kw_only=True)

    def setup_hardware_interfaces(self) -> None:
        pass

    def setup_joint_states(self) -> list[JointState]:
        return []

    @property
    def drive_servos(self) -> DriveServos:
        """
        Servos tuned for the base, which carries the whole robot, to follow its
        commanded velocity; they are not taken from the real robot.
        """
        return DriveServos(
            translation_gains=VelocityServoGains(damping=4000.0, torque_limit=2000.0),
            rotation_gains=VelocityServoGains(damping=1000.0, torque_limit=1000.0),
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
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "base_link"),
        )


@dataclass(eq=False)
class HSRB(AbstractRobot, HasMobileBase[HSRBMobileBase]):
    """
    The HSRB Robot built by Toyota.

    https://robotsguide.com/robots/hsr

    .. note:: The torque limits of its joint drives are the effort limits of its
        description; their stiffness, damping and armature are not taken from the real
        robot, they are tuned for the simulated joints to follow their commanded
        positions.
    """

    @classmethod
    def get_ros_file_path(cls) -> str:
        return "package://hsr_description/robots/hsrb4s.urdf.xacro"

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "base_footprint"

    def _setup_collision_rules(self):
        srdf_path = os.path.join(
            Path(files("semantic_digital_twin")).parent.parent,
            "resources",
            "collision_configs",
            "hsrb.srdf",
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
            AvoidExternalCollisions(
                buffer_zone_distance=0.1,
                violated_distance=0.03,
                robot=self,
                body_subset={
                    self._world.get_body_in_branch_by_name(self.root, "base_link")
                },
            )
        )
        self._world.collision_manager.add_default_rule(
            AvoidSelfCollisions(
                buffer_zone_distance=0.03,
                violated_distance=0.0,
                robot=self,
            )
        )

        self._world.collision_manager.max_avoided_bodies_rules.append(
            MaxAvoidedCollisionsOverride(
                2,
                bodies={self._world.get_body_in_branch_by_name(self.root, "base_link")},
            )
        )
        self._world.collision_manager.max_avoided_bodies_rules.append(
            MaxAvoidedCollisionsOverride(
                4,
                bodies=set(
                    self._world.get_direct_child_bodies_with_collision(
                        self._world.get_body_in_branch_by_name(
                            self.root, "wrist_roll_link"
                        )
                    )
                ),
            )
        )

    def _setup_velocity_limits(self):
        vel_limits = defaultdict(lambda: 1.0)
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)

    @property
    def end_effector(self) -> HSRBGripper:
        return self.mobile_base.torso.arm.end_effector
