from __future__ import annotations

# %% Imports
import os
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from importlib.resources import files
from pathlib import Path
from typing_extensions import Generic, Self


from krrood.ormatic.utils import classproperty
from semantic_digital_twin.adapters.sensors.lidar import Lidar, LidarSource
from semantic_digital_twin.collision_checking.collision_matrix import (
    MaxAvoidedCollisionsOverride,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    SelfCollisionMatrixRule,
    AvoidExternalCollisions,
    AvoidSelfCollisions,
)
from semantic_digital_twin.datastructures.definitions import (
    JointName,
    StaticJointState,
    GripperState,
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
    HasLeftRightArm,
    HasTorso,
    HasMobileBase,
    HasTwoFingers,
    TGenericEndEffector,
    TGenericLeftFinger,
    TGenericRightFinger,
)
from semantic_digital_twin.robots.robot_parts import (
    ServoedRobotPart,
    MobileBase,
    Torso,
    Arm,
    Neck,
    Finger,
    RobotCamera,
    AbstractRobot,
    EndEffector,
)
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world_description.connection_properties import (
    DriveServos,
    JointDrive,
    ServoGains,
    VelocityServoGains,
)
from semantic_digital_twin.world_description.connections import (
    ActiveConnection,
    OmniDrive,
)
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% Robot parts


class PR2Topic(StrEnum):
    """
    Topics the PR2 publishes the state of its parts on, where it does not follow the
    conventional name.
    """

    ODOMETRY = "odom"
    """
    The topic the mobile base reports its odometry on.
    """


class PR2Joint(JointName):
    """
    Names of the PR2's commandable connections, as spelled in its URDF.

    Members are usable wherever a connection name is expected, so a configuration keyed by
    them stays a plain mapping of names to positions.

    ..note:: Connections that no controller commands, such as the caster wheels and the
        grippers' screw, slider and finger tip joints, are left out.
    """

    TORSO_LIFT = "torso_lift_joint"
    HEAD_PAN = "head_pan_joint"
    HEAD_TILT = "head_tilt_joint"

    LEFT_SHOULDER_PAN = "l_shoulder_pan_joint"
    LEFT_SHOULDER_LIFT = "l_shoulder_lift_joint"
    LEFT_UPPER_ARM_ROLL = "l_upper_arm_roll_joint"
    LEFT_ELBOW_FLEX = "l_elbow_flex_joint"
    LEFT_FOREARM_ROLL = "l_forearm_roll_joint"
    LEFT_WRIST_FLEX = "l_wrist_flex_joint"
    LEFT_WRIST_ROLL = "l_wrist_roll_joint"
    LEFT_GRIPPER_LEFT_FINGER = "l_gripper_l_finger_joint"
    LEFT_GRIPPER_RIGHT_FINGER = "l_gripper_r_finger_joint"

    RIGHT_SHOULDER_PAN = "r_shoulder_pan_joint"
    RIGHT_SHOULDER_LIFT = "r_shoulder_lift_joint"
    RIGHT_UPPER_ARM_ROLL = "r_upper_arm_roll_joint"
    RIGHT_ELBOW_FLEX = "r_elbow_flex_joint"
    RIGHT_FOREARM_ROLL = "r_forearm_roll_joint"
    RIGHT_WRIST_FLEX = "r_wrist_flex_joint"
    RIGHT_WRIST_ROLL = "r_wrist_roll_joint"
    RIGHT_GRIPPER_LEFT_FINGER = "r_gripper_l_finger_joint"
    RIGHT_GRIPPER_RIGHT_FINGER = "r_gripper_r_finger_joint"


@dataclass(eq=False)
class PR2KinectV1(RobotCamera[FieldOfViewCameraModel]):
    """
    Kinect V1 camera attached to its RGB optical frame.
    """

    def setup_hardware_interfaces(self) -> None:
        pass

    def setup_joint_states(self) -> list[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        """
        Attach the camera to the Kinect RGB optical frame below the robot root.

        :param robot_root: Root of the robot branch containing the optical frame.
        :return: Camera configured for the model's optical frame.
        """
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "head_mount_kinect_rgb_optical_frame"
            ),
            camera_model=FieldOfViewCameraModel(
                angular_extent=FieldOfView(
                    horizontal_angle=0.99483, vertical_angle=0.75049
                )
            ),
            minimal_height=1.27,
            maximal_height=1.60,
            default_camera=True,
        )


@dataclass(eq=False)
class PR2BaseLidar(Lidar):
    """
    The Hokuyo scanner sweeping the floor in front of the PR2's base.
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
                minimum_angle=-2.2689,
                maximum_angle=2.2689,
                angle_increment=0.0043633,
                minimum_range=0.023,
                maximum_range=60.0,
            ),
            source=source,
        )


@dataclass(eq=False)
class PR2RightGripperLeftFinger(Finger):
    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> list[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "r_gripper_l_finger_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "r_gripper_l_finger_tip_link"
            ),
        )


@dataclass(eq=False)
class PR2RightGripperRightFinger(Finger):
    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> list[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "r_gripper_r_finger_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "r_gripper_r_finger_tip_link"
            ),
        )


@dataclass(eq=False)
class PR2LeftGripperLeftFinger(Finger):
    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> list[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "l_gripper_l_finger_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "l_gripper_l_finger_tip_link"
            ),
        )


@dataclass(eq=False)
class PR2LeftGripperRightFinger(Finger):
    def setup_hardware_interfaces(self):
        return

    def setup_joint_states(self) -> list[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "l_gripper_r_finger_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "l_gripper_r_finger_tip_link"
            ),
        )


@dataclass(eq=False)
class PR2Gripper(
    EndEffector,
    HasTwoFingers[TGenericLeftFinger, TGenericRightFinger],
    ServoedRobotPart,
    Generic[TGenericLeftFinger, TGenericRightFinger],
    ABC,
):
    """
    A gripper of the PR2: two fingers coupled by a mimic linkage, driven through the
    joint of the thumb.
    """

    @property
    def drive_of_each_joint(self) -> dict[str, JointDrive]:
        """
        The joint of the thumb, and with it every finger joint that follows it.
        """
        thumb_degree_of_freedom = self.thumb.root.parent_connection.raw_dof
        thumb_drive = JointDrive(
            gains=ServoGains(stiffness=50.0, damping=5.0, torque_limit=5.0),
            armature=0.05,
        )
        return {
            connection.name.name: thumb_drive
            for connection in self.active_connections
            if connection.raw_dof is thumb_degree_of_freedom
        }


@dataclass(eq=False)
class PR2RightGripper(
    PR2Gripper[PR2RightGripperLeftFinger, PR2RightGripperRightFinger]
):
    def setup_joint_states(self) -> list[JointState]:
        right_gripper_joints = self.active_connections

        right_gripper_open = JointState.from_mapping(
            name=PrefixedName("right_gripper_open", prefix=self.name.name),
            mapping=dict(zip(right_gripper_joints, [0.548, 0.548])),
            state_type=GripperState.OPEN,
        )

        right_gripper_close = JointState.from_mapping(
            name=PrefixedName("right_gripper_close", prefix=self.name.name),
            mapping=dict(zip(right_gripper_joints, [0.0, 0.0])),
            state_type=GripperState.CLOSE,
        )

        return [right_gripper_open, right_gripper_close]

    def setup_hardware_interfaces(self):
        return

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
                robot_root, "r_gripper_palm_link"
            ),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "r_gripper_tool_frame"
            ),
        )


@dataclass(eq=False)
class PR2LeftGripper(PR2Gripper[PR2LeftGripperLeftFinger, PR2LeftGripperRightFinger]):
    def setup_joint_states(self) -> list[JointState]:
        left_gripper_joints = self.active_connections
        left_gripper_open = JointState.from_mapping(
            name=PrefixedName("left_gripper_open", prefix=self.name.name),
            mapping=dict(zip(left_gripper_joints, [0.548, 0.548])),
            state_type=GripperState.OPEN,
        )

        left_gripper_close = JointState.from_mapping(
            name=PrefixedName("left_gripper_close", prefix=self.name.name),
            mapping=dict(zip(left_gripper_joints, [0.0, 0.0])),
            state_type=GripperState.CLOSE,
        )

        return [left_gripper_open, left_gripper_close]

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
                robot_root, "l_gripper_palm_link"
            ),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "l_gripper_tool_frame"
            ),
        )

    def setup_hardware_interfaces(self):
        return


@dataclass(eq=False)
class PR2Neck(Neck[PR2KinectV1], ServoedRobotPart):
    def setup_hardware_interfaces(self):

        controlled_joints = [
            PR2Joint.HEAD_PAN,
            PR2Joint.HEAD_TILT,
        ]
        for joint_name in controlled_joints:
            connection: ActiveConnection = self._world.get_connection_by_name(
                joint_name
            )
            connection.has_hardware_interface = True

    def setup_joint_states(self) -> list[JointState]:
        return []

    @property
    def drive_of_each_joint(self) -> dict[str, JointDrive]:
        return {
            PR2Joint.HEAD_PAN: JointDrive(
                gains=ServoGains(stiffness=200.0, damping=60.0, torque_limit=20.0),
                armature=0.5,
            ),
            PR2Joint.HEAD_TILT: JointDrive(
                gains=ServoGains(stiffness=400.0, damping=40.0, torque_limit=40.0),
                armature=0.5,
            ),
        }

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "torso_lift_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "head_tilt_link"
            ),
        )


@dataclass(eq=False)
class PR2Arm(
    Arm[TGenericEndEffector], ServoedRobotPart, Generic[TGenericEndEffector], ABC
):
    """
    An arm of the PR2.
    """

    @property
    def drive_of_each_joint(self) -> dict[str, JointDrive]:
        """
        The shoulder, upper arm, elbow and forearm joints may exert more torque than the
        two wrist joints, which carry only the gripper.
        """
        upper = JointDrive(
            gains=ServoGains(stiffness=2000.0, damping=200.0, torque_limit=100.0),
            armature=0.5,
        )
        wrist = JointDrive(
            gains=ServoGains(stiffness=500.0, damping=30.0, torque_limit=30.0),
            armature=0.1,
        )
        drive_by_unprefixed_name = {
            "shoulder_pan_joint": upper,
            "shoulder_lift_joint": upper,
            "upper_arm_roll_joint": upper,
            "elbow_flex_joint": upper,
            "forearm_roll_joint": upper,
            "wrist_flex_joint": wrist,
            "wrist_roll_joint": wrist,
        }
        return {
            connection.name.name: drive_by_unprefixed_name[
                connection.name.name.removeprefix("l_").removeprefix("r_")
            ]
            for connection in self.active_connections
        }


@dataclass(eq=False)
class PR2LeftArm(PR2Arm[PR2LeftGripper]):
    def setup_hardware_interfaces(self):
        controlled_joints = [
            PR2Joint.LEFT_SHOULDER_PAN,
            PR2Joint.LEFT_SHOULDER_LIFT,
            PR2Joint.LEFT_UPPER_ARM_ROLL,
            PR2Joint.LEFT_FOREARM_ROLL,
            PR2Joint.LEFT_ELBOW_FLEX,
            PR2Joint.LEFT_WRIST_FLEX,
            PR2Joint.LEFT_WRIST_ROLL,
        ]
        for joint_name in controlled_joints:
            connection: ActiveConnection = self._world.get_connection_by_name(
                joint_name
            )
            connection.has_hardware_interface = True

    def setup_joint_states(self) -> list[JointState]:
        left_arm_park = JointState.from_mapping(
            name=PrefixedName("left_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    [c for c in self.active_connections],
                    [
                        1.712,
                        -0.264,
                        1.38,
                        -2.12,
                        4.2,
                        -0.073,
                        0.0,
                    ],
                )
            ),
            state_type=StaticJointState.PARK,
        )

        return [left_arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "torso_lift_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "l_force_torque_link"
            ),
        )


@dataclass(eq=False)
class PR2RightArm(PR2Arm[PR2RightGripper]):
    def setup_hardware_interfaces(self):
        controlled_joints = [
            PR2Joint.RIGHT_SHOULDER_PAN,
            PR2Joint.RIGHT_SHOULDER_LIFT,
            PR2Joint.RIGHT_UPPER_ARM_ROLL,
            PR2Joint.RIGHT_FOREARM_ROLL,
            PR2Joint.RIGHT_ELBOW_FLEX,
            PR2Joint.RIGHT_WRIST_FLEX,
            PR2Joint.RIGHT_WRIST_ROLL,
        ]
        for joint_name in controlled_joints:
            connection: ActiveConnection = self._world.get_connection_by_name(
                joint_name
            )
            connection.has_hardware_interface = True

    def setup_joint_states(self) -> list[JointState]:
        right_arm_park = JointState.from_mapping(
            name=PrefixedName("right_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    [c for c in self.active_connections],
                    [-1.712, -0.256, -1.463, -2.12, 1.766, -0.07, 0.051],
                )
            ),
            state_type=StaticJointState.PARK,
        )

        return [right_arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "torso_lift_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "r_wrist_roll_link"
            ),
        )


@dataclass(eq=False)
class PR2Torso(
    Torso,
    HasLeftRightArm[PR2LeftArm, PR2RightArm],
    HasNeck[PR2Neck],
    ServoedRobotPart,
):
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

    def setup_hardware_interfaces(self):
        controlled_joints = [
            PR2Joint.TORSO_LIFT,
        ]
        for joint_name in controlled_joints:
            connection: ActiveConnection = self._world.get_connection_by_name(
                joint_name
            )
            connection.has_hardware_interface = True

    def setup_joint_states(self) -> list[JointState]:
        torso_joint = self.active_connections
        torso_low = JointState.from_mapping(
            name=PrefixedName("torso_low", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.0115])),
            state_type=TorsoState.LOW,
        )

        torso_mid = JointState.from_mapping(
            name=PrefixedName("torso_mid", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.15])),
            state_type=TorsoState.MID,
        )

        torso_high = JointState.from_mapping(
            name=PrefixedName("torso_high", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.3])),
            state_type=TorsoState.HIGH,
        )

        return [torso_low, torso_mid, torso_high]

    @property
    def drive_of_each_joint(self) -> dict[str, JointDrive]:
        """
        The torso lift joint, which carries the whole upper body.
        """
        return {
            PR2Joint.TORSO_LIFT: JointDrive(
                gains=ServoGains(
                    stiffness=100000.0, damping=8000.0, torque_limit=10000.0
                ),
                armature=1.0,
            )
        }


@dataclass(eq=False)
class PR2MobileBase(MobileBase[OmniDrive], HasTorso[PR2Torso], HasLidar[PR2BaseLidar]):
    @classproperty
    def topic_name(cls) -> str:
        """
        The topic the PR2 publishes the pose of its base on.
        """
        return PR2Topic.ODOMETRY

    @classproperty
    def forward_axis(cls) -> Vector3:
        return Vector3.X()

    def setup_hardware_interfaces(self):
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
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "base_link"),
        )


@dataclass(eq=False)
class PR2(AbstractRobot, HasMobileBase[PR2MobileBase]):
    """
    The PR2 robot built by Willow Garage.

    https://robotsguide.com/robots/pr2

    .. note:: Its joint drives are not taken from the real robot; they are tuned for the
        simulated joints to follow their commanded positions.
    """

    @classmethod
    def get_ros_file_path(cls) -> str:
        return "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro"

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "base_footprint"

    def _setup_velocity_limits(self):
        vel_limits = defaultdict(
            lambda: 1.0,
            {
                self._world.get_connection_by_name(PR2Joint.HEAD_TILT): 3.5,
                self._world.get_connection_by_name(PR2Joint.RIGHT_SHOULDER_PAN): 0.15,
                self._world.get_connection_by_name(PR2Joint.LEFT_SHOULDER_PAN): 0.15,
                self._world.get_connection_by_name(PR2Joint.LEFT_SHOULDER_LIFT): 0.2,
                self._world.get_connection_by_name(PR2Joint.RIGHT_SHOULDER_LIFT): 0.2,
            },
        )
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)

    def _setup_collision_rules(self):
        """
        Loads the SRDF file for the PR2 robot, if it exists.
        """
        srdf_path = os.path.join(
            Path(files("semantic_digital_twin")).parent.parent,
            "resources",
            "collision_configs",
            "pr2.srdf",
        )
        self._world.collision_manager.add_ignore_collision_rule(
            SelfCollisionMatrixRule.from_collision_srdf(srdf_path, self._world)
        )

        self._world.collision_manager.extend_default_rules(
            [
                AvoidExternalCollisions(
                    buffer_zone_distance=0.1, violated_distance=0.0, robot=self
                ),
                AvoidExternalCollisions(
                    buffer_zone_distance=0.05,
                    violated_distance=0.0,
                    robot=self,
                    body_subset=(
                        set(self.mobile_base.torso.left_arm.bodies_with_collision)
                        - set(
                            self.mobile_base.torso.left_arm.end_effector.bodies_with_collision
                        )
                    )
                    | (
                        set(self.mobile_base.torso.right_arm.bodies_with_collision)
                        - set(
                            self.mobile_base.torso.right_arm.end_effector.bodies_with_collision
                        )
                    ),
                ),
                AvoidExternalCollisions(
                    buffer_zone_distance=0.15,
                    violated_distance=0.05,
                    robot=self,
                    body_subset={
                        self._world.get_body_in_branch_by_name(self.root, "base_link")
                    },
                ),
                AvoidSelfCollisions(
                    buffer_zone_distance=0.05, violated_distance=0.0, robot=self
                ),
            ]
        )

        self._world.collision_manager.extend_max_avoided_bodies_rules(
            [
                MaxAvoidedCollisionsOverride(
                    2,
                    bodies={
                        self._world.get_body_in_branch_by_name(self.root, "base_link")
                    },
                ),
                MaxAvoidedCollisionsOverride(
                    4,
                    bodies=set(
                        self._world.get_direct_child_bodies_with_collision(
                            self._world.get_body_in_branch_by_name(
                                self.root, "r_wrist_roll_link"
                            )
                        )
                    )
                    | set(
                        self._world.get_direct_child_bodies_with_collision(
                            self._world.get_body_in_branch_by_name(
                                self.root, "l_wrist_roll_link"
                            )
                        )
                    ),
                ),
            ]
        )

    @property
    def left_arm(self) -> PR2LeftArm:
        return self.torso.left_arm

    @property
    def right_arm(self) -> PR2RightArm:
        return self.torso.right_arm

    @property
    def torso(self) -> PR2Torso:
        return self.mobile_base.torso
