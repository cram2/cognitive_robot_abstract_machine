from __future__ import annotations

# %% Imports
from collections import defaultdict
from dataclasses import dataclass
from typing_extensions import Self

import numpy as np

from krrood.ormatic.utils import classproperty
from semantic_digital_twin.datastructures.definitions import (
    JointName,
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import (
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
    Torso,
    MobileBase,
    EndEffector,
)
from semantic_digital_twin.datastructures.camera_model import FieldOfViewCameraModel
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    OmniDrive,
)
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% Robot parts


class MMPDresdenJoint(JointName):
    """
    Names of the MMPDresden's commandable connections, as spelled in its URDF.

    Members are usable wherever a connection name is expected, so a configuration keyed by
    them stays a plain mapping of names to positions.

    ..note:: Connections that no controller commands, such as the base wheels and the
        gripper's coupled knuckle and finger tip joints, are left out.
    """

    SHOULDER_PAN = "arm_0_shoulder_pan_joint"
    SHOULDER_LIFT = "arm_0_shoulder_lift_joint"
    ELBOW = "arm_0_elbow_joint"
    WRIST_1 = "arm_0_wrist_1_joint"
    WRIST_2 = "arm_0_wrist_2_joint"
    WRIST_3 = "arm_0_wrist_3_joint"
    GRIPPER_LEFT_KNUCKLE = "arm_0_gripper_robotiq_85_left_knuckle_joint"


@dataclass(eq=False)
class MMPDresdenThumb(Finger):
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
                robot_root, "arm_0_gripper_robotiq_85_left_knuckle_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_0_gripper_robotiq_85_left_finger_tip_link"
            ),
        )


@dataclass(eq=False)
class MMPDresdenIndexFinger(Finger):
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
                robot_root, "arm_0_gripper_robotiq_85_right_knuckle_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_0_gripper_robotiq_85_right_finger_tip_link"
            ),
        )


@dataclass(eq=False)
class MMPDresdenGripper(
    EndEffector, HasTwoFingers[MMPDresdenThumb, MMPDresdenIndexFinger]
):
    def setup_hardware_interfaces(self) -> None:
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> list[JointState]:
        """
        The gripper stands open with its left knuckle, which every other joint of the
        gripper mimics, at its lower limit and closed with it at its upper limit.
        """
        knuckle: ActiveConnection1DOF = self._world.get_connection_by_name(
            MMPDresdenJoint.GRIPPER_LEFT_KNUCKLE
        )
        limits = knuckle.raw_dof.limits

        gripper_open = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_open", prefix=self.name.name),
            mapping={knuckle: limits.lower.position},
            state_type=GripperState.OPEN,
        )

        gripper_close = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_close", prefix=self.name.name),
            mapping={knuckle: limits.upper.position},
            state_type=GripperState.CLOSE,
        )

        return [gripper_open, gripper_close]

    @property
    def approach_axis(self) -> Vector3:
        return Vector3.Z(reference_frame=self.tool_frame)

    @property
    def closing_axis(self) -> Vector3:
        return Vector3.X(reference_frame=self.tool_frame)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_0_flange"
            ),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_0_tool_frame"
            ),
        )


@dataclass(eq=False)
class MMPDresdenCamera(RobotCamera[FieldOfViewCameraModel]):
    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "pan_and_tilt_camera_color_optical_frame"
            ),
            camera_model=FieldOfViewCameraModel(
                angular_extent=FieldOfView(horizontal_angle=1.047, vertical_angle=0.785)
            ),
            minimal_height=0.8,
            maximal_height=1.7,
            default_camera=True,
        )

    def setup_hardware_interfaces(self) -> None:
        pass

    def setup_joint_states(self) -> list[JointState]:
        return []


@dataclass(eq=False)
class MMPDresdenArm(Arm[MMPDresdenGripper]):
    def setup_hardware_interfaces(self) -> None:
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> list[JointState]:
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
        return [arm_park, arm_both]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "hub_holder_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_0_flange"
            ),
        )


@dataclass(eq=False)
class MMPDresdenTorso(Torso, HasOneArm[MMPDresdenArm], HasSensors[MMPDresdenCamera]):
    def setup_hardware_interfaces(self) -> None:
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> list[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "base_link"),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "hub_holder_link"
            ),
        )


@dataclass(eq=False)
class MMPDresdenMobileBase(MobileBase[OmniDrive], HasTorso[MMPDresdenTorso]):
    @classproperty
    def forward_axis(cls) -> Vector3:
        return Vector3.X()

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "base_link"),
        )

    def setup_hardware_interfaces(self) -> None:
        pass

    def setup_joint_states(self) -> list[JointState]:
        return []


@dataclass(eq=False)
class MMPDresden(AbstractRobot, HasMobileBase[MMPDresdenMobileBase]):
    """
    The  Mobile Manipulation Platform (MMP) Dresden version.

    http://www.rp.mybotshop.de/projects/robot_mmp_ridgeback/html/mmp_dre.html
    """

    def _setup_collision_rules(self):
        pass

    @classmethod
    def get_ros_file_path(cls) -> str:
        return "package://iai_smart_mobility/urdf/mmp_dresden.urdf"

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "base_link"

    def _setup_velocity_limits(self):
        vel_limits = defaultdict(lambda: 1.0)
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)
