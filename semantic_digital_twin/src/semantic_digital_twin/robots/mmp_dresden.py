from __future__ import annotations

from abc import ABC

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Self, List

from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import (
    HasOneArm,
    HasTorso,
    HasMobileBase,
    HasTwoFingers,
    HasEndEffector,
    HasSensors,
)
from semantic_digital_twin.robots.robot_parts import (
    AbstractRobot,
    Arm,
    Camera,
    Finger,
    Torso,
    MobileBase,
    EndEffector,
)
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@dataclass(eq=False)
class MMPDresdenThumb(Finger):

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
                robot_root, "arm_0_gripper_robotiq_85_left_knuckle_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_0_gripper_robotiq_85_left_finger_tip_link"
            ),
        )


@dataclass(eq=False)
class MMPDresdenIndexFinger(Finger):

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

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
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

        return [gripper_open, gripper_close]

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
            front_facing_orientation=Quaternion(0.5, 0.5, 0.5, 0.5),
        )


@dataclass(eq=False)
class MMPDresdenCamera(Camera):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "pan_and_tilt_camera_link"
            ),
            forward_facing_axis=Vector3.Z(),
            field_of_view=FieldOfView(horizontal_angle=1.047, vertical_angle=0.785),
            minimal_height=0.8,
            maximal_height=1.7,
            default_camera=True,
        )

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []


@dataclass(eq=False)
class MMPDresdenArm(Arm[MMPDresdenGripper]):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
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

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
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
class MMPDresdenMobileBase(MobileBase, HasTorso[MMPDresdenTorso]):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "base_link"),
        )

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []


@dataclass(eq=False)
class MMPDresden(AbstractRobot, HasMobileBase[MMPDresdenMobileBase]):
    """
    The  Mobile Manipulation Platform (MMP) Dresden version. http://www.rp.mybotshop.de/projects/robot_mmp_ridgeback/html/mmp_dre.html
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
