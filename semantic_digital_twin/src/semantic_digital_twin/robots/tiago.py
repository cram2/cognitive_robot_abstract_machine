from __future__ import annotations

import os
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Self, List

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
    HasLeftRightArm,
    HasMobileBase,
    HasNeck,
    HasTorso,
    HasTwoFingers,
    TGenericLeftFinger,
    TGenericRightFinger,
    HasEndEffector,
    HasSensors,
)
from semantic_digital_twin.robots.robot_parts import (
    AbstractRobot,
    Arm,
    Camera,
    Finger,
    MobileBase,
    Neck,
    Torso,
    EndEffector,
)
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.world_description.connections import ActiveConnection
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


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
            "gripper_left_finger_joint",
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
            front_facing_orientation=Quaternion(0, 0, 0, 1),
        )


@dataclass(eq=False)
class TiagoRightGripper(
    EndEffector, HasTwoFingers[TiagoRightThumb, TiagoRightIndexFinger]
):

    def setup_hardware_interfaces(self):
        controlled_joints = [
            "gripper_right_finger_joint",
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
            front_facing_orientation=Quaternion(0, 0, 0, 1),
        )


@dataclass(eq=False)
class TiagoLeftArm(Arm[TiagoLeftGripper]):

    def setup_hardware_interfaces(self):
        controlled_joints = [
            "arm_left_1_joint",
            "arm_left_2_joint",
            "arm_left_3_joint",
            "arm_left_4_joint",
            "arm_left_5_joint",
            "arm_left_6_joint",
            "arm_left_7_joint",
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
                    [1.57, 0.33, 1.57, 1.57, 0.0, 1.1, 0.0],
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
            "arm_right_1_joint",
            "arm_right_2_joint",
            "arm_right_3_joint",
            "arm_right_4_joint",
            "arm_right_5_joint",
            "arm_right_6_joint",
            "arm_right_7_joint",
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
                    [1.57, 0.33, 1.57, 1.57, 0.0, 1.1, 0.0],
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

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "head_front_camera_optical_frame"
            ),
            forward_facing_axis=Vector3.Z(),
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
            "head_1_joint",
            "head_2_joint",
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
            "torso_lift_joint",
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
class TiagoMobileBase(MobileBase, HasTorso[TiagoTorso]):

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
            forward_axis=Vector3.X(),
        )


@dataclass(eq=False)
class Tiago(AbstractRobot, HasMobileBase[TiagoMobileBase]):
    """
    The Tiago++ robot by PAL Robotics with updated Robotiq grippers. https://pal-robotics.com/blog/tiago-bi-manual-robot-research/
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
                robot_root, "arm_left_7_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_left_left_finger_link"
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
                robot_root, "arm_left_7_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_left_right_finger_link"
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
                robot_root, "arm_right_7_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_right_left_finger_link"
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
                robot_root, "arm_right_7_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "gripper_right_right_finger_link"
            ),
        )


@dataclass(eq=False)
class TiagoMujocoLeftGripper(
    EndEffector, HasTwoFingers[TiagoMujocoLeftThumb, TiagoMujocoLeftIndexFinger]
):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
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
        return [gripper_close, gripper_open]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_left_7_link"
            ),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_left_7_link"
            ),
            front_facing_orientation=Quaternion(0, 0, 0, 1),
        )


@dataclass(eq=False)
class TiagoMujocoRightGripper(
    EndEffector, HasTwoFingers[TiagoMujocoRightThumb, TiagoMujocoRightIndexFinger]
):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
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
        return [gripper_close, gripper_open]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_right_7_link"
            ),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_right_7_link"
            ),
            front_facing_orientation=Quaternion(0, 0, 0, 1),
        )


@dataclass(eq=False)
class TiagoMujocoLeftArm(Arm[TiagoMujocoLeftGripper]):

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
                robot_root, "torso_lift_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_left_7_link"
            ),
        )


@dataclass(eq=False)
class TiagoMujocoRightArm(Arm[TiagoMujocoRightGripper]):

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
                robot_root, "torso_lift_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "arm_right_7_link"
            ),
        )


@dataclass(eq=False)
class TiagoMujocoNeck(Neck[TiagoCamera]):

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
                robot_root, "torso_lift_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(robot_root, "head_2_link"),
        )


@dataclass(eq=False)
class TiagoMujocoTorso(
    Torso,
    HasLeftRightArm[TiagoMujocoLeftArm, TiagoMujocoRightArm],
    HasNeck[TiagoMujocoNeck],
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
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "base_link"),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "torso_lift_link"
            ),
        )


@dataclass(eq=False)
class TiagoMujocoMobileBase(MobileBase, HasTorso[TiagoMujocoTorso]):

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
            forward_axis=Vector3.X(),
            full_body_controlled=False,
        )


@dataclass(eq=False)
class TiagoMujoco(AbstractRobot, HasMobileBase[TiagoMujocoMobileBase]):
    """
    Class that describes the Take It And Go Robot (TIAGo). This version is based on the MuJoCo model, which contains
    less bodies and connections than the URDF version, including missing some crucial links like the camera etc.
    """

    @classmethod
    def get_ros_file_path(cls) -> str:
        raise NotImplementedError(f"Filepath unknown, please update")

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "base_link"

    def _setup_collision_rules(self):
        pass

    def _setup_velocity_limits(self):
        pass
