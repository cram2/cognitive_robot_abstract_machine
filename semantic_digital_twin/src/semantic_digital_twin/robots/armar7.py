from __future__ import annotations

import os
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Self, Union, List

from semantic_digital_twin.collision_checking.collision_rules import (
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
    HasNeck,
    HasTorso,
    HasMobileBase,
    HasFingers,
    TGenericFingerOtherThanThumb,
    HasSensors,
)
from semantic_digital_twin.robots.robot_parts import (
    AbstractRobot,
    Arm,
    Camera,
    Finger,
    Neck,
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
class Armar7LeftThumb(Finger):

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
                robot_root, "Hand L Palm_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "Thumb L Tip_link"
            ),
        )


@dataclass(eq=False)
class Armar7LeftRingFinger(Finger):

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
                robot_root, "Hand L Palm_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "Ring L Tip_link"
            ),
        )


@dataclass(eq=False)
class Armar7LeftPinkyFinger(Finger):

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
                robot_root, "Hand L Palm_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "Pinky L Tip_link"
            ),
        )


@dataclass(eq=False)
class Armar7LeftMiddleFinger(Finger):

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
                robot_root, "Hand L Palm_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "Middle L Tip_link"
            ),
        )


@dataclass(eq=False)
class Armar7LeftIndexFinger(Finger):

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
                robot_root, "Hand L Palm_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "Index L Tip_link"
            ),
        )


@dataclass(eq=False)
class Armar7RightThumb(Finger):

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
                robot_root, "Hand R Palm_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "Thumb R Tip_link"
            ),
        )


@dataclass(eq=False)
class Armar7RightRingFinger(Finger):

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
                robot_root, "Hand R Palm_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "Ring R Tip_link"
            ),
        )


@dataclass(eq=False)
class Armar7RightPinkyFinger(Finger):

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
                robot_root, "Hand R Palm_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "Pinky R Tip_link"
            ),
        )


@dataclass(eq=False)
class Armar7RightMiddleFinger(Finger):

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
                robot_root, "Hand R Palm_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "Middle R Tip_link"
            ),
        )


@dataclass(eq=False)
class Armar7RightIndexFinger(Finger):

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
                robot_root, "Hand R Palm_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "Index R Tip_link"
            ),
        )


@dataclass(eq=False)
class Armar7LeftGripper(
    EndEffector,
    HasFingers[
        Armar7LeftThumb,
        Armar7LeftRingFinger,
        Armar7LeftPinkyFinger,
        Armar7LeftMiddleFinger,
        Armar7LeftIndexFinger,
    ],
):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        gripper_joints = self.active_connections

        gripper_open = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_open", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.0] * len(gripper_joints))),
            state_type=GripperState.OPEN,
        )

        gripper_close = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_close", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [1.57] * len(gripper_joints))),
            state_type=GripperState.CLOSE,
        )

        return [gripper_open, gripper_close]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "ArmL8_Wrist_Hemisphere_B_link"
            ),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "Hand L TCP_link"
            ),
            front_facing_orientation=Quaternion(-0.5, 0.5, -0.5, 0.5),
        )


@dataclass(eq=False)
class Armar7RightGripper(
    EndEffector,
    HasFingers[
        Armar7RightThumb,
        Armar7RightRingFinger,
        Armar7RightPinkyFinger,
        Armar7RightMiddleFinger,
        Armar7RightIndexFinger,
    ],
):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        gripper_joints = self.active_connections

        gripper_open = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_open", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.0] * len(gripper_joints))),
            state_type=GripperState.OPEN,
        )

        gripper_close = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_close", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [1.57] * len(gripper_joints))),
            state_type=GripperState.CLOSE,
        )

        return [gripper_open, gripper_close]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "ArmR8_Wrist_Hemisphere_B_link"
            ),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "Hand R TCP_link"
            ),
            front_facing_orientation=Quaternion(-0.5, 0.5, -0.5, 0.5),
        )


@dataclass(eq=False)
class Armar7LeftArm(Arm[Armar7LeftGripper]):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        vals = [0.0, 0.0, 0.25, 0.5, 1.0, 1.0, 0.0, 0.0]
        arm_park = JointState.from_mapping(
            name=PrefixedName("left_arm_park", prefix=self.name.name),
            mapping=dict(zip(self.active_connections, vals)),
            state_type=StaticJointState.PARK,
        )
        return [arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "CenterArms_fixed_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "ArmL8_Wrist_Hemisphere_B_link"
            ),
        )


@dataclass(eq=False)
class Armar7RightArm(Arm[Armar7RightGripper]):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        vals = [0.0, 0.0, 0.25, -0.5, 1.0, -1.0, 0.0, 0.0]
        arm_park = JointState.from_mapping(
            name=PrefixedName("right_arm_park", prefix=self.name.name),
            mapping=dict(zip(self.active_connections, vals)),
            state_type=StaticJointState.PARK,
        )
        return [arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "CenterArms_fixed_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "ArmR8_Wrist_Hemisphere_B_link"
            ),
        )


@dataclass(eq=False)
class AzureKinectRGB(Camera):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "AzureKinect_RGB_link"
            ),
            forward_facing_axis=Vector3.Z(),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=1.3715,
            maximal_height=1.7365,
            default_camera=True,
        )

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []


@dataclass(eq=False)
class Armar7Neck(Neck[AzureKinectRGB]):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "Neck_Root_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "Head_Root_link"
            ),
        )


@dataclass(eq=False)
class Armar7Torso(
    Torso, HasLeftRightArm[Armar7LeftArm, Armar7RightArm], HasNeck[Armar7Neck]
):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        torso_joints = self.active_connections
        torso_low = JointState.from_mapping(
            name=PrefixedName("torso_low", prefix=self.name.name),
            mapping=dict(zip(torso_joints, [-0.757037, 1.74533, 2.18166 / 2])),
            state_type=TorsoState.LOW,
        )
        torso_mid = JointState.from_mapping(
            name=PrefixedName("torso_mid", prefix=self.name.name),
            mapping=dict(zip(torso_joints, [-0.757037 / 2, 1.74533 / 2, 2.18166 / 4])),
            state_type=TorsoState.MID,
        )
        torso_high = JointState.from_mapping(
            name=PrefixedName("torso_high", prefix=self.name.name),
            mapping=dict(zip(torso_joints, [0.0, 0.0, 0.0])),
            state_type=TorsoState.HIGH,
        )
        return [torso_low, torso_mid, torso_high]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(
                robot_root, "Platform_link"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "CenterArms_fixed_link"
            ),
        )


@dataclass(eq=False)
class Armar7MobileBase(MobileBase, HasTorso[Armar7Torso]):

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
                robot_root, "Dummy_Platform_link"
            ),
            forward_axis=Vector3.Y(),
        )


@dataclass(eq=False)
class Armar7(AbstractRobot, HasMobileBase[Armar7MobileBase]):
    """
    The Armar7 robot built by the KIT.

    https://h2t.iar.kit.edu/397.php
    """

    @classmethod
    def get_ros_file_path(cls) -> str:
        return "package://iai_kit_armar7/urdf/Armar7.urdf"

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "Dummy_Platform_link"

    def _setup_collision_rules(self):
        pass

    def _setup_velocity_limits(self):
        pass
