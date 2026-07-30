from __future__ import annotations

from abc import ABC

import numpy as np
from dataclasses import dataclass
from typing import Self, Union, List

from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import (
    HasLeftRightArm,
    HasNeck,
    HasTorso,
    HasMobileBase,
    HasFingers,
    HasSensors,
    TGenericFingerOtherThanThumb,
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
class ICub3LeftThumb(Finger):

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
                robot_root, "l_hand_thumb_0"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "l_hand_thumb_tip"
            ),
        )


@dataclass(eq=False)
class ICub3LeftIndexFinger(Finger):

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
                robot_root, "l_hand_index_0"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "l_hand_index_tip"
            ),
        )


@dataclass(eq=False)
class ICub3LeftMiddleFinger(Finger):

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
                robot_root, "l_hand_middle_0"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "l_hand_middle_tip"
            ),
        )


@dataclass(eq=False)
class ICub3LeftRingFinger(Finger):

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
                robot_root, "l_hand_ring_0"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "l_hand_ring_tip"
            ),
        )


@dataclass(eq=False)
class ICub3LeftLittleFinger(Finger):

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
                robot_root, "l_hand_little_0"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "l_hand_little_tip"
            ),
        )


@dataclass(eq=False)
class ICub3RightThumb(Finger):

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
                robot_root, "r_hand_thumb_0"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "r_hand_thumb_tip"
            ),
        )


@dataclass(eq=False)
class ICub3RightIndexFinger(Finger):

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
                robot_root, "r_hand_index_0"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "r_hand_index_tip"
            ),
        )


@dataclass(eq=False)
class ICub3RightMiddleFinger(Finger):

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
                robot_root, "r_hand_middle_0"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "r_hand_middle_tip"
            ),
        )


@dataclass(eq=False)
class ICub3RightRingFinger(Finger):

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
                robot_root, "r_hand_ring_0"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "r_hand_ring_tip"
            ),
        )


@dataclass(eq=False)
class ICub3RightLittleFinger(Finger):

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
                robot_root, "r_hand_little_0"
            ),
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "r_hand_little_tip"
            ),
        )


@dataclass(eq=False)
class ICub3LeftHand(
    EndEffector,
    HasFingers[
        ICub3LeftThumb,
        ICub3LeftIndexFinger,
        ICub3LeftMiddleFinger,
        ICub3LeftRingFinger,
        ICub3LeftLittleFinger,
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

        close_vals = [
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
            -0.3490658503988659,
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
            0.3490658503988659,
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
            0.3490658503988659,
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
            0.3490658503988659,
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
        ]

        gripper_close = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_close", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, close_vals)),
            state_type=GripperState.CLOSE,
        )

        return [gripper_open, gripper_close]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "l_hand"),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "l_gripper_tool_frame"
            ),
            front_facing_orientation=Quaternion(0.5, 0.5, 0.5, 0.5),
        )


@dataclass(eq=False)
class ICub3RightHand(
    EndEffector,
    HasFingers[
        ICub3RightThumb,
        ICub3RightIndexFinger,
        ICub3RightMiddleFinger,
        ICub3RightRingFinger,
        ICub3RightLittleFinger,
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

        close_vals = [
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
            -0.3490658503988659,
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
            0.3490658503988659,
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
            0.3490658503988659,
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
            0.3490658503988659,
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
        ]

        gripper_close = JointState.from_mapping(
            name=PrefixedName(f"{self.name.name}_close", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, close_vals)),
            state_type=GripperState.CLOSE,
        )

        return [gripper_open, gripper_close]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "r_hand"),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "r_gripper_tool_frame"
            ),
            front_facing_orientation=Quaternion(0.5, 0.5, 0.5, 0.5),
        )


@dataclass(eq=False)
class ICub3LeftArm(Arm[ICub3LeftHand]):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        arm_park = JointState.from_mapping(
            name=PrefixedName("left_arm_park", prefix=self.name.name),
            mapping=dict(
                zip(self.active_connections, [0.0] * len(self.active_connections))
            ),
            state_type=StaticJointState.PARK,
        )
        return [arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "root_link"),
            tip=robot_root._world.get_body_in_branch_by_name(robot_root, "l_hand"),
        )


@dataclass(eq=False)
class ICub3RightArm(Arm[ICub3RightHand]):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        arm_park = JointState.from_mapping(
            name=PrefixedName("right_arm_park", prefix=self.name.name),
            mapping=dict(
                zip(self.active_connections, [0.0] * len(self.active_connections))
            ),
            state_type=StaticJointState.PARK,
        )
        return [arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "root_link"),
            tip=robot_root._world.get_body_in_branch_by_name(robot_root, "r_hand"),
        )


@dataclass(eq=False)
class ICub3Camera(Camera):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "head"),
            forward_facing_axis=Vector3.Z(),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=0.75049,
            maximal_height=0.99483,
            default_camera=True,
        )

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []


@dataclass(eq=False)
class ICub3Neck(Neck[ICub3Camera]):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "chest"),
            tip=robot_root._world.get_body_in_branch_by_name(robot_root, "head"),
        )


@dataclass(eq=False)
class ICub3Torso(
    Torso, HasLeftRightArm[ICub3LeftArm, ICub3RightArm], HasNeck[ICub3Neck]
):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "root_link"),
            tip=robot_root._world.get_body_in_branch_by_name(robot_root, "chest"),
        )


@dataclass(eq=False)
class ICub3MobileBase(MobileBase):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "l_hip_1"),
        )

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []


@dataclass(eq=False)
class ICub3(AbstractRobot, HasTorso[ICub3Torso], HasMobileBase[ICub3MobileBase]):
    """
    The ICub3 robot built by the Istituto Italiano di Tecnologia.

    https://ami.iit.it/telexistence
    """

    def _setup_collision_rules(self):
        pass

    @classmethod
    def get_ros_file_path(cls) -> str:
        return (
            "package://iai_icub_description/robots/iCubGazeboV3_visuomanip/iCub3.urdf"
        )

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "base_footprint"
