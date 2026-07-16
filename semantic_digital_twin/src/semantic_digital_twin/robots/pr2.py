import os
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Self, List

from typing_extensions import Type

from semantic_digital_twin.collision_checking.collision_matrix import (
    MaxAvoidedCollisionsOverride,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    SelfCollisionMatrixRule,
    AvoidExternalCollisions,
    AvoidSelfCollisions,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.definitions import (
    StaticJointState,
    GripperState,
    TorsoState,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import (
    HasNeck,
    HasLeftRightArm,
    HasTorso,
    HasMobileBase,
    HasTwoFingers,
    HasSensors,
    HasEndEffector,
    TGenericLeftFinger,
    TGenericRightFinger,
    HasFingers,
    TGenericFingerOtherThanThumb,
)
from semantic_digital_twin.robots.robot_parts import (
    MobileBase,
    Torso,
    Arm,
    Neck,
    Finger,
    Camera,
    AbstractRobot,
    EndEffector,
)
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.world_description.connections import ActiveConnection
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@dataclass(eq=False)
class PR2KinectV1(Camera):

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
                robot_root, "wide_stereo_optical_frame"
            ),
            forward_facing_axis=Vector3.Z(),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=1.27,
            maximal_height=1.60,
            default_camera=True,
        )


@dataclass(eq=False)
class PR2RightGripperLeftFinger(Finger):

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

    def setup_joint_states(self) -> List[JointState]:
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

    def setup_joint_states(self) -> List[JointState]:
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

    def setup_joint_states(self) -> List[JointState]:
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
class PR2RightGripper(
    EndEffector, HasTwoFingers[PR2RightGripperLeftFinger, PR2RightGripperRightFinger]
):

    def setup_joint_states(self) -> List[JointState]:
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
            front_facing_orientation=Quaternion(0, 0, 0, 1),
        )


@dataclass(eq=False)
class PR2LeftGripper(
    EndEffector, HasTwoFingers[PR2LeftGripperLeftFinger, PR2LeftGripperRightFinger]
):

    def setup_joint_states(self) -> List[JointState]:
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
            front_facing_orientation=Quaternion(0, 0, 0, 1),
        )

    def setup_hardware_interfaces(self):
        return


@dataclass(eq=False)
class PR2Neck(Neck[PR2KinectV1]):

    def setup_hardware_interfaces(self):

        controlled_joints = [
            "head_pan_joint",
            "head_tilt_joint",
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
            tip=robot_root._world.get_body_in_branch_by_name(
                robot_root, "head_tilt_link"
            ),
        )


@dataclass(eq=False)
class PR2LeftArm(Arm[PR2LeftGripper]):

    def setup_hardware_interfaces(self):
        controlled_joints = [
            "l_shoulder_pan_joint",
            "l_shoulder_lift_joint",
            "l_upper_arm_roll_joint",
            "l_forearm_roll_joint",
            "l_elbow_flex_joint",
            "l_wrist_flex_joint",
            "l_wrist_roll_joint",
        ]
        for joint_name in controlled_joints:
            connection: ActiveConnection = self._world.get_connection_by_name(
                joint_name
            )
            connection.has_hardware_interface = True

    def setup_joint_states(self) -> List[JointState]:
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
                robot_root, "l_wrist_roll_link"
            ),
        )


@dataclass(eq=False)
class PR2RightArm(Arm[PR2RightGripper]):

    def setup_hardware_interfaces(self):
        controlled_joints = [
            "r_shoulder_pan_joint",
            "r_shoulder_lift_joint",
            "r_upper_arm_roll_joint",
            "r_forearm_roll_joint",
            "r_elbow_flex_joint",
            "r_wrist_flex_joint",
            "r_wrist_roll_joint",
        ]
        for joint_name in controlled_joints:
            connection: ActiveConnection = self._world.get_connection_by_name(
                joint_name
            )
            connection.has_hardware_interface = True

    def setup_joint_states(self) -> List[JointState]:
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
class PR2Torso(Torso, HasLeftRightArm[PR2LeftArm, PR2RightArm], HasNeck[PR2Neck]):

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


@dataclass(eq=False)
class PR2MobileBase(MobileBase, HasTorso[PR2Torso]):

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
class PR2(AbstractRobot, HasMobileBase[PR2MobileBase]):
    """
    The PR2 robot built by Willow Garage.

    https://robotsguide.com/robots/pr2
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
                self._world.get_connection_by_name("head_tilt_joint"): 3.5,
                self._world.get_connection_by_name("r_shoulder_pan_joint"): 0.15,
                self._world.get_connection_by_name("l_shoulder_pan_joint"): 0.15,
                self._world.get_connection_by_name("r_shoulder_lift_joint"): 0.2,
                self._world.get_connection_by_name("l_shoulder_lift_joint"): 0.2,
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
                    buffer_zone_distance=0.2,
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
