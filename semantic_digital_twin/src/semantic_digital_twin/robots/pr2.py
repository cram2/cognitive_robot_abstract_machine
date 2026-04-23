import os
from collections import defaultdict
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Self

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
    HasCameras,
    HasParallelGripper,
    HasNeck,
    HasLeftRightArm,
    HasTorso,
    HasMobileBase,
)
from semantic_digital_twin.robots.robot_parts import (
    MobileBase,
    Torso,
    Arm,
    Neck,
    ParallelGripper,
    Finger,
    Camera,
    AbstractRobot,
    FieldOfView,
)
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.world_description.connections import ActiveConnection
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@dataclass(eq=False)
class PR2KinectV1(Camera):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        camera = cls(
            root=world.get_body_in_branch_by_name(
                robot_root, "wide_stereo_optical_frame"
            ),
            forward_facing_axis=Vector3.Z(),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=1.27,
            maximal_height=1.60,
            default_camera=True,
        )
        world.add_semantic_annotation(camera)
        return camera


@dataclass(eq=False)
class PR2RightGripperLeftFinger(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(
                robot_root, "r_gripper_l_finger_link"
            ),
            tip=world.get_body_in_branch_by_name(
                robot_root, "r_gripper_l_finger_tip_link"
            ),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class PR2RightGripperRightFinger(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(
                robot_root, "r_gripper_r_finger_link"
            ),
            tip=world.get_body_in_branch_by_name(
                robot_root, "r_gripper_r_finger_tip_link"
            ),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class PR2LeftGripperLeftFinger(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(
                robot_root, "l_gripper_l_finger_link"
            ),
            tip=world.get_body_in_branch_by_name(
                robot_root, "l_gripper_l_finger_tip_link"
            ),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class PR2LeftGripperRightFinger(Finger):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        finger = cls(
            root=world.get_body_in_branch_by_name(
                robot_root, "l_gripper_r_finger_link"
            ),
            tip=world.get_body_in_branch_by_name(
                robot_root, "l_gripper_r_finger_tip_link"
            ),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class PR2RightGripper(ParallelGripper):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
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

        self.add_joint_state(right_gripper_close)
        self.add_joint_state(right_gripper_open)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        right_gripper = cls(
            root=world.get_body_in_branch_by_name(robot_root, "r_gripper_palm_link"),
            tool_frame=world.get_body_in_branch_by_name(
                robot_root, "r_gripper_tool_frame"
            ),
            front_facing_orientation=Quaternion(0, 0, 0, 1),
        )
        world.add_semantic_annotation(right_gripper)
        return right_gripper

    def setup_finger_semantic_annotations(self):
        thumb = PR2RightGripperLeftFinger.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_thumb(thumb)
        finger = PR2RightGripperRightFinger.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_finger(finger)


@dataclass(eq=False)
class PR2LeftGripper(ParallelGripper):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
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

        self.add_joint_state(left_gripper_close)
        self.add_joint_state(left_gripper_open)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        left_gripper = cls(
            root=world.get_body_in_branch_by_name(robot_root, "l_gripper_palm_link"),
            tool_frame=world.get_body_in_branch_by_name(
                robot_root, "l_gripper_tool_frame"
            ),
            front_facing_orientation=Quaternion(0, 0, 0, 1),
        )
        world.add_semantic_annotation(left_gripper)
        return left_gripper

    def setup_finger_semantic_annotations(self):
        thumb = PR2LeftGripperRightFinger.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_thumb(thumb)
        finger = PR2LeftGripperLeftFinger.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_finger(finger)


@dataclass(eq=False)
class PR2Neck(Neck, HasCameras):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
        pass

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        neck = cls(
            root=world.get_body_in_branch_by_name(robot_root, "torso_lift_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "head_tilt_link"),
        )
        world.add_semantic_annotation(neck)
        return neck

    def setup_sensor_semantic_annotations(self):
        neck = PR2KinectV1.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_sensor(neck)


@dataclass(eq=False)
class PR2LeftArm(Arm, HasParallelGripper):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
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
                        16.996 + 3.14159,
                        -0.073,
                        0.0,
                    ],
                )
            ),
            state_type=StaticJointState.PARK,
        )

        self.add_joint_state(left_arm_park)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        left_arm = cls(
            root=world.get_body_in_branch_by_name(robot_root, "torso_lift_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "l_wrist_roll_link"),
        )
        world.add_semantic_annotation(left_arm)
        return left_arm

    def setup_end_effector_semantic_annotation(self):
        gripper = PR2LeftGripper.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_end_effector(gripper)
        gripper.setup_finger_semantic_annotations()


@dataclass(eq=False)
class PR2RightArm(Arm, HasParallelGripper):

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
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

        self.add_joint_state(right_arm_park)

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        right_arm = cls(
            root=world.get_body_in_branch_by_name(robot_root, "torso_lift_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "r_wrist_roll_link"),
        )
        world.add_semantic_annotation(right_arm)
        return right_arm

    def setup_end_effector_semantic_annotation(self):
        gripper = PR2RightGripper.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_end_effector(gripper)
        gripper.setup_finger_semantic_annotations()


@dataclass(eq=False)
class PR2Torso(Torso, HasLeftRightArm, HasNeck):

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        torso = cls(
            root=world.get_body_in_branch_by_name(robot_root, "base_link"),
            tip=world.get_body_in_branch_by_name(robot_root, "torso_lift_link"),
        )
        world.add_semantic_annotation(torso)
        return torso

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self):
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

        self.add_joint_state(torso_low)
        self.add_joint_state(torso_mid)
        self.add_joint_state(torso_high)

    def setup_arm_semantic_annotations(self):
        left_arm = PR2LeftArm.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_arm(left_arm)
        left_arm.setup_end_effector_semantic_annotation()

        right_arm = PR2RightArm.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        self.add_arm(right_arm)
        right_arm.setup_end_effector_semantic_annotation()

    def setup_neck_semantic_annotation(self):
        neck = PR2Neck.setup_default_configuration_in_world_below_robot_root(self.root)
        neck.setup_sensor_semantic_annotations()
        self.add_neck(neck)


@dataclass(eq=False)
class PR2MobileBase(MobileBase, HasTorso):

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self):
        pass

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        self = cls(
            root=world.get_body_in_branch_by_name(robot_root, "base_link"),
        )
        world.add_semantic_annotation(self)
        return self

    def setup_torso_semantic_annotation(self):
        torso = PR2Torso.setup_default_configuration_in_world_below_robot_root(
            self.root
        )
        torso.setup_arm_semantic_annotations()
        torso.setup_neck_semantic_annotation()
        self.add_torso(torso)


@dataclass(eq=False)
class PR2(AbstractRobot, HasMobileBase):

    @classmethod
    def get_ros_file_path(cls) -> str:
        return "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro"

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "base_footprint"

    def setup_mobile_base_semantic_annotation(self):
        mobile_base = (
            PR2MobileBase.setup_default_configuration_in_world_below_robot_root(
                self.root
            )
        )
        mobile_base.setup_torso_semantic_annotation()
        self.add_mobile_base(mobile_base)

    def setup_robot_part_semantic_annotations(self):
        self.setup_mobile_base_semantic_annotation()

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
                    body_subset=set(
                        self.mobile_base.torso.left_arm.bodies_with_collision
                    )
                    | set(self.mobile_base.torso.right_arm.bodies_with_collision),
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
