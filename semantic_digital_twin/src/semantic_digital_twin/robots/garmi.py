from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import ClassVar, Self

from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidExternalCollisions,
    AvoidSelfCollisions,
)
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
    TorsoState,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import (
    AbstractRobot,
    Arm,
    Base,
    Camera,
    Finger,
    FieldOfView,
    Neck,
    ParallelGripper,
    Torso,
)
from semantic_digital_twin.robots.robot_mixins import HasNeck, SpecifiesLeftRightArm
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection,
    FixedConnection,
)


@dataclass(eq=False)
class Garmi(AbstractRobot, SpecifiesLeftRightArm, HasNeck):
    """
    Semantic annotation for GARMI, a mobile service robot with a mecanum base,
    lift, two Franka FR3 arms, parallel grippers, and a pan/tilt head.
    """

    ARM_PARK_CONFIGURATION: ClassVar[dict[str, float]] = {
        "fr3_joint1": 0.0,
        "fr3_joint2": -0.7853981633974483,
        "fr3_joint3": 0.0,
        "fr3_joint4": -2.356194490192345,
        "fr3_joint5": 0.0,
        "fr3_joint6": 1.5707963267948966,
        "fr3_joint7": 0.7853981633974483,
    }

    @classmethod
    def _init_empty_robot(cls, world: World) -> Self:
        return cls(
            name=PrefixedName(name="garmi", prefix=world.name),
            root=world.get_body_by_name("base_link"),
            _world=world,
            full_body_controlled=True,
        )

    def _setup_semantic_annotations(self):
        base = Base(
            name=PrefixedName("base", prefix=self.name.name),
            root=self._world.get_body_by_name("chassis_link"),
            tip=self._world.get_body_by_name("chassis_link"),
            main_axis=Vector3(1, 0, 0),
            _world=self._world,
        )
        self.add_base(base)

        torso = Torso(
            name=PrefixedName("torso", prefix=self.name.name),
            root=self._world.get_body_by_name("lift_0_base_link"),
            tip=self._world.get_body_by_name("lift_0_mount_rotated_link"),
            _world=self._world,
        )
        self.add_torso(torso)

        for side, arm_id, mount in (
            ("left", "arm_0", "arm_mount_left_link"),
            ("right", "arm_1", "arm_mount_right_link"),
        ):
            arm = self._create_arm(side=side, arm_id=arm_id, mount=mount)
            self.add_arm(arm)

        head_camera = Camera(
            name=PrefixedName("head_camera", prefix=self.name.name),
            root=self._world.get_body_by_name("head"),
            forward_facing_axis=Vector3(1, 0, 0),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=0.75049,
            maximal_height=0.99483,
            _world=self._world,
            default_camera=True,
        )

        neck = Neck(
            name=PrefixedName("neck", prefix=self.name.name),
            sensors=[head_camera],
            root=self._world.get_body_by_name("neck_1"),
            tip=self._world.get_body_by_name("head"),
            yaw_body=self._world.get_body_by_name("neck_2"),
            pitch_body=self._world.get_body_by_name("head"),
            _world=self._world,
        )
        self.add_neck(neck)

    def _create_arm(self, side: str, arm_id: str, mount: str) -> Arm:
        gripper_thumb = Finger(
            name=PrefixedName(f"{side}_gripper_thumb", prefix=self.name.name),
            root=self._world.get_body_by_name(f"{arm_id}_gripper_fr3_hand"),
            tip=self._world.get_body_by_name(f"{arm_id}_gripper_fr3_leftfinger"),
            _world=self._world,
        )
        gripper_finger = Finger(
            name=PrefixedName(f"{side}_gripper_finger", prefix=self.name.name),
            root=self._world.get_body_by_name(f"{arm_id}_gripper_fr3_hand"),
            tip=self._world.get_body_by_name(f"{arm_id}_gripper_fr3_rightfinger"),
            _world=self._world,
        )
        gripper = ParallelGripper(
            name=PrefixedName(f"{side}_gripper", prefix=self.name.name),
            root=self._world.get_body_by_name(f"{arm_id}_gripper_fr3_hand"),
            tool_frame=self._world.get_body_by_name(f"{arm_id}_gripper_fr3_hand_tcp"),
            front_facing_orientation=Quaternion(0, 0, 0, 1),
            front_facing_axis=Vector3(0, 0, 1),
            thumb=gripper_thumb,
            finger=gripper_finger,
            _world=self._world,
        )
        return Arm(
            name=PrefixedName(f"{side}_arm", prefix=self.name.name),
            root=self._world.get_body_by_name(mount),
            tip=self._world.get_body_by_name(f"{arm_id}_fr3_link8"),
            manipulator=gripper,
            _world=self._world,
        )

    def _setup_collision_rules(self):
        self._world.collision_manager.add_default_rule(
            AvoidExternalCollisions(
                buffer_zone_distance=0.05, violated_distance=0.0, robot=self
            )
        )
        self._world.collision_manager.add_default_rule(
            AvoidSelfCollisions(
                buffer_zone_distance=0.03,
                violated_distance=0.0,
                robot=self,
            )
        )

    def _setup_velocity_limits(self):
        vel_limits = defaultdict(lambda: 0.2)
        for joint_name in (
            "front_left_wheel_joint",
            "front_right_wheel_joint",
            "rear_left_wheel_joint",
            "rear_right_wheel_joint",
        ):
            vel_limits[self._world.get_connection_by_name(joint_name)] = 1.3
        for joint_name in ("head_pan_joint", "head_tilt_joint"):
            vel_limits[self._world.get_connection_by_name(joint_name)] = 1.0
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)

    def _setup_hardware_interfaces(self):
        controlled_joints = [
            "front_left_wheel_joint",
            "front_right_wheel_joint",
            "rear_left_wheel_joint",
            "rear_right_wheel_joint",
            "lift_0_lower_joint",
            "lift_0_upper_joint",
            "head_pan_joint",
            "head_tilt_joint",
        ]
        controlled_joints.extend(
            f"arm_{arm_index}_fr3_joint{joint_index}"
            for arm_index in range(2)
            for joint_index in range(1, 8)
        )
        controlled_joints.extend(
            f"arm_{arm_index}_gripper_fr3_finger_joint{finger_index}"
            for arm_index in range(2)
            for finger_index in range(1, 3)
        )

        for joint_name in controlled_joints:
            connection: ActiveConnection = self._world.get_connection_by_name(
                joint_name
            )
            connection.has_hardware_interface = True

    def _setup_joint_states(self):
        for arm_index, arm in enumerate(self.arms):
            arm_park = JointState.from_mapping(
                name=PrefixedName(f"{arm.name.name}_park", prefix=self.name.name),
                mapping={
                    connection: position
                    for connection in arm.connections
                    if type(connection) != FixedConnection
                    for joint_name, position in self.ARM_PARK_CONFIGURATION.items()
                    if connection.name.name.endswith(joint_name)
                },
                state_type=StaticJointState.PARK,
            )
            arm.add_joint_state(arm_park)

            gripper_joint_names = [
                f"arm_{arm_index}_gripper_fr3_finger_joint1",
                f"arm_{arm_index}_gripper_fr3_finger_joint2",
            ]
            gripper_joints = [
                self._world.get_connection_by_name(name) for name in gripper_joint_names
            ]
            gripper_open = JointState.from_mapping(
                name=PrefixedName(
                    f"{arm.name.name}_gripper_open", prefix=self.name.name
                ),
                mapping=dict(zip(gripper_joints, [0.04, 0.04])),
                state_type=GripperState.OPEN,
            )
            gripper_close = JointState.from_mapping(
                name=PrefixedName(
                    f"{arm.name.name}_gripper_close", prefix=self.name.name
                ),
                mapping=dict(zip(gripper_joints, [0.0, 0.0])),
                state_type=GripperState.CLOSE,
            )
            arm.manipulator.add_joint_state(gripper_open)
            arm.manipulator.add_joint_state(gripper_close)

        lift_joints = [
            self._world.get_connection_by_name("lift_0_lower_joint"),
            self._world.get_connection_by_name("lift_0_upper_joint"),
        ]
        torso_states = (
            ("torso_low", [0.0, 0.0], TorsoState.LOW),
            ("torso_mid", [0.2, 0.2], TorsoState.MID),
            ("torso_high", [0.4, 0.4], TorsoState.HIGH),
        )
        for name, positions, state_type in torso_states:
            self.torso.add_joint_state(
                JointState.from_mapping(
                    name=PrefixedName(name, prefix=self.name.name),
                    mapping=dict(zip(lift_joints, positions)),
                    state_type=state_type,
                )
            )
