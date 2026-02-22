import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Self

from pkg_resources import resource_filename

from .abstract_robot import (
    AbstractRobot,
    Arm,
    Neck,
    Finger,
    ParallelGripper,
    Camera,
    Torso,
    FieldOfView,
    Base,
)
from .robot_mixins import HasNeck, HasArms
from ..collision_checking.collision_matrix import MaxAvoidedCollisionsOverride
from ..collision_checking.collision_rules import (
    SelfCollisionMatrixRule,
    AvoidExternalCollisions,
    AvoidSelfCollisions,
)
from ..datastructures.definitions import StaticJointState, GripperState, TorsoState
from ..datastructures.joint_state import JointState
from ..datastructures.prefixed_name import PrefixedName
from ..spatial_types import Quaternion
from ..spatial_types.spatial_types import Vector3
from ..world import World
from ..world_description.connections import FixedConnection, ActiveConnection


@dataclass(eq=False)
class HSRB(AbstractRobot, HasArms, HasNeck):
    """
    Class that describes the Human Support Robot variant B (https://upmroboticclub.wordpress.com/robot/).
    """

    @classmethod
    def _init_empty_robot(cls, world: World) -> Self:
        return cls(
            name=PrefixedName("hsrb", prefix=world.name),
            root=world.get_body_by_name("base_footprint"),
            _world=world,
        )

    @property
    def arm(self) -> Arm:
        return self.arms[0]

    def _setup_semantic_annotations(self):
        gripper_thumb = Finger(
            name=PrefixedName("thumb", prefix=self.name.name),
            root=self._world.get_body_by_name("hand_l_proximal_link"),
            tip=self._world.get_body_by_name("hand_l_finger_tip_frame"),
            _world=self._world,
        )

        gripper_finger = Finger(
            name=PrefixedName("finger", prefix=self.name.name),
            root=self._world.get_body_by_name("hand_r_proximal_link"),
            tip=self._world.get_body_by_name("hand_r_finger_tip_frame"),
            _world=self._world,
        )

        gripper = ParallelGripper(
            name=PrefixedName("gripper", prefix=self.name.name),
            root=self._world.get_body_by_name("hand_palm_link"),
            tool_frame=self._world.get_body_by_name("hand_gripper_tool_frame"),
            thumb=gripper_thumb,
            finger=gripper_finger,
            front_facing_axis=Vector3(0, 0, 1),
            front_facing_orientation=Quaternion(
                -0.70710678,
                0.0,
                -0.70710678,
                0.0,
            ),
            _world=self._world,
        )

        # the min and max height are incorrect, same with the FoV. needs to be corrected using the real robot
        hand_camera = Camera(
            name=PrefixedName("hand_camera", prefix=self.name.name),
            root=self._world.get_body_by_name("hand_camera_frame"),
            forward_facing_axis=Vector3(0, 0, 1),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=0.75049,
            maximal_height=0.99483,
            _world=self._world,
        )

        arm = Arm(
            name=PrefixedName("arm", prefix=self.name.name),
            root=self._world.get_body_by_name("arm_lift_link"),
            tip=self._world.get_body_by_name("hand_palm_link"),
            manipulator=gripper,
            sensors={hand_camera},
            _world=self._world,
        )
        self.add_arm(arm)

        # Create camera and neck
        head_center_camera = Camera(
            name=PrefixedName("head_center_camera", prefix=self.name.name),
            root=self._world.get_body_by_name("head_center_camera_frame"),
            forward_facing_axis=Vector3(0, 0, 1),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=0.75049,
            maximal_height=0.99483,
            _world=self._world,
            default_camera=True,
        )

        head_r_camera = Camera(
            name=PrefixedName("head_right_camera", prefix=self.name.name),
            root=self._world.get_body_by_name("head_r_stereo_camera_link"),
            forward_facing_axis=Vector3(0, 0, 1),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=0.75049,
            maximal_height=0.99483,
            _world=self._world,
        )

        head_l_camera = Camera(
            name=PrefixedName("head_left_camera", prefix=self.name.name),
            root=self._world.get_body_by_name("head_l_stereo_camera_link"),
            forward_facing_axis=Vector3(0, 0, 1),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=0.75049,
            maximal_height=0.99483,
            _world=self._world,
        )

        head_rgbd_camera = Camera(
            name=PrefixedName("head_rgbd_camera", prefix=self.name.name),
            root=self._world.get_body_by_name("head_rgbd_sensor_link"),
            forward_facing_axis=Vector3(0, 0, 1),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=0.75049,
            maximal_height=0.99483,
            _world=self._world,
        )

        neck = Neck(
            name=PrefixedName("neck", prefix=self.name.name),
            sensors=[
                head_center_camera,
                head_r_camera,
                head_l_camera,
                head_rgbd_camera,
            ],
            root=self._world.get_body_by_name("head_pan_link"),
            tip=self._world.get_body_by_name("head_tilt_link"),
            _world=self._world,
        )
        self.add_neck(neck)

        # Create torso
        torso = Torso(
            name=PrefixedName("torso", prefix=self.name.name),
            root=self._world.get_body_by_name("base_link"),
            tip=self._world.get_body_by_name("torso_lift_link"),
            _world=self._world,
        )
        self.add_torso(torso)

        base = Base(
            name=PrefixedName("base", prefix=self.name.name),
            root=self._world.get_body_by_name("base_link"),
            tip=self._world.get_body_by_name("base_link"),
            _world=self._world,
        )

        self.add_base(base)

        self.full_body_controlled = True

    def _setup_collision_rules(self):
        srdf_path = os.path.join(
            resource_filename("semantic_digital_twin", "../../"),
            "resources",
            "collision_configs",
            "hsrb.srdf",
        )
        self._world.collision_manager.ignore_collision_rules.append(
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
                body_subset={self._world.get_body_by_name("base_link")},
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
                2, bodies={self._world.get_body_by_name("base_link")}
            )
        )
        self._world.collision_manager.max_avoided_bodies_rules.append(
            MaxAvoidedCollisionsOverride(
                4,
                bodies=set(
                    self._world.get_direct_child_bodies_with_collision(
                        self._world.get_body_by_name("wrist_roll_link")
                    )
                ),
            )
        )

    def _setup_velocity_limits(self):
        vel_limits = defaultdict(lambda: 1)
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)

    def _setup_hardware_interfaces(self):
        controlled_joints = [
            "arm_flex_joint",
            "arm_lift_joint",
            "arm_roll_joint",
            "head_pan_joint",
            "head_tilt_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
        ]
        for joint_name in controlled_joints:
            connection: ActiveConnection = self._world.get_connection_by_name(
                joint_name
            )
            connection.has_hardware_interface = True

    def _setup_joint_states(self):
        # Create states
        arm_park = JointState.from_mapping(
            name=PrefixedName("arm_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    [c for c in self.arm.connections if type(c) != FixedConnection],
                    [0.0, 1.5, -1.85, 0.0],
                )
            ),
            state_type=StaticJointState.PARK,
        )

        self.arm.add_joint_state(arm_park)

        gripper_joints = [
            self._world.get_connection_by_name("hand_l_proximal_joint"),
            self._world.get_connection_by_name("hand_r_proximal_joint"),
            self._world.get_connection_by_name("hand_motor_joint"),
        ]

        gripper_open = JointState.from_mapping(
            name=PrefixedName("gripper_open", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.3, 0.3, 0.3])),
            state_type=GripperState.OPEN,
        )

        gripper_close = JointState.from_mapping(
            name=PrefixedName("gripper_close", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.0, 0.0, 0.0])),
            state_type=GripperState.CLOSE,
        )

        self.arm.manipulator.add_joint_state(gripper_close)
        self.arm.manipulator.add_joint_state(gripper_open)

        torso_joint = [self._world.get_connection_by_name("torso_lift_joint")]

        torso_low = JointState.from_mapping(
            name=PrefixedName("torso_low", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.0])),
            state_type=TorsoState.LOW,
        )

        torso_mid = JointState.from_mapping(
            name=PrefixedName("torso_mid", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.17])),
            state_type=TorsoState.MID,
        )

        torso_high = JointState.from_mapping(
            name=PrefixedName("torso_high", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.32])),
            state_type=TorsoState.HIGH,
        )

        self.torso.add_joint_state(torso_low)
        self.torso.add_joint_state(torso_mid)
        self.torso.add_joint_state(torso_high)
