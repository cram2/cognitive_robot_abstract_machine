from collections import defaultdict
from dataclasses import dataclass
from typing import Self

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
from ..datastructures.definitions import StaticJointState, GripperState, TorsoState
from ..datastructures.joint_state import JointState
from ..datastructures.prefixed_name import PrefixedName
from ..spatial_types import Quaternion
from ..spatial_types.spatial_types import Vector3
from ..world import World
from ..world_description.connections import FixedConnection


@dataclass(eq=False)
class HSRB(AbstractRobot, HasArms, HasNeck):
    """
    Class that describes the Human Support Robot variant B (https://upmroboticclub.wordpress.com/robot/).
    """

    def load_srdf(self):
        """
        Loads the SRDF file for the PR2 robot, if it exists.
        """
        ...

    @classmethod
    def from_world(cls, world: World) -> Self:
        """
        Creates an HSRB (Human Support Robot B) semantic annotation from a World that was parsed from
        resources/urdf/robots/hsrb.urdf. Assumes all URDF link names exist in the world.
        """
        with world.modify_world():
            hsrb = cls(
                name=PrefixedName("hsrb", prefix=world.name),
                root=world.get_body_by_name("base_footprint"),
                _world=world,
            )

            gripper_thumb = Finger(
                name=PrefixedName("thumb", prefix=hsrb.name.name),
                root=world.get_body_by_name("hand_l_proximal_link"),
                tip=world.get_body_by_name("hand_l_finger_tip_frame"),
                _world=world,
            )

            gripper_finger = Finger(
                name=PrefixedName("finger", prefix=hsrb.name.name),
                root=world.get_body_by_name("hand_r_proximal_link"),
                tip=world.get_body_by_name("hand_r_finger_tip_frame"),
                _world=world,
            )

            gripper = ParallelGripper(
                name=PrefixedName("gripper", prefix=hsrb.name.name),
                root=world.get_body_by_name("hand_palm_link"),
                tool_frame=world.get_body_by_name("hand_gripper_tool_frame"),
                thumb=gripper_thumb,
                finger=gripper_finger,
                front_facing_axis=Vector3(0, 0, 1),
                front_facing_orientation=Quaternion(
                    -0.70710678,
                    0.0,
                    -0.70710678,
                    0.0,
                ),
                _world=world,
            )

            # the min and max height are incorrect, same with the FoV. needs to be corrected using the real robot
            hand_camera = Camera(
                name=PrefixedName("hand_camera", prefix=hsrb.name.name),
                root=world.get_body_by_name("hand_camera_frame"),
                forward_facing_axis=Vector3(0, 0, 1),
                field_of_view=FieldOfView(
                    horizontal_angle=0.99483, vertical_angle=0.75049
                ),
                minimal_height=0.75049,
                maximal_height=0.99483,
                _world=world,
            )

            arm = Arm(
                name=PrefixedName("arm", prefix=hsrb.name.name),
                root=world.get_body_by_name("arm_lift_link"),
                tip=world.get_body_by_name("hand_palm_link"),
                manipulator=gripper,
                sensors={hand_camera},
                _world=world,
            )
            hsrb.add_arm(arm)

            # Create camera and neck
            head_center_camera = Camera(
                name=PrefixedName("head_center_camera", prefix=hsrb.name.name),
                root=world.get_body_by_name("head_center_camera_frame"),
                forward_facing_axis=Vector3(0, 0, 1),
                field_of_view=FieldOfView(
                    horizontal_angle=0.99483, vertical_angle=0.75049
                ),
                minimal_height=0.75049,
                maximal_height=0.99483,
                _world=world,
                default_camera=True,
            )

            head_r_camera = Camera(
                name=PrefixedName("head_right_camera", prefix=hsrb.name.name),
                root=world.get_body_by_name("head_r_stereo_camera_link"),
                forward_facing_axis=Vector3(0, 0, 1),
                field_of_view=FieldOfView(
                    horizontal_angle=0.99483, vertical_angle=0.75049
                ),
                minimal_height=0.75049,
                maximal_height=0.99483,
                _world=world,
            )

            head_l_camera = Camera(
                name=PrefixedName("head_left_camera", prefix=hsrb.name.name),
                root=world.get_body_by_name("head_l_stereo_camera_link"),
                forward_facing_axis=Vector3(0, 0, 1),
                field_of_view=FieldOfView(
                    horizontal_angle=0.99483, vertical_angle=0.75049
                ),
                minimal_height=0.75049,
                maximal_height=0.99483,
                _world=world,
            )

            head_rgbd_camera = Camera(
                name=PrefixedName("head_rgbd_camera", prefix=hsrb.name.name),
                root=world.get_body_by_name("head_rgbd_sensor_link"),
                forward_facing_axis=Vector3(0, 0, 1),
                field_of_view=FieldOfView(
                    horizontal_angle=0.99483, vertical_angle=0.75049
                ),
                minimal_height=0.75049,
                maximal_height=0.99483,
                _world=world,
            )

            neck = Neck(
                name=PrefixedName("neck", prefix=hsrb.name.name),
                sensors=[
                    head_center_camera,
                    head_r_camera,
                    head_l_camera,
                    head_rgbd_camera,
                ],
                root=world.get_body_by_name("head_pan_link"),
                tip=world.get_body_by_name("head_tilt_link"),
                _world=world,
            )
            hsrb.add_neck(neck)

            # Create torso
            torso = Torso(
                name=PrefixedName("torso", prefix=hsrb.name.name),
                root=world.get_body_by_name("base_link"),
                tip=world.get_body_by_name("torso_lift_link"),
                _world=world,
            )
            hsrb.add_torso(torso)

            # Create states
            arm_park = JointState.from_mapping(
                name=PrefixedName("arm_park", prefix=hsrb.name.name),
                mapping=dict(
                    zip(
                        [c for c in arm.connections if type(c) != FixedConnection],
                        [0.0, 1.5, -1.85, 0.0],
                    )
                ),
                state_type=StaticJointState.PARK,
            )

            arm.add_joint_state(arm_park)

            gripper_joints = [
                world.get_connection_by_name("hand_l_proximal_joint"),
                world.get_connection_by_name("hand_r_proximal_joint"),
                world.get_connection_by_name("hand_motor_joint"),
            ]

            gripper_open = JointState.from_mapping(
                name=PrefixedName("gripper_open", prefix=hsrb.name.name),
                mapping=dict(zip(gripper_joints, [0.3, 0.3, 0.3])),
                state_type=GripperState.OPEN,
            )

            gripper_close = JointState.from_mapping(
                name=PrefixedName("gripper_close", prefix=hsrb.name.name),
                mapping=dict(zip(gripper_joints, [0.0, 0.0, 0.0])),
                state_type=GripperState.CLOSE,
            )

            gripper.add_joint_state(gripper_close)
            gripper.add_joint_state(gripper_open)

            torso_joint = [world.get_connection_by_name("torso_lift_joint")]

            torso_low = JointState.from_mapping(
                name=PrefixedName("torso_low", prefix=hsrb.name.name),
                mapping=dict(zip(torso_joint, [0.0])),
                state_type=TorsoState.LOW,
            )

            torso_mid = JointState.from_mapping(
                name=PrefixedName("torso_mid", prefix=hsrb.name.name),
                mapping=dict(zip(torso_joint, [0.17])),
                state_type=TorsoState.MID,
            )

            torso_high = JointState.from_mapping(
                name=PrefixedName("torso_high", prefix=hsrb.name.name),
                mapping=dict(zip(torso_joint, [0.32])),
                state_type=TorsoState.HIGH,
            )

            torso.add_joint_state(torso_low)
            torso.add_joint_state(torso_mid)
            torso.add_joint_state(torso_high)

            # Create the robot base
            base = Base(
                name=PrefixedName("base", prefix=hsrb.name.name),
                root=world.get_body_by_name("base_link"),
                tip=world.get_body_by_name("base_link"),
                _world=world,
            )

            hsrb.add_base(base)
            hsrb.full_body_controlled = True

            world.add_semantic_annotation(hsrb)

            vel_limits = defaultdict(lambda: 1)
            hsrb.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)

        return hsrb
