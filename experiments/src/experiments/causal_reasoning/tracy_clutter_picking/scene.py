"""
The ten-milk clutter on Tracy's own table, built for MuJoCo from a
:class:`~experiments.causal_reasoning.tracy_clutter_picking.domain.ClutterSceneLayout`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from coraplex.datastructures.enums import Arms
from typing_extensions import Dict, List

from experiments.causal_reasoning.tracy_clutter_picking.domain import ClutterSceneLayout
from experiments.causal_reasoning.tracy_clutter_picking.tracy_mujoco_addons.equipment import (
    add_box,
    apply_gravity_compensation,
    equip_arms_with_servos,
    equip_grippers_with_servos,
    exclude_self_collision,
    joint_state_of_type,
    mount_stationary_robot,
    parse_tracy,
    tracy_table_mount_position,
)
from experiments.causal_reasoning.tracy_clutter_picking.tracy_mujoco_addons.grasp_contact import (
    ContactParameters,
)
from experiments.causal_reasoning.tracy_clutter_picking.tracy_mujoco_addons.trajectory_planning import (
    RobotiqGripper,
    arm_of,
)
from semantic_digital_twin.adapters.multi_sim import MujocoCamera, MujocoLight
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types.spatial_types import Point3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color, Scale
from semantic_digital_twin.world_description.world_entity import Actuator, Body


@dataclass
class MilkClutterWorld:
    """
    Tracy, its table, and one layout's cartons standing on it, equipped to be driven by
    direct MuJoCo actuator control.
    """

    layout: ClutterSceneLayout
    """
    The layout the cartons stand in.
    """

    milk_size: Scale = field(default_factory=lambda: Scale(0.06, 0.06, 0.15))
    """
    Edge lengths of a milk carton, in metres: a small carton, narrow enough to leave the
    Robotiq 2F-85's 85mm opening room to close around it even when it stands a little
    turned.
    """

    milk_color: Color = field(default_factory=lambda: Color(0.95, 0.95, 0.9, 1.0))
    """
    Colour of every carton but the target.
    """

    target_color: Color = field(default_factory=lambda: Color(0.2, 0.5, 0.9, 1.0))
    """
    Colour of the carton to pick, so a viewer can tell it apart.
    """

    pick_arm: Arms = Arms.LEFT
    """
    The arm that picks; the clutter stands in front of it.
    """

    mount_x: float = 0.0
    """
    Where Tracy's own root is bolted along the scene's x-axis, in metres.
    """

    mount_y: float = 0.0
    """
    Where Tracy's own root is bolted along the scene's y-axis, in metres.
    """

    surface_contact: ContactParameters = field(
        default_factory=ContactParameters.surface
    )
    """
    The contact parameters of the table top the cartons stand on.
    """

    camera_name: str = "clutter_overview_camera"
    """
    Name of the fixed camera framing the clutter, for screenshots of a run.
    """

    camera_frame_margin: float = 0.15
    """
    How far, in metres, the camera's framed box extends beyond the cartons on every
    side, so the gripper reaching in stays in view.
    """

    camera_distance_factor: float = 1.1
    """
    How far the camera stands back from the framed box, as a multiple of the box's
    diagonal.
    """

    light_name: str = "clutter_light"
    """
    Name of the directional light over the table.
    """

    world: World = field(init=False)
    """
    The assembled world.
    """

    robot: Tracy = field(init=False)
    """
    The mounted robot.
    """

    table_top_z: float = field(init=False)
    """
    Height of the table's top surface above the world root, in metres.
    """

    milks: List[Body] = field(init=False, default_factory=list)
    """
    The cartons, in the order of :attr:`ClutterSceneLayout.objects`.
    """

    actuators: Dict[str, Actuator] = field(init=False, default_factory=dict)
    """
    Every arm and gripper joint's own actuator, keyed by joint name.
    """

    def __post_init__(self):
        tracy_world = parse_tracy()
        mount_position, self.table_top_z = tracy_table_mount_position(
            tracy_world, x=self.mount_x, y=self.mount_y
        )
        self.world = World()
        with self.world.modify_world():
            self.world.add_kinematic_structure_entity(
                Body(name=PrefixedName(name="root", prefix="clutter"))
            )
        self.robot = mount_stationary_robot(
            self.world, Tracy, tracy_world, mount_position
        )
        self._add_milks()
        self._add_camera_and_light()
        self._equip_robot()

    @staticmethod
    def milk_name(index: int) -> str:
        """
        :param index: The carton's position in the layout's object list.
        :return: The name its body gets.
        """
        return f"milk_{index}"

    @property
    def target(self) -> Body:
        """
        The carton to pick.
        """
        return self.milks[self.layout.target_index]

    @property
    def neighbours(self) -> List[Body]:
        """
        Every carton but the target, in the order of
        :attr:`ClutterSceneLayout.neighbours`.
        """
        return [
            milk
            for index, milk in enumerate(self.milks)
            if index != self.layout.target_index
        ]

    @property
    def grasp_contact(self) -> ContactParameters:
        """
        The contact parameters of the grasp: the layout's own friction coefficient on a
        grasped object's solver settings.
        """
        return ContactParameters.grasped_object(
            sliding_friction=self.layout.friction_coefficient
        )

    def _add_milks(self) -> None:
        """
        Stand every carton of the layout on the table, with the layout's own friction on
        the cartons and on the picking gripper's fingertip pads alike.

        MuJoCo gives a contact the larger of its two geoms' friction, so a carton's
        friction only governs the grasp if the pads closing on it carry no more than
        that themselves.
        """
        for index, placed in enumerate(self.layout.objects):
            self.milks.append(
                add_box(
                    self.world,
                    self.milk_name(index),
                    Point3(placed.x, placed.y, self.table_top_z + self.milk_size.z / 2),
                    self.milk_size,
                    (
                        self.target_color
                        if index == self.layout.target_index
                        else self.milk_color
                    ),
                    yaw=placed.yaw,
                    contact=self.grasp_contact,
                )
            )
        gripper = RobotiqGripper(self.pick_arm)
        ContactParameters(friction=self.grasp_contact.friction).apply_to(
            [
                self.world.get_body_by_name(gripper.left_fingertip_name),
                self.world.get_body_by_name(gripper.right_fingertip_name),
            ]
        )
        self.surface_contact.apply_to([self.robot.root])

    def _add_camera_and_light(self) -> None:
        """
        Attach a fixed camera framing the cartons, and a light over them, to the world
        root.
        """
        x_coordinates = [placed.x for placed in self.layout.objects]
        y_coordinates = [placed.y for placed in self.layout.objects]
        bounds = np.array(
            [
                [
                    min(x_coordinates) - self.camera_frame_margin,
                    min(y_coordinates) - self.camera_frame_margin,
                    self.table_top_z,
                ],
                [
                    max(x_coordinates) + self.camera_frame_margin,
                    max(y_coordinates) + self.camera_frame_margin,
                    self.table_top_z + self.milk_size.z + self.camera_frame_margin,
                ],
            ]
        )
        pose = MujocoCamera.overview_pose(
            bounds, distance_factor=self.camera_distance_factor
        )
        quaternion_xyzw = pose.to_quaternion().to_np().tolist()
        self.world.root.simulator_additional_properties.append(
            MujocoCamera(
                name=self.camera_name,
                body=self.world.root,
                position=pose.to_position().to_np()[:3].tolist(),
                quaternion=[quaternion_xyzw[3]] + quaternion_xyzw[:3],
            )
        )
        self.world.root.simulator_additional_properties.append(
            MujocoLight(
                name=self.light_name,
                body=self.world.root,
                directional=True,
                position=[float(bounds[1][0]), float(bounds[0][1]), 3.0],
                direction=[-0.3, 0.3, -1.0],
                ambient=[0.3, 0.3, 0.3],
                diffuse=[0.6, 0.6, 0.6],
            )
        )

    def _equip_robot(self) -> None:
        """
        Park both arms, open the picking gripper, close the other, and give every joint
        a position servo.
        """
        for arm in self.robot.get_arms():
            joint_state_of_type(arm, StaticJointState.PARK).apply_to(self.world)
            joint_state_of_type(arm.end_effector, GripperState.CLOSE).apply_to(
                self.world
            )
        joint_state_of_type(
            arm_of(self.robot, self.pick_arm).end_effector, GripperState.OPEN
        ).apply_to(self.world)
        self.world.notify_state_change()
        apply_gravity_compensation(self.world, self.robot)
        exclude_self_collision(self.world, self.robot)
        self.actuators = {
            **equip_arms_with_servos(self.world, self.robot),
            **equip_grippers_with_servos(self.world, self.robot),
        }
