"""
The ten-milk clutter on Tracy's own table, built for MuJoCo from a
:class:`~experiments.causal_reasoning.tracy_clutter_picking.domain.ClutterSceneLayout`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from coraplex.datastructures.enums import Arms
from typing_extensions import List

from experiments.causal_reasoning.tracy_clutter_picking.domain import ClutterSceneLayout
from experiments.causal_reasoning.tracy_clutter_picking.tracy_mujoco_addons.live_motion import (
    arm_of,
)
from semantic_digital_twin.adapters.multi_sim import MujocoCamera, MujocoLight
from semantic_digital_twin.api import (
    BodySpecification,
    Connection6DoFSpecification,
    RobotSpecification,
)
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.contact import ContactParameters
from semantic_digital_twin.world_description.geometry import Box, Color, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class MilkClutterWorld:
    """
    Tracy, its table, and one layout's cartons standing on it, equipped to be simulated
    physically in MuJoCo.
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
        default_factory=ContactParameters.create_for_surface
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

    def __post_init__(self):
        self.world = World()
        with self.world.modify_world():
            self.world.add_kinematic_structure_entity(
                Body(name=PrefixedName(name="root", prefix="clutter"))
            )
        self.robot = RobotSpecification(
            Tracy,
            world_T_odom=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=self.mount_x, y=self.mount_y
            ),
        ).spawn(self.world)
        self.table_top_z = self.robot.table.top_z
        self._add_milks()
        self._add_camera_and_light()
        self._pose_robot()

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
        grasped object's contact stiffness and impedance.
        """
        return ContactParameters.create_for_grasped_object(
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
                self._stand_carton(
                    self.milk_name(index),
                    Point3(placed.x, placed.y, self.table_top_z + self.milk_size.z / 2),
                    placed.yaw,
                    (
                        self.target_color
                        if index == self.layout.target_index
                        else self.milk_color
                    ),
                )
            )
        gripper = arm_of(self.robot, self.pick_arm).end_effector
        ContactParameters(friction=self.grasp_contact.friction).apply_to(
            [gripper.left_fingertip, gripper.right_fingertip]
        )
        self.surface_contact.apply_to([self.robot.root])

    def _stand_carton(
        self, name: str, position: Point3, yaw: float, color: Color
    ) -> Body:
        """
        Stand one free carton in the world, with real collision geometry the fingers can
        close on and the grasp's own contact parameters.

        :param name: Name of the carton's body.
        :param position: Where the carton's centre starts, in the world root frame.
        :param yaw: Rotation of the carton about the vertical, in radians.
        :param color: Colour of the carton.
        :return: The carton.
        """
        specification = BodySpecification(
            name=name,
            shapes=ShapeCollection(
                [
                    Box(
                        origin=HomogeneousTransformationMatrix(),
                        scale=self.milk_size,
                        color=color,
                    )
                ]
            ),
            connection_specification=Connection6DoFSpecification(),
        )
        carton = specification.spawn(
            self.world,
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=position.x, y=position.y, z=position.z, yaw=yaw
            ),
        )
        self.grasp_contact.apply_to([carton])
        return carton

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
        self.world.root.add_simulator_property(
            MujocoCamera(
                name=self.camera_name,
                body=self.world.root,
                position=pose.to_position().to_np()[:3].tolist(),
                quaternion=[quaternion_xyzw[3]] + quaternion_xyzw[:3],
            )
        )
        self.world.root.add_simulator_property(
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

    def _pose_robot(self) -> None:
        """
        Park both arms, open the picking gripper and close the other.
        """
        for arm in self.robot.get_arms():
            arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(self.world)
            arm.end_effector.get_joint_state_by_type(GripperState.CLOSE).apply_to(
                self.world
            )
        arm_of(self.robot, self.pick_arm).end_effector.get_joint_state_by_type(
            GripperState.OPEN
        ).apply_to(self.world)
        self.world.notify_state_change()
