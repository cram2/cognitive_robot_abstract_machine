"""
Semantic annotation for the UFACTORY xArm 5.

A table-mounted 5-DoF serial arm with a bare end-effector flange and a placeholder
camera. Mirrors the structure of the other robots in this package (see ``pr2.py`` /
``stretch.py``): each part is a leaf-first class implementing the three abstract
methods, and the robot class ties them together.

.. note:: ``get_ros_file_path`` returns a ``package://xarm_description/...`` path. For
   it to resolve, ``xarm_description`` must be a discoverable ROS package (installed in
   the overlay workspace, exactly like ``iai_pr2_description`` is for the PR2). The
   self-collision matrix, by contrast, ships with this package under
   ``resources/collision_configs/xarm5.srdf`` and is always available.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import List, Self

from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidExternalCollisions,
    SelfCollisionMatrixRule,
)
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import HasOneArm, HasSensors
from semantic_digital_twin.robots.robot_parts import (
    AbstractRobot,
    Arm,
    Camera,
    EndEffector,
)
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.world_description.world_entity import KinematicStructureEntity


@dataclass(eq=False)
class XArm5Camera(Camera):
    """
    Placeholder camera at the ``link_eef`` flange.

    The xArm 5 URDF has no camera link, so this stands in so that a robot with a camera
    can be validated. Replace ``root`` with a real optical frame and tune the optical
    parameters once a camera is actually mounted.
    """

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "link_eef"),
            forward_facing_axis=Vector3.Z(),
            minimal_height=0.0,
            maximal_height=1.0,
            field_of_view=FieldOfView(horizontal_angle=1.047, vertical_angle=0.785),
            default_camera=True,
        )

    def setup_hardware_interfaces(self):
        return None

    def setup_joint_states(self) -> List[JointState]:
        return []


@dataclass(eq=False)
class XArm5Gripper(EndEffector):
    """
    Fingerless end-effector: a bare flange with no gripper.

    ``root`` and ``tool_frame`` are both ``link_eef`` because there is no separate
    tool-center-point link.
    """

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "link_eef"),
            tool_frame=robot_root._world.get_body_in_branch_by_name(
                robot_root, "link_eef"
            ),
            front_facing_orientation=Quaternion(0, 0, 0, 1),
        )

    def setup_hardware_interfaces(self):
        return None

    def setup_joint_states(self) -> List[JointState]:
        return []


@dataclass(eq=False)
class XArm5Arm(Arm[XArm5Gripper]):
    """
    The arm chain from ``link_base`` (root) to ``link5`` (tip), carrying the fingerless
    gripper as its end-effector.
    """

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "link_base"),
            tip=robot_root._world.get_body_in_branch_by_name(robot_root, "link5"),
        )

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        arm_park = JointState.from_mapping(
            name=PrefixedName("arm_park", prefix=self.name.name),
            mapping=dict(
                zip(self.active_connections, [0.0] * len(self.active_connections))
            ),
            state_type=StaticJointState.PARK,
        )
        return [arm_park]


@dataclass(eq=False)
class XArm5(AbstractRobot, HasOneArm[XArm5Arm], HasSensors[XArm5Camera]):
    """
    UFACTORY xArm 5: a table-mounted 5-DoF serial arm with a bare end-effector flange
    and a placeholder camera. It owns the arm and the camera directly (no MobileBase,
    Torso or Neck).
    """

    @classmethod
    def get_ros_file_path(cls) -> str:
        return "package://xarm_description/urdf/xarm5.urdf"

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "link_base"

    def _setup_collision_rules(self):
        srdf_path = os.path.join(
            Path(files("semantic_digital_twin")).parent.parent,
            "resources",
            "collision_configs",
            "xarm5.srdf",
        )
        self._world.collision_manager.add_ignore_collision_rule(
            SelfCollisionMatrixRule.from_collision_srdf(srdf_path, self._world)
        )
        self._world.collision_manager.add_default_rule(
            AvoidExternalCollisions(
                buffer_zone_distance=0.05, violated_distance=0.0, robot=self
            )
        )
