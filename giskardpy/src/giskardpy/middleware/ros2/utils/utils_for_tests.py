from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from threading import Thread
from time import sleep
from typing import Tuple, Iterable

import numpy as np

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.python_interface import GiskardWrapperNode
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.collision_checking.collision_detector import (
    CollisionCheckingResult,
)
from semantic_digital_twin.collision_checking.collision_rules import AvoidAllCollisions
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import WorldEntityNotFoundError
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    OmniDrive,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    Scale,
    Cylinder,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)

# %% comparing spatial types


def compare_poses(
    actual_pose: cas.HomogeneousTransformationMatrix,
    desired_pose: cas.HomogeneousTransformationMatrix,
    decimal: int = 2,
) -> None:
    """
    Assert that two transformations describe the same position and orientation.

    :param decimal: Number of decimal places the two have to agree on.
    """
    compare_points(
        actual_point=actual_pose.to_position(),
        desired_point=desired_pose.to_position(),
        decimal=decimal,
    )
    compare_orientations(
        actual_orientation=actual_pose.to_quaternion(),
        desired_orientation=desired_pose.to_quaternion(),
        decimal=decimal,
    )


def compare_points(
    actual_point: cas.Point3,
    desired_point: cas.Point3,
    decimal: int = 2,
) -> None:
    """
    Assert that two points are the same up to an absolute tolerance.

    .. note:: The tolerance is absolute because coordinates near the origin of their
        reference frame are common, and a relative one degenerates into an exact
        comparison for them.

    :param decimal: Number of decimal places the two have to agree on.
    """
    np.testing.assert_array_almost_equal(
        actual_point, desired_point, decimal=decimal
    )


def compare_orientations(
    actual_orientation: cas.Quaternion,
    desired_orientation: cas.Quaternion,
    decimal: int = 2,
) -> None:
    """
    Assert that two quaternions describe the same orientation.

    A quaternion and its negation are the same orientation, so the desired one is
    flipped onto the hemisphere of the actual one before they are compared.

    :param decimal: Number of decimal places the two have to agree on.
    """
    actual_quaternion = actual_orientation.to_np()
    desired_quaternion = desired_orientation.to_np()
    if np.dot(actual_quaternion, desired_quaternion) < 0:
        desired_quaternion = -desired_quaternion
    np.testing.assert_array_almost_equal(
        actual_quaternion, desired_quaternion, decimal=decimal
    )


@dataclass
class GiskardTester(ABC):
    api: GiskardWrapperNode = field(init=False)
    giskard: Giskard = field(init=False)

    default_env_name: str | None = None

    def __post_init__(self):
        self.giskard = self.setup_giskard()
        self.giskard.setup()
        self.default_root = self.world.root
        self.motion_server_thread = Thread(
            target=self.giskard.motion_server.live, name="motion server"
        )
        self.motion_server_thread.start()
        self.wait_for_cycles(1)
        self.api = GiskardWrapperNode(node_name="tests")

    @abstractmethod
    def setup_giskard(self) -> Giskard: ...

    @property
    def world(self) -> World:
        return self.giskard.executor.context.world

    def get_odometry_joint(self) -> OmniDrive:
        return self.world.get_semantic_annotations_by_type(AbstractRobot)[0].drive

    def has_odometry_joint(self) -> bool:
        try:
            joint = self.get_odometry_joint()
        except WorldEntityNotFoundError:
            return False
        return isinstance(joint, (OmniDrive,))

    def wait_for_cycles(self, number_of_cycles: int = 5) -> None:
        """
        Block until the motion server completed that many more cycles.

        Control cycles count too, so this also returns while a goal is being executed.

        :param number_of_cycles: How many cycles to wait for.
        """
        cycle_counter = self.giskard.motion_server.cycle_counter
        first_cycle = cycle_counter.completed_cycles
        while cycle_counter.completed_cycles < first_cycle + number_of_cycles:
            sleep(0.001)

    def close(self):
        """
        Detach Giskard from the world so nothing of this test reacts to the next one.

        The ros node is destroyed between tests while worlds are kept alive, so a
        callback left registered here would publish on a node that is already gone.
        """
        self.giskard.close_world_model_ros_interface()

    #
    # BULLET WORLD #####################################################################################################
    #

    def detach_group(self, name: str) -> None:
        with self.api.world.modify_world():
            body = self.api.world.get_body_by_name(name)
            parent_T_connection = self.api.world.compute_forward_kinematics(
                self.api.world.root, body
            )
            new_connection = FixedConnection(
                parent=self.api.world.root,
                child=body,
                parent_T_connection_expression=parent_T_connection,
            )
            self.api.world.remove_connection(body.parent_connection)
            self.api.world.add_connection(new_connection)
        self.wait_for_cycles()

    def add_box_to_world(
        self,
        name: str,
        size: Tuple[float, float, float],
        pose: HomogeneousTransformationMatrix,
        parent_link: KinematicStructureEntity | None = None,
    ) -> None:
        parent_link = parent_link or self.api.world.root

        parent_T_pose = self.api.world.transform(
            spatial_object=pose,
            target_frame=parent_link,
        )
        with self.api.world.modify_world():
            box = Body(name=PrefixedName(name))
            box_shape = Box(scale=Scale(*size))
            box.collision.append(box_shape)
            box.visual.append(box_shape)

            connection = FixedConnection(
                parent=parent_link,
                child=box,
                parent_T_connection_expression=parent_T_pose,
            )
            self.api.world.add_connection(connection)
        self.wait_for_cycles()

    def add_cylinder_to_world(
        self,
        name: str,
        height: float,
        radius: float,
        pose: HomogeneousTransformationMatrix,
        parent_link: KinematicStructureEntity | None = None,
    ) -> None:
        parent_link = parent_link or self.api.world.root

        parent_T_pose = self.api.world.transform(
            spatial_object=pose,
            target_frame=parent_link,
        )
        with self.api.world.modify_world():
            cylinder = Body(name=PrefixedName(name))
            cylinder_shape = Cylinder(width=radius * 2, height=height)
            cylinder.collision.append(cylinder_shape)
            cylinder.visual.append(cylinder_shape)

            connection = FixedConnection(
                parent=parent_link,
                child=cylinder,
                parent_T_connection_expression=parent_T_pose,
            )
            self.api.world.add_connection(connection)
        self.wait_for_cycles()

    def add_urdf_to_world(
        self,
        name: str,
        urdf: str,
        pose: HomogeneousTransformationMatrix,
        parent_link: str | PrefixedName | None = None,
    ) -> None:
        if parent_link is None:
            parent_link = self.api.world.root
        else:
            parent_link = self.api.world.get_kinematic_structure_entity_by_name(
                parent_link
            )
        pr2_parser = URDFParser(urdf=urdf, prefix=name)
        world_with_pr2 = pr2_parser.parse()
        with self.api.world.modify_world():
            c_map_root = FixedConnection(
                parent=parent_link,
                child=world_with_pr2.root,
                parent_T_connection_expression=pose,
            )
            self.api.world.merge_world(world_with_pr2, root_connection=c_map_root)

        self.wait_for_cycles()

    def update_parent_link_of_group(
        self,
        name: str,
        parent_link: str | PrefixedName | None = None,
    ) -> None:
        with self.api.world.modify_world():
            body = self.api.world.get_kinematic_structure_entity_by_name(name)
            parent = self.api.world.get_kinematic_structure_entity_by_name(parent_link)
            self.api.world.move_branch(branch_root=body, new_parent=parent)
        self.wait_for_cycles()

    def compute_all_collisions(self) -> CollisionCheckingResult:
        collision_manager = self.world.collision_manager
        collision_manager.clear_temporary_rules()
        collision_manager.add_temporary_rule(
            AvoidAllCollisions(buffer_zone_distance=0.5)
        )
        collision_manager.update_collision_matrix()
        return collision_manager.compute_collisions()

    def check_cpi_geq(self, bodies: Iterable[Body], distance_threshold: float):
        collisions = self.compute_all_collisions()
        assert len(collisions.contacts) > 0
        for collision in collisions.contacts:
            if collision.body_a in bodies or collision.body_b in bodies:
                assert collision.distance >= distance_threshold, (
                    f"{collision.distance} < {distance_threshold} "
                    f"({collision.body_a} with {collision.body_b})"
                )

    def check_cpi_leq(
        self,
        bodies: Iterable[Body],
        distance_threshold: float,
    ):
        collisions = self.compute_all_collisions()
        min_contact = None
        for collision in collisions.contacts:
            if collision.body_a not in bodies and collision.body_b not in bodies:
                continue
            if min_contact is None or collision.distance <= min_contact.distance:
                min_contact = collision
        assert min_contact.distance <= distance_threshold, (
            f"{min_contact.distance} > {distance_threshold} "
            f"({min_contact.body_a} with {min_contact.body_b})"
        )
