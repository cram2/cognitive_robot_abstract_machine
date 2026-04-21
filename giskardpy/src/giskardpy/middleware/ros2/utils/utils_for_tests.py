import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from threading import Thread
from time import sleep
from typing import Tuple, Optional, List, Dict, Union, Iterable

import numpy as np
from angles import shortest_angular_distance
from geometry_msgs.msg import PoseStamped, Point, PointStamped, Quaternion, Pose

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.python_interface import GiskardWrapperNode
from giskardpy.tree.blackboard_utils import GiskardBlackboard
from semantic_digital_twin.adapters.ros import (
    Ros2ToSemDTConverter,
    SemDTToRos2Converter,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.collision_checking.collision_detector import (
    CollisionCheckingResult,
)
from semantic_digital_twin.collision_checking.collision_rules import AvoidAllCollisions
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import WorldEntityNotFoundError
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import (
    OmniDrive,
    FixedConnection,
    ActiveConnection1DOF,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    Scale,
    Sphere,
    Cylinder,
    Mesh,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)


def compare_poses(
    actual_pose: Union[cas.HomogeneousTransformationMatrix, Pose],
    desired_pose: Union[cas.HomogeneousTransformationMatrix, Pose],
    decimal: int = 2,
) -> None:
    if isinstance(actual_pose, cas.HomogeneousTransformationMatrix):
        actual_pose = SemDTToRos2Converter.convert(actual_pose).pose
    if isinstance(desired_pose, cas.HomogeneousTransformationMatrix):
        desired_pose = SemDTToRos2Converter.convert(desired_pose).pose
    compare_points(
        actual_point=actual_pose.position,
        desired_point=desired_pose.position,
        decimal=decimal,
    )
    compare_orientations(
        actual_orientation=actual_pose.orientation,
        desired_orientation=desired_pose.orientation,
        decimal=decimal,
    )


def compare_points(
    actual_point: Union[cas.Point3, Point],
    desired_point: Union[cas.Point3, Point],
    decimal: int = 2,
) -> None:
    if isinstance(actual_point, cas.Point3):
        actual_point = SemDTToRos2Converter.convert(actual_point).point
    if isinstance(desired_point, cas.Point3):
        desired_point = SemDTToRos2Converter.convert(desired_point).point
    np.testing.assert_almost_equal(actual_point.x, desired_point.x, decimal=decimal)
    np.testing.assert_almost_equal(actual_point.y, desired_point.y, decimal=decimal)
    np.testing.assert_almost_equal(actual_point.z, desired_point.z, decimal=decimal)


def compare_orientations(
    actual_orientation: Union[Quaternion, np.ndarray],
    desired_orientation: Union[Quaternion, np.ndarray],
    decimal: int = 2,
) -> None:
    if isinstance(actual_orientation, Quaternion):
        q1 = np.array(
            [
                actual_orientation.x,
                actual_orientation.y,
                actual_orientation.z,
                actual_orientation.w,
            ]
        )
    else:
        q1 = actual_orientation
    if isinstance(desired_orientation, Quaternion):
        q2 = np.array(
            [
                desired_orientation.x,
                desired_orientation.y,
                desired_orientation.z,
                desired_orientation.w,
            ]
        )
    else:
        q2 = desired_orientation
    try:
        np.testing.assert_almost_equal(q1[0], q2[0], decimal=decimal)
        np.testing.assert_almost_equal(q1[1], q2[1], decimal=decimal)
        np.testing.assert_almost_equal(q1[2], q2[2], decimal=decimal)
        np.testing.assert_almost_equal(q1[3], q2[3], decimal=decimal)
    except:
        np.testing.assert_almost_equal(q1[0], -q2[0], decimal=decimal)
        np.testing.assert_almost_equal(q1[1], -q2[1], decimal=decimal)
        np.testing.assert_almost_equal(q1[2], -q2[2], decimal=decimal)
        np.testing.assert_almost_equal(q1[3], -q2[3], decimal=decimal)


@dataclass
class GiskardTester(ABC):
    api: GiskardWrapperNode = field(init=False)
    giskard: Giskard = field(init=False)

    total_time_spend_giskarding: int = 0
    total_time_spend_moving: int = 0
    default_env_name: Optional[str] = None
    robot_names: List[PrefixedName] = field(default_factory=list)

    def __post_init__(self):
        self.async_loop = asyncio.new_event_loop()
        self.giskard = self.setup_giskard()
        self.giskard.setup()
        self.robot_names = [
            v.name
            for v in GiskardBlackboard().executor.context.world.get_semantic_annotations_by_type(
                AbstractRobot
            )
        ]
        self.default_root = GiskardBlackboard().executor.context.world.root

        self.original_number_of_links = len(
            GiskardBlackboard().executor.context.world.bodies
        )
        self.heart = Thread(target=GiskardBlackboard().tree.live, name="bt ticker")
        self.heart.start()
        self.wait_heartbeats(1)
        self.api = GiskardWrapperNode(node_name="tests")

    @abstractmethod
    def setup_giskard(self) -> Giskard: ...

    def get_odometry_joint(self) -> OmniDrive:
        return (
            GiskardBlackboard()
            .giskard.executor.context.world.get_semantic_annotations_by_type(
                AbstractRobot
            )[0]
            .drive
        )

    def compute_fk_pose(self, root_link: str, tip_link: str) -> PoseStamped:
        root_T_tip = GiskardBlackboard().executor.context.world.compute_forward_kinematics(
            root=GiskardBlackboard().executor.context.world.get_kinematic_structure_entity_by_name(
                root_link
            ),
            tip=GiskardBlackboard().executor.context.world.get_kinematic_structure_entity_by_name(
                tip_link
            ),
        )
        return SemDTToRos2Converter.convert(root_T_tip.to_pose())

    def compute_fk_point(self, root_link: str, tip_link: str) -> PointStamped:
        root_T_tip = (
            GiskardBlackboard()
            .executor.world.compute_forward_kinematics(
                root=GiskardBlackboard().executor.context.world.get_kinematic_structure_entity_by_name(
                    root_link
                ),
                tip=GiskardBlackboard().executor.context.world.get_kinematic_structure_entity_by_name(
                    tip_link
                ),
            )
            .to_position()
        )
        return SemDTToRos2Converter.convert(root_T_tip)

    def has_odometry_joint(self) -> bool:
        try:
            joint = self.get_odometry_joint()
        except WorldEntityNotFoundError as e:
            return False
        return isinstance(joint, (OmniDrive,))

    def wait_heartbeats(self, number=5):
        behavior_tree = GiskardBlackboard().tree
        c = behavior_tree.count
        while behavior_tree.count < c + number:
            sleep(0.001)

    def print_stats(self):
        giskarding_time = self.total_time_spend_giskarding
        if not GiskardBlackboard().tree_config.is_standalone():
            giskarding_time -= self.total_time_spend_moving
        rospy.node.get_logger().info(f"total time spend giskarding: {giskarding_time}")
        rospy.node.get_logger().info(
            f"total time spend moving: {self.total_time_spend_moving}"
        )

    def compare_joint_state(
        self,
        current_js: Dict[Union[str, PrefixedName], float],
        goal_js: Dict[Union[str, PrefixedName], float],
        decimal: int = 2,
    ):
        for joint_name in goal_js:
            goal = goal_js[joint_name]
            current = current_js[joint_name]
            connection: (
                ActiveConnection1DOF
            ) = GiskardBlackboard().executor.context.world.get_connection_by_name(
                joint_name
            )
            if not connection.dof.has_position_limits():
                np.testing.assert_almost_equal(
                    shortest_angular_distance(goal, current),
                    0,
                    decimal=decimal,
                    err_msg=f"{joint_name}: actual: {current} desired: {goal}",
                )
            else:
                np.testing.assert_almost_equal(
                    current,
                    goal,
                    decimal,
                    err_msg=f"{joint_name}: actual: {current} desired: {goal}",
                )

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
        self.wait_heartbeats()

    def add_box_to_world(
        self,
        name: str,
        size: Tuple[float, float, float],
        pose: HomogeneousTransformationMatrix,
        parent_link: Optional[KinematicStructureEntity] = None,
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
        self.wait_heartbeats()

    def add_sphere_to_world(
        self,
        name: str,
        radius: float = 1.0,
        pose: PoseStamped = None,
        parent_link: str | PrefixedName | None = None,
    ) -> None:
        if parent_link is None:
            parent_link = self.api.world.root
        else:
            parent_link = self.api.world.get_kinematic_structure_entity_by_name(
                parent_link
            )
        with self.api.world.modify_world():
            sphere = Body(name=PrefixedName(name))
            sphere_shape = Sphere(radius=radius)
            sphere.collision.append(sphere_shape)
            sphere.visual.append(sphere_shape)

            connection = FixedConnection(
                parent=parent_link,
                child=sphere,
                parent_T_connection_expression=Ros2ToSemDTConverter.convert(
                    pose, self.api.world
                ),
            )
            self.api.world.add_connection(connection)
        self.wait_heartbeats()

    def add_cylinder_to_world(
        self,
        name: str,
        height: float,
        radius: float,
        pose: PoseStamped = None,
        parent_link: str | PrefixedName | None = None,
    ) -> None:
        if parent_link is None:
            parent_link = self.api.world.root
        else:
            parent_link = self.api.world.get_kinematic_structure_entity_by_name(
                parent_link
            )
        parent_T_pose = self.api.world.transform(
            spatial_object=Ros2ToSemDTConverter.convert(pose, self.api.world),
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
        self.wait_heartbeats()

    def add_mesh_to_world(
        self,
        pose: PoseStamped,
        name: str = "meshy",
        mesh: str = "",
        parent_link: str | PrefixedName | None = None,
        scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        if parent_link is None:
            parent_link = self.api.world.root
        else:
            parent_link = self.api.world.get_kinematic_structure_entity_by_name(
                parent_link
            )
        parent_T_pose = self.api.world.transform(
            spatial_object=Ros2ToSemDTConverter.convert(pose, self.api.world),
            target_frame=parent_link,
        )
        with self.api.world.modify_world():
            mesh_body = Body(name=PrefixedName(name))
            mesh_shape = Mesh(filename=mesh, scale=Scale(*scale))
            mesh_body.collision.append(mesh_shape)
            mesh_body.visual.append(mesh_shape)

            connection = FixedConnection(
                parent=parent_link,
                child=mesh_body,
                parent_T_connection_expression=parent_T_pose,
            )
            self.api.world.add_connection(connection)
        self.wait_heartbeats()

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

        self.wait_heartbeats()

    def update_parent_link_of_group(
        self,
        name: str,
        parent_link: str | PrefixedName | None = None,
    ) -> None:
        with self.api.world.modify_world():
            body = self.api.world.get_kinematic_structure_entity_by_name(name)
            parent = self.api.world.get_kinematic_structure_entity_by_name(parent_link)
            self.api.world.move_branch(branch_root=body, new_parent=parent)
        self.wait_heartbeats()

    def compute_all_collisions(self) -> CollisionCheckingResult:
        collision_manager = GiskardBlackboard().executor.context.world.collision_manager
        collision_manager.clear_temporary_rules()
        collision_manager.add_temporary_rule(
            AvoidAllCollisions(buffer_zone_distance=0.5)
        )
        collision_manager.update_collision_matrix()
        return collision_manager.compute_collisions()

    def check_cpi_geq(
        self,
        bodies: Iterable[Body],
        distance_threshold: float,
        check_external: bool = True,
        check_self: bool = True,
    ):
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
        check_external: bool = True,
        check_self: bool = True,
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
