from __future__ import annotations

import itertools
from copy import deepcopy
from typing import Optional, List

import trimesh.sample
from krrood.entity_query_language.entity import (
    variable,
    entity,
    and_,
    not_,
    contains,
)
from krrood.entity_query_language.entity_result_processors import an, the

from ..collision_checking.collision_detector import Collision, CollisionCheck
from ..collision_checking.trimesh_collision_detector import TrimeshCollisionDetector
from ..robots.abstract_robot import AbstractRobot, ParallelGripper
from ..spatial_computations.raytracer import RayTracer
from ..spatial_types import HomogeneousTransformationMatrix
from ..world_description.world_entity import Body


def robot_in_collision(
    robot: AbstractRobot,
    ignore_collision_with: Optional[List[Body]] = None,
    threshold: float = 0.001,
) -> List[Collision]:
    """
    Check if the robot collides with any object in the world at the given pose.

    :param robot: The robot object
    :param ignore_collision_with: A list of objects to ignore collision with
    :param threshold: The threshold for contact detection
    :return: True if the robot collides with any object, False otherwise
    """

    if ignore_collision_with is None:
        ignore_collision_with = []

    body = variable(type_=Body, domain=robot._world.bodies_with_enabled_collision)
    possible_collisions_bodies = an(
        entity(body).where(
            and_(
                not_(contains(robot.bodies, body)),
                not_(contains(ignore_collision_with, body)),
            ),
        ),
    )
    possible_collisions_bodies = possible_collisions_bodies.evaluate()

    tcd = TrimeshCollisionDetector(robot._world)

    collisions = tcd.check_collisions(
        {
            CollisionCheck(robot_body, collision_body, threshold, robot._world)
            for robot_body, collision_body in itertools.product(
                robot.bodies_with_collisions, possible_collisions_bodies
            )
        }
    )
    return collisions


def robot_holds_body(robot: AbstractRobot, body: Body) -> bool:
    """
    Check if a robot is holding an object.

    :param robot: The robot object
    :param body: The body to check if it is picked
    :return: True if the robot is holding the object, False otherwise
    """
    g = variable(ParallelGripper, robot._world.semantic_annotations)
    grippers = an(
        entity(g).where(
            g._robot == robot,
        )
    )

    return any(
        [is_body_in_gripper(body, gripper) > 0.0 for gripper in grippers.evaluate()]
    )


def blocking(
    pose: HomogeneousTransformationMatrix,
    root: Body,
    tip: Body,
) -> List[Collision]:
    """
    Get the bodies that are blocking the robot from reaching a given position.
    The blocking are all bodies that are in collision with the robot when reaching for the pose.

    :param pose: The pose to reach
    :param root: The root of the kinematic chain.
    :param tip: The threshold between the end effector and the position.
    :return: A list of bodies the robot is in collision with when reaching for the specified object or None if the pose or object is not reachable.
    """
    result = root._world.compute_inverse_kinematics(
        root=root, tip=tip, target=pose, max_iterations=1000
    )
    with root._world.modify_world():
        for dof, state in result.items():
            root._world.state[dof.id].position = state

    r = variable(AbstractRobot, root._world.semantic_annotations)
    robot = the(
        entity(r).where(
            contains(r.bodies, tip),
        )
    )
    return robot_in_collision(robot.evaluate(), [])


def is_body_in_gripper(
    body: Body, gripper: ParallelGripper, sample_size: int = 100
) -> float:
    """
    Check if the body in the gripper.

    This method samples random rays between the finger and the thumb and returns the marginal probability that the rays
    intersect.

    :param body: The body for which the check should be done.
    :param gripper: The gripper for which the check should be done.
    :param sample_size: The number of rays to sample.

    :return: The percentage of rays between the fingers that hit the body.
    """

    # Retrieve meshes in local frames
    thumb_mesh = gripper.thumb.tip.collision.combined_mesh.copy()
    finger_mesh = gripper.finger.tip.collision.combined_mesh.copy()
    body_mesh = body.collision.combined_mesh.copy()

    # Transform copies of the meshes into the world frame
    body_mesh.apply_transform(body.global_pose.to_np())
    thumb_mesh.apply_transform(gripper.thumb.tip.global_pose.to_np())
    finger_mesh.apply_transform(gripper.finger.tip.global_pose.to_np())

    # get random points from thumb mesh
    finger_points = trimesh.sample.sample_surface(finger_mesh, sample_size)[0]
    thumb_points = trimesh.sample.sample_surface(thumb_mesh, sample_size)[0]

    rt = RayTracer(gripper._world)
    rt.update_scene()

    points, index_ray, bodies = rt.ray_test(finger_points, thumb_points)
    return len([b for b in bodies if b == body]) / sample_size
