from __future__ import annotations

from typing import Optional, List

import trimesh.sample

from krrood.entity_query_language.factories import (
    entity,
    variable,
    contains,
    an,
    the,
)
from krrood.entity_query_language.predicate import symbolic_function
from semantic_digital_twin.collision_checking.collision_detector import (
    ClosestPoints,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
    AvoidExternalCollisions,
    AllowSelfCollisions,
)
from semantic_digital_twin.reasoning.predicates import is_place_occupied
from semantic_digital_twin.robots.abstract_robot import (
    AbstractRobot,
    Manipulator,
)
from semantic_digital_twin.robots.robot_parts import ParallelGripper
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import BoundingBox
from semantic_digital_twin.world_description.world_entity import Body


@symbolic_function
def robot_in_collision(
    robot: AbstractRobot,
    ignore_collision_with: Optional[List[Body]] = None,
    threshold: float = 0.001,
) -> List[ClosestPoints]:
    """
    Check if the robot collides with any object in the world at the given pose.

    :param robot: The robot object
    :param ignore_collision_with: A list of objects to ignore collision with
    :param threshold: The threshold for contact detection
    :return: True if the robot collides with any object, False otherwise
    """

    if ignore_collision_with is None:
        ignore_collision_with = []

    world = robot._world

    with world.modify_world():
        world.collision_manager.clear_temporary_rules()
        world.collision_manager.add_temporary_rule(
            AvoidExternalCollisions(
                buffer_zone_distance=threshold,
                robot=robot,
            )
        )
        world.collision_manager.add_temporary_rule(AllowSelfCollisions(robot=robot))
        world.collision_manager.add_temporary_rule(
            AllowCollisionBetweenGroups(
                body_group_a=robot.bodies, body_group_b=ignore_collision_with
            )
        )
    world.collision_manager.update_collision_matrix()

    collisions = world.collision_manager.compute_collisions()

    return collisions.contacts


@symbolic_function
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


@symbolic_function
def blocking(
    pose: HomogeneousTransformationMatrix,
    root: Body,
    tip: Body,
) -> List[ClosestPoints]:
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
    return robot_in_collision(robot.first(), [])


@symbolic_function
def bodies_in_gripper(gripper: ParallelGripper, sample_size: int = 100) -> List[Body]:
    """
    Gets all bodies which are between the finger of the gripper.
    This method uses samples of rays which are cast between the finger

    :param gripper: The gripper for which the check should be done.
    :param sample_size: The number of rays to sample.
    """
    # Retrieve meshes in local frames
    thumb_mesh = gripper.thumb.tip.collision.combined_mesh.copy()
    finger_mesh = gripper.finger.tip.collision.combined_mesh.copy()

    # Transform copies of the meshes into the world frame
    # body_mesh.apply_transform(body.global_transform.to_np())
    thumb_mesh.apply_transform(gripper.thumb.tip.global_transform.to_np())
    finger_mesh.apply_transform(gripper.finger.tip.global_transform.to_np())

    # get random points from thumb mesh
    finger_points = trimesh.sample.sample_surface(finger_mesh, sample_size)[0]
    thumb_points = trimesh.sample.sample_surface(thumb_mesh, sample_size)[0]

    rt = gripper._world.ray_tracer
    rt.update_scene()

    points, index_ray, bodies = rt.ray_test(finger_points, thumb_points)
    return list(set(bodies) - set(gripper.finger.bodies) - set(gripper.thumb.bodies))


@symbolic_function
def is_body_in_gripper(
    body: Body, gripper: Manipulator, sample_size: int = 100
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
    bodies = bodies_in_gripper(gripper, sample_size)
    return len([b for b in bodies if b == body]) / sample_size


@symbolic_function
def is_gripper_holding_something(gripper: Manipulator) -> bool:
    """
    Check if the gripper is holding something.

    :param gripper: The gripper for which the check should be done.
    :return: True if there is a body mounted beneath the gripper in the kinematic chain.
    """
    bodies_under_tcp = gripper._world.get_kinematic_structure_entities_of_branch(
        gripper.tool_frame
    )
    return len(bodies_under_tcp) > 0


@symbolic_function
def is_pose_free_for_robot(
    robot: AbstractRobot, pose: HomogeneousTransformationMatrix
) -> bool:
    robot_bb = robot.mobile_base.bounding_box.transform_to_origin(
        robot.root.global_pose
    )
    target_bb = BoundingBox(
        robot_bb.min_x,
        robot_bb.min_y,
        robot_bb.min_z,
        robot_bb.max_x,
        robot_bb.max_y,
        robot_bb.max_z,
        pose,
    )
    return not is_place_occupied(target_bb, robot._world, robot.bodies_with_collision)
