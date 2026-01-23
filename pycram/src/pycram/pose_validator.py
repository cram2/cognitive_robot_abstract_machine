import logging
import random
from copy import deepcopy

import numpy as np
from numpy.ma import masked_array
from skimage.measure import label

from giskardpy.executor import Executor
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.collision_checking.collision_detector import (
    CollisionCheck,
    Collision,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.spatial_computations.ik_solver import (
    MaxIterationsException,
    UnreachableException,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)
from typing_extensions import List, Union, Dict, Iterable, Optional, Iterator, Callable

from .costmaps import Costmap
from .datastructures.pose import PoseStamped, TransformStamped
from .failures import IKError, RobotInCollision
from .tf_transformations import quaternion_from_euler

logger = logging.getLogger("pycram")


def visibility_validator(
    robot: AbstractRobot, object_or_pose: Union[Body, PoseStamped], world: World
) -> bool:
    """
    This method validates if the robot can see the target position from a given
    pose candidate. The target position can either be a position, in world coordinate
    system, or an object in the World. The validation is done by shooting a
    ray from the camera to the target position and checking that it does not collide
    with anything else.

    :param robot: The robot object for which this should be validated
    :param object_or_pose: The target position or object for which the pose candidate should be validated.
    :param world: The world in which the visibility should be validated.
    :return: True if the target is visible for the robot, None in any other case.
    """
    if isinstance(object_or_pose, PoseStamped):
        gen_body = Body(
            name=PrefixedName("vist_test_obj", "pycram"),
            collision=ShapeCollection([Box(scale=Scale(0.1, 0.1, 0.1))]),
        )
        with world.modify_world():
            world.add_connection(
                Connection6DoF.create_with_dofs(
                    parent=world.root, child=gen_body, world=world
                )
            )
        gen_body.parent_connection.origin = object_or_pose.to_spatial_type()
    else:
        gen_body = object_or_pose
    r_t = world.ray_tracer
    camera = list(robot.neck.sensors)[0]
    ray = r_t.ray_test(
        camera.bodies[0].global_pose.to_position().to_np()[:3],
        gen_body.global_pose.to_position().to_np()[:3],
        multiple_hits=True,
    )

    hit_bodies = [b for b in ray[2] if not b in robot.bodies]

    if isinstance(object_or_pose, PoseStamped):
        with world.modify_world():
            world.remove_connection(gen_body.parent_connection)
            world.remove_kinematic_structure_entity(gen_body)

    return hit_bodies[0] == gen_body if len(hit_bodies) > 0 else False


def reachability_validator(
    target_pose: PoseStamped,
    tip_link: KinematicStructureEntity,
    robot_view: AbstractRobot,
    world: World,
    use_fullbody_ik: bool = False,
) -> bool:
    """
    Evaluates if a pose can be reached with the tip_link in the given world. This uses giskard motion state charts
    for testing.

    :param target_pose: The sequence of poses which the tip_link needs to reach
    :param tip_link: The tip link which should be used for reachability
    :param robot_view: The semantic annotation of the robot which should be evaluated for reachability
    :param world: The world in which the visibility should be validated.
    :param use_fullbody_ik: If true the base will be used in trying to reach the poses
    """
    return pose_sequence_reachability_validator(
        [target_pose], tip_link, robot_view, world, use_fullbody_ik
    )


def pose_sequence_reachability_validator(
    target_sequence: List[PoseStamped],
    tip_link: KinematicStructureEntity,
    robot_view: AbstractRobot,
    world: World,
    use_fullbody_ik: bool = False,
) -> bool:
    """
    Evaluates the pose sequence by executing the pose sequence with giskard motion state charts.

    :param target_sequence: The sequence of poses which the tip_link needs to reach
    :param tip_link: The tip link which should be used for reachability
    :param robot_view: The semantic annotation of the robot which should be evaluated for reachability
    :param world: The world in which the visibility should be validated.
    :param use_fullbody_ik: If true the base will be used in trying to reach the poses
    """
    old_state = deepcopy(world.state.data)
    root = robot_view.root if not use_fullbody_ik else world.root

    msc = MotionStatechart()
    msc.add_node(
        cart_sequence := Sequence(
            [
                CartesianPose(
                    root_link=root, tip_link=tip_link, goal_pose=pose.to_spatial_type()
                )
                for pose in target_sequence
            ]
        )
    )
    msc.add_node(EndMotion.when_true(cart_sequence))

    executor = Executor(
        world,
        controller_config=QPControllerConfig(
            target_frequency=50, prediction_horizon=4, verbose=False
        ),
    )
    executor.compile(msc)

    try:
        executor.tick_until_end()
    except TimeoutError:
        failed_nodes = []

        logger.debug(f"Timeout while executing pose sequence: {target_sequence}")
        return False
    finally:
        world.state.data = old_state
        world.notify_state_change()
    return True


def collision_check(
    robot: AbstractRobot, allowed_collision: List[Body], world: World
) -> List[Collision]:
    """
    This method checks if a given robot collides with any object within the world
    which it is not allowed to collide with.
    This is done checking iterating over every object within the world and checking
    if the robot collides with it. Careful the floor will be ignored.
    If there is a collision with an object that was not within the allowed collision
    list the function will raise a RobotInCollision exception.

    :param robot: The robot object in the (Bullet)World where it should be checked if it collides with something
    :param allowed_collision: dict of objects with which the robot is allowed to collide each object correlates to a list of links of which this object consists
    :param world: The world in which collision should be checked
    :raises: RobotInCollision if the robot collides with an object it is not allowed to collide with.
    """
    collision_matrix = create_collision_matrix(allowed_collision, world, robot)

    return world.collision_detector.check_collisions(collision_matrix)


def create_collision_matrix(
    ignore_collision_with: List[Body], world: World, robot: AbstractRobot
) -> List[CollisionCheck]:
    """
    CCreates a list of collision checks that should be performed

    :param ignore_collision_with: List of objects for which collision should be ignored
    :param world: The world in which the collision check should be performed
    :param robot: The robot for which the collision check should be performed
    :return: A list of collision checks that should be performed
    """
    collision_checks = []
    attached_bodies = set(robot.bodies) - set(
        world.get_kinematic_structure_entities_of_branch(robot.root)
    )
    allowed_collision_with = ignore_collision_with + list(attached_bodies)

    for robot_body in robot.bodies_with_enabled_collision:
        for world_body in world.bodies_with_enabled_collision:
            if (
                world_body in allowed_collision_with
                or robot_body in allowed_collision_with
                or world_body in robot.bodies
            ):
                continue
            collision_checks.append(CollisionCheck(robot_body, world_body, 0.01, world))

    return collision_checks
