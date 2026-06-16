from copy import deepcopy

from typing_extensions import List, Union, Optional

from krrood.adapters.json_serializer import list_like_classes
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.locations.backends import GiskardLocationBackend
from coraplex.locations.base import Location
from coraplex.locations.costmaps import OccupancyCostmap, RingCostmap, VisibilityCostmap
from coraplex.locations.pose_validator import (
    AreReachableBy,
    IsVisibleBy,
)
from coraplex.view_manager import ViewManager
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cabinet,
    Drawer,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


def _get_object_in_hand(
    test_robot: AbstractRobot, test_world: World, arm: Arms
) -> Optional[Body]:
    """
    Util method that finds the object a robot is holding in the given arm.

    :param test_robot: The robot that is holding something
    :parm test_world: The world in which the robot is located
    :param arm: The arm that is holding something
    :returns: The body that the robot is holding in the given arm or None
    """

    manipulator = ViewManager.get_end_effector_view(
        arm,
        test_robot,
    )

    objs = set()
    objs.update(
        test_world.get_kinematic_structure_entities_of_branch(manipulator.tool_frame)
    )
    objs.remove(manipulator.tool_frame)
    return objs.pop() if objs else None


def occupancy_location(target_pose: Pose, context: Context) -> Location:
    """
    Factory that creates a Location for robot base poses, does not have any validators

    :param target_pose: Target pose around which robot base poses should be sampled
    :praam context: Context of the plan in which the location should be created
    :returns: The Location for robot base poses
    """
    return Location(
        context, target_pose, OccupancyCostmap.default_map(context, target_pose), []
    )


def reachability_location(
    target: Union[Pose, Body],
    context: Context,
    arm: Arms,
    grasp_description: GraspDescription = None,
    mean_distance_to_target: float = 0.6,
) -> Location:
    """
    Factory method that creates a Location for robot poses from which the target can be picked up or placed.

    :param target: Target pose or body that should be reached by the robot
    :param context: The context in which to create the location
    :param arm: The arm with which to reach the target
    :param grasp_description: The grasp description with which to grasp the target
    :param mean_distance_to_target: The mean distance between the base pose of the robot and the target pose in the xy-plane, can be imagined as a ring around the target pose from which poses are sampled. The mean distance is the radius of the ring.
    :returns: A location that is reachable from the target pose.
    """
    target_pose, target_body = (
        (target.global_pose, target) if isinstance(target, Body) else (target, None)
    )
    man = ViewManager.get_end_effector_view(arm, context.robot)

    grasp_description = grasp_description or GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        man,
    )
    costmap = OccupancyCostmap.default_map(context, target_pose) & RingCostmap(
        resolution=0.02,
        width=200,
        height=200,
        std=15,
        distance=mean_distance_to_target,  # That needs to be replaced with an estimate of the reachability space of the robot arms
        world=context.world,
        origin=target_pose,
    )
    return Location(
        context,
        target_pose,
        costmap,
        [
            AreReachableBy(
                pose_sequence=grasp_description._pose_sequence(
                    target_pose,
                    _get_object_in_hand(context.robot, context.world, arm)
                    or target_body,
                ),
                tip_link=man.tool_frame,
                world=context.world,
                robot=context.robot,
            )
        ],
    )


def accessing_location(
    container: Union[Drawer, Cabinet], context: Context, arm: Arms
) -> Location:
    """
    Factory that creates a location for robot base poses for opening and closing container.

    :param container: The container that should be accessed
    :param context: Plan context in which to create the location
    :param arm: Arm with which to access the container
    :returns: A location that is accessible from the container.
    """
    return reachability_location(
        container.handle.root.global_pose,
        context,
        arm,
        GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            ViewManager.get_end_effector_view(Arms.BOTH, context.robot),
        ),
    )


def visibility_location(target: Union[Pose, Body], context: Context) -> Location:
    """
    Factory that creates a location for robot base poses from which the target is visible.

    :param target: Target pose or body that should be visible
    :param context: Plan context in which to create the location
    :returns: A location that is visible from the target pose.
    """
    target_pose, target_body = (
        (target.global_pose, target) if isinstance(target, Body) else (target, None)
    )

    camera = context.robot.get_default_camera()
    costmap = OccupancyCostmap.default_map(context, target_pose) & VisibilityCostmap(
        min_height=camera.minimal_height,
        max_height=camera.maximal_height,
        world=context.world,
        width=200,
        height=200,
        resolution=0.02,
        origin=target_pose,
    )
    return Location(
        context,
        target_pose,
        costmap,
        [
            IsVisibleBy(
                world=context.world,
                robot=context.robot,
                target_pose=target_pose,
                target_body=target_body,
            )
        ],
    )


def giskard_reachability_location(
    target: Union[Pose, Body],
    context: Context,
    arm: Arms,
    grasp_description: GraspDescription = None,
) -> Location:
    """
    Factory method that creates a location with a Giskard backend, the giskard backend uses the Giskard full-body control
    to find a robot pose.

    :param target: Target pose or body that should be reachable
    :param context: Plan context in which to create the location
    :param arm: Arm to use for reachability estimation
    :param grasp_description: Grap that should be used for reachability estimation
    :returns: A location that is reachable from the target pose, using Giskard for reachability estimation.
    """
    target_pose, target_body = (
        (target.global_pose, target) if isinstance(target, Body) else (target, None)
    )

    man = ViewManager.get_end_effector_view(arm, context.robot)

    grasp_description = grasp_description or GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        man,
    )

    backend = GiskardLocationBackend(
        target, arm, grasp_description, context.robot, context.world
    )

    return Location(
        context,
        target_pose,
        backend,
        [
            AreReachableBy(
                pose_sequence=grasp_description._pose_sequence(
                    target_pose,
                    _get_object_in_hand(context.robot, context.world, arm)
                    or target_body,
                ),
                robot=context.robot,
                world=context.world,
                tip_link=man.tool_frame,
            )
        ],
    )
