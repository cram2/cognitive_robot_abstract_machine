from copy import deepcopy

from typing_extensions import List, Union, Optional

from krrood.adapters.json_serializer import list_like_classes
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.locations.backends import GiskardLocationBackend
from pycram.locations.base import Location
from pycram.locations.costmaps import OccupancyCostmap, RingCostmap, VisibilityCostmap
from pycram.locations.pose_validator import (
    ReachabilitySequenceValidator,
    VisibilityValidator,
)
from pycram.view_manager import ViewManager
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cabinet,
    Drawer,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


def _make_default_occupancy_costmap(context: Context, target: Pose) -> OccupancyCostmap:
    """
    Creates an occupancy cost map with the default parameters.

    :param context: The context for which the map is created.
    :param target: The target pose for the occupancy cost map.
    :returns: A occupancy cost map with default parameters.
    """
    ground_pose = deepcopy(target)
    ground_pose.z = 0

    base_bb = context.robot.base.bounding_box

    return OccupancyCostmap(
        resolution=0.02,
        width=200,
        height=200,
        world=context.world,
        distance_to_obstacle=(base_bb.depth / 2 + base_bb.width / 2) / 2,
        robot_view=context.robot,
        origin=ground_pose,
    )


def _get_object_in_hand(
    test_robot: AbstractRobot, test_world: World, arm: Arms = Arms.BOTH
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
    manipulators = (
        [manipulator] if not isinstance(manipulator, list_like_classes) else manipulator
    )
    objs = set()
    for man in manipulators:
        objs.update(
            test_world.get_kinematic_structure_entities_of_branch(man.tool_frame)
        )
        objs.remove(man.tool_frame)
    return objs.pop() if objs else None


def occupancy_location(target_pose: Pose, context: Context) -> Location:
    """
    Factory that creates a Location for robot base poses, does not have any validators

    :param target_pose: Target pose around which robot base poses should be sampled
    :praam context: Context of the plan in which the location should be created
    :returns: The Location for robot base poses
    """
    return Location(
        context,
        target_pose,
        _make_default_occupancy_costmap(context, target_pose),
    )


def reachability_location(
    target: Union[Pose, Body],
    context: Context,
    arm: Arms,
    grasp_description: GraspDescription = None,
) -> Location:
    """
    Factory method that creates a Location for robot poses from which the target can be picked up or placed.

    :param target: Target pose or body that should be reached by the robot
    :param context: The context in which to create the location
    :param arm: The arm with which to reach the target
    :param grasp_description: The grasp description with which to grasp the target
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
    costmap = _make_default_occupancy_costmap(context, target_pose) & RingCostmap(
        resolution=0.02,
        width=200,
        height=200,
        std=15,
        distance=0.4,  # That needs to be replaced with an estimate of the reachability space of the robot arms
        world=context.world,
        origin=target_pose,
    )
    return Location(
        context,
        target_pose,
        costmap,
        [
            ReachabilitySequenceValidator(
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
    costmap = _make_default_occupancy_costmap(context, target_pose) & VisibilityCostmap(
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
            VisibilityValidator(
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
            ReachabilitySequenceValidator(
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
