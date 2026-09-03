from __future__ import annotations

from copy import deepcopy

from typing_extensions import List, Union, Optional

from krrood.adapters.json_serializer import list_like_classes
from coraplex.datastructures.dataclasses import Context
from coraplex.config.action_conf import ActionConfig
from coraplex.datastructures.enums import Arms
from coraplex.locations.backends import GiskardLocationBackend
from coraplex.locations.base import Location
from coraplex.locations.costmaps import OccupancyCostmap, RingCostmap, VisibilityCostmap
from coraplex.locations.pose_validator import (
    AreReachableBy,
    IsVisibleBy,
)
from coraplex.view_manager import ViewManager
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cabinet,
    Drawer,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body


def occupancy_location(target_pose: Pose, context: Context) -> Location:
    """
    Factory that creates a Location for robot base poses, does not have any validators.

    :param target_pose: Target pose around which robot base poses should be sampled
    :param context: Context of the plan in which the location should be created
    :returns: The Location for robot base poses
    """
    return Location(
        context, target_pose, OccupancyCostmap.default_map(context, target_pose), []
    )


def _grasp_to_clear(
    target_T_grasp: Optional[Pose],
    target_body: Optional[Body],
    context: Context,
    arm: Arms,
) -> Optional[Pose]:
    """
    The grasp whose geometry the robot has to reach around, in its body's own frame.

    Reaching for something reads the grasp off the target itself; carrying something to
    a place instead reads the grasp the gripper already has on it, which is what will
    have to clear the target.

    :param target_T_grasp: The grasp relative to the target, or ``None`` to grasp the
        target at its own origin.
    :param target_body: The body being reached for, when there is one.
    :param context: The context holding the robot and world.
    :param arm: The arm doing the reaching.
    :return: The grasp in its body's frame, or ``None`` when no body is involved.
    """
    end_effector = ViewManager.get_end_effector_view(arm, context.robot)
    if end_effector.held_body is not None:
        return end_effector.held_body_T_grasp
    if target_body is None:
        return None
    if target_T_grasp is None:
        return Pose(reference_frame=target_body)
    return target_T_grasp


def _grasp_frame_at(target_pose: Pose, target_T_grasp: Optional[Pose]) -> Pose:
    """
    Place a grasp frame at a target pose.

    :param target_pose: Where the grasped object is, or is going to be.
    :param target_T_grasp: The grasp frame relative to that object. ``None`` grasps the
        object at its own origin.
    :return: The grasp frame, in ``target_pose``'s own frame.
    """
    if target_T_grasp is None:
        return target_pose
    return (
        target_pose.to_homogeneous_matrix() @ target_T_grasp.to_homogeneous_matrix()
    ).to_pose()


def reachability_location(
    target: Union[Pose, Body],
    context: Context,
    arm: Arms,
    grasp_pose: Optional[Pose] = None,
    approach_clearance: float = ActionConfig.approach_clearance,
    retreat_distance: float = ActionConfig.retreat_distance,
) -> Location:
    """
    Factory method that creates a Location for robot poses from which the target can be
    picked up or placed.

    :param target: Target pose or body that should be reached by the robot
    :param context: The context in which to create the location
    :param arm: The arm with which to reach the target
    :param grasp_pose: The grasp frame with which to grasp the target, in the target's
        own frame. ``None`` grasps the target at its origin.
    :param approach_clearance: The gap left between the object and the gripper before
        the final approach.
    :param retreat_distance: How far the gripper rises after closing on the object.
    :returns: A location that is reachable from the target pose.
    """
    target_pose, target_body = (
        (target.global_pose, target) if isinstance(target, Body) else (target, None)
    )
    man = ViewManager.get_end_effector_view(arm, context.robot)
    grasp_frame = _grasp_frame_at(target_pose, grasp_pose)
    body_T_grasp = _grasp_to_clear(grasp_pose, target_body, context, arm)

    costmap = OccupancyCostmap.default_map(context, target_pose) & RingCostmap(
        resolution=0.02,
        width=200,
        height=200,
        std=15,
        distance=ViewManager.get_arm_view(arm, context.robot).approximate_length()
        * 0.66,  # That needs to be replaced with an estimate of the reachability space of the robot arms
        world=context.world,
        origin=target_pose,
    )
    return Location(
        context,
        target_pose,
        costmap,
        [
            AreReachableBy.for_grasp(
                grasp_frame,
                man,
                body_T_grasp,
                context=Context(
                    world=context.world,
                    robot=context.robot,
                    alternative_motion_mappings=context.alternative_motion_mappings,
                ),
                approach_clearance=approach_clearance,
                retreat_distance=retreat_distance,
            )
        ],
    )


def accessing_location(
    container: Union[Drawer, Cabinet], context: Context, arm: Arms
) -> Location:
    """
    Factory that creates a location for robot base poses for opening and closing
    container.

    :param container: The container that should be accessed
    :param context: Plan context in which to create the location
    :param arm: Arm with which to access the container
    :returns: A location that is accessible from the container.
    """
    return reachability_location(container.handle.root, context, arm)


def visibility_location(target: Union[Pose, Body], context: Context) -> Location:
    """
    Factory that creates a location for robot base poses from which the target is
    visible.

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
                context=Context(
                    world=context.world,
                    robot=context.robot,
                    alternative_motion_mappings=context.alternative_motion_mappings,
                ),
                target_pose=target_pose,
                target_body=target_body,
            )
        ],
    )


def giskard_reachability_location(
    target: Union[Pose, Body],
    context: Context,
    arm: Arms,
    grasp_pose: Optional[Pose] = None,
    approach_clearance: float = ActionConfig.approach_clearance,
    retreat_distance: float = ActionConfig.retreat_distance,
) -> Location:
    """
    Factory method that creates a location with a Giskard backend, the giskard backend
    uses the Giskard full-body control to find a robot pose.

    :param target: Target pose or body that should be reachable
    :param context: Plan context in which to create the location
    :param arm: Arm to use for reachability estimation
    :param grasp_pose: The grasp frame with which to grasp the target, in the target's
        own frame. ``None`` grasps the target at its origin.
    :param approach_clearance: The gap left between the object and the gripper before
        the final approach.
    :param retreat_distance: How far the gripper rises after closing on the object.
    :returns: A location that is reachable from the target pose, using Giskard for
        reachability estimation.
    """
    target_pose, target_body = (
        (target.global_pose, target) if isinstance(target, Body) else (target, None)
    )

    man = ViewManager.get_end_effector_view(arm, context.robot)
    grasp_frame = _grasp_frame_at(target_pose, grasp_pose)
    body_T_grasp = _grasp_to_clear(grasp_pose, target_body, context, arm)

    backend = GiskardLocationBackend(
        target,
        arm,
        grasp_frame,
        context.robot,
        context.world,
        approach_clearance=approach_clearance,
        retreat_distance=retreat_distance,
    )

    return Location(
        context,
        target_pose,
        backend,
        [
            AreReachableBy.for_grasp(
                grasp_frame,
                man,
                body_T_grasp,
                context=Context(
                    robot=context.robot,
                    world=context.world,
                    alternative_motion_mappings=context.alternative_motion_mappings,
                ),
                approach_clearance=approach_clearance,
                retreat_distance=retreat_distance,
            )
        ],
    )
