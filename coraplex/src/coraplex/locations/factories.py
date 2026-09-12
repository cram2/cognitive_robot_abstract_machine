from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from typing_extensions import Iterable, Iterator, List, Optional, Union

from krrood.adapters.json_serializer import list_like_classes
from coraplex.datastructures.dataclasses import Context
from coraplex.config.action_conf import ActionConfig
from coraplex.datastructures.enums import Arms
from coraplex.locations.backends import GiskardLocationBackend
from coraplex.locations.base import Location
from coraplex.locations.sampling import CostmapSamplingStrategy, WeightedByRating
from coraplex.locations.costmaps import OccupancyCostmap, RingCostmap, VisibilityCostmap
from coraplex.locations.pose_validator import (
    AreReachableBy,
    IsObjectReachableBy,
    IsVisibleBy,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cabinet,
    Drawer,
)
from semantic_digital_twin.semantic_annotations.mixins import HasGraspPoses
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


def reachability_location(
    body: Body,
    context: Context,
    arm: Arms,
    grasp_pose: Optional[Pose] = None,
    destination: Optional[Pose] = None,
    approach_clearance: float = ActionConfig.approach_clearance,
    retreat_distance: float = ActionConfig.retreat_distance,
    reach_fraction: float = ActionConfig.reach_fraction,
    sampling_strategy: Optional[CostmapSamplingStrategy] = None,
) -> Location:
    """
    Factory method that creates a Location for robot poses from which a body can be
    grasped where it is, or released where it is going to be.

    :param body: The body the gripper grasps or holds.
    :param context: The context in which to create the location
    :param arm: The arm with which to reach the body
    :param grasp_pose: The grasp frame on the body, in the body's own frame. ``None``
        grasps the body at its origin.
    :param destination: Where the body is going to be, such as where a carried body is
        placed. ``None`` reaches the body where it is. A body reached at a destination
        is released there, which runs the approach backwards, so the check follows it
        backwards too.
    :param approach_clearance: The gap left between the object and the gripper before
        the final approach.
    :param retreat_distance: How far the gripper rises after closing on the object.
    :param reach_fraction: The fraction of the arm's length the robot stands off the
        target by.
    :returns: A location from which the grasp can be reached.
    """
    body_T_grasp = grasp_pose or Pose(reference_frame=body)
    target_pose = destination or body.global_pose
    releases_the_body = destination is not None
    return Location(
        context,
        target_pose,
        OccupancyCostmap.default_map(context, target_pose)
        & RingCostmap.from_arm_reach_distance(
            context, arm, target_pose, reach_fraction=reach_fraction
        ),
        [
            AreReachableBy.for_grasp(
                target_pose.to_homogeneous_matrix() @ body_T_grasp,
                arm,
                body_T_grasp=body_T_grasp,
                context=context,
                reverse=releases_the_body,
                approach_clearance=approach_clearance,
                retreat_distance=retreat_distance,
            )
        ],
        sampling_strategy=sampling_strategy
        or WeightedByRating(seed=context.sampling_seed),
    )


def grasping_location(
    graspable: HasGraspPoses,
    context: Context,
    arm: Arms,
    approach_clearance: float = ActionConfig.approach_clearance,
    retreat_distance: float = ActionConfig.retreat_distance,
    sampling_strategy: Optional[CostmapSamplingStrategy] = None,
) -> Location:
    """
    Factory that creates a Location for robot poses from which the object can be grasped
    somehow, rather than from which one particular grasp can be reached.

    A grasp is only reachable from somewhere, so settling on one before a standing pose
    is known picks it from wherever the robot happens to be. This asks the other way
    round: a pose qualifies when any of the object's grasps can be reached from it, and
    the validator keeps the one that was, in
    :attr:`~coraplex.locations.pose_validator.IsObjectReachableBy.reachable_grasp`.

    :param graspable: The annotation of the object that should be grasped.
    :param context: The context in which to create the location.
    :param arm: The arm with which to grasp the object.
    :param approach_clearance: The gap left between the object and the gripper before
        the final approach.
    :param retreat_distance: How far the gripper rises after closing on the object.
    :returns: A location from which the object can be grasped.
    """
    target_pose = graspable.root.global_pose
    return Location(
        context,
        target_pose,
        OccupancyCostmap.default_map(context, target_pose)
        & RingCostmap.from_arm_reach_distance(context, arm, target_pose),
        [
            IsObjectReachableBy(
                context=context,
                arm=arm,
                graspable=graspable,
                approach_clearance=approach_clearance,
                retreat_distance=retreat_distance,
            )
        ],
        sampling_strategy=sampling_strategy
        or WeightedByRating(seed=context.sampling_seed),
    )


@dataclass
class ReachableGrasps(Iterable[Pose]):
    """
    The grasps of an object that some standing pose reaches, worked out when they are
    asked for rather than when the plan is built.

    Which grasps qualify depends on where the robot may stand and on where everything
    else has got to, so answering while the plan is still being built answers about a
    world the action will not run in. Used as the domain of a ``grasp_pose`` variable,
    this is asked once the underspecified action grounds, during execution.

    .. warning::
        :meth:`__iter__` must stay a generator. The domain is wrapped rather than
        consumed by :func:`~krrood.entity_query_language.factories.variable`, so a
        generator is what defers the search to the first ``next``; building the grasps
        eagerly would put the staleness straight back.
    """

    graspable: HasGraspPoses
    """
    The annotation of the object that should be grasped.
    """

    context: Context
    """
    The context the reaching is judged in.
    """

    arm: Arms
    """
    The arm that should do the grasping.
    """

    approach_clearance: float = ActionConfig.approach_clearance
    """
    The gap left between the object and the gripper before the final approach.
    """

    retreat_distance: float = ActionConfig.retreat_distance
    """
    How far the gripper rises after closing on the object.
    """

    def __iter__(self) -> Iterator[Pose]:
        location = grasping_location(
            self.graspable,
            self.context,
            self.arm,
            approach_clearance=self.approach_clearance,
            retreat_distance=self.retreat_distance,
        )
        (validator,) = location.validators
        for _ in location:
            yield validator.reachable_grasp


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
    return reachability_location(
        container.handle.root,
        context,
        arm,
        reach_fraction=ActionConfig.accessing_reach_fraction,
    )


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
                context=context,
                target_pose=target_pose,
                target_body=target_body,
            )
        ],
    )


def giskard_reachability_location(
    body: Body,
    context: Context,
    arm: Arms,
    grasp_pose: Optional[Pose] = None,
    destination: Optional[Pose] = None,
    approach_clearance: float = ActionConfig.approach_clearance,
    retreat_distance: float = ActionConfig.retreat_distance,
) -> Location:
    """
    Factory method that creates a location with a Giskard backend, the giskard backend
    uses the Giskard full-body control to find a robot pose.

    :param body: The body the gripper grasps or holds.
    :param context: Plan context in which to create the location
    :param arm: Arm to use for reachability estimation
    :param grasp_pose: The grasp frame on the body, in the body's own frame. ``None``
        grasps the body at its origin.
    :param destination: Where the body is going to be, such as where a carried body is
        placed. ``None`` reaches the body where it is.
    :param approach_clearance: The gap left between the object and the gripper before
        the final approach.
    :param retreat_distance: How far the gripper rises after closing on the object.
    :returns: A location from which the grasp can be reached, using Giskard for
        reachability estimation.
    """
    body_T_grasp = grasp_pose or Pose(reference_frame=body)
    target_pose = destination or body.global_pose
    grasp_frame = target_pose.to_homogeneous_matrix() @ body_T_grasp
    releases_the_body = destination is not None

    backend = GiskardLocationBackend(
        target_pose,
        arm,
        grasp_frame,
        context.robot,
        context.world,
        body_T_grasp=body_T_grasp,
        contact_bodies=[body],
        reverse=releases_the_body,
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
                arm,
                body_T_grasp=body_T_grasp,
                context=context,
                reverse=releases_the_body,
                approach_clearance=approach_clearance,
                retreat_distance=retreat_distance,
            )
        ],
    )
