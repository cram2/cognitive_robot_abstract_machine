from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from typing_extensions import Iterable, Iterator, List, Optional, Union

from krrood.adapters.json_serializer import list_like_classes
from coraplex.datastructures.dataclasses import Context
from coraplex.config.action_conf import ActionConfig
from coraplex.locations.backends import GiskardLocationBackend
from coraplex.locations.base import Location
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
from semantic_digital_twin.robots.robot_parts import Arm, EndEffector
from semantic_digital_twin.semantic_annotations.mixins import HasGraspPoses
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body


def occupancy_location(target_pose: Pose, context: Context) -> Location:
    """
    Where the robot can stand around a target without standing in anything.

    Nothing else is asked of a candidate: the poses are offered as the map has them.

    :param target_pose: The pose the standing poses are drawn around.
    :param context: The context in which to create the location.
    :returns: A location of poses clear of the surroundings.
    """
    return Location(
        context=context,
        target_pose=target_pose,
        generator=OccupancyCostmap.default_map(context=context, target=target_pose),
    )


def reachability_location(
    body: Body,
    context: Context,
    arm: Arm,
    grasp_pose: Optional[Pose] = None,
    destination: Optional[Pose] = None,
    approach_clearance: float = ActionConfig.approach_clearance,
    retreat_distance: float = ActionConfig.retreat_distance,
    reach_fraction: float = ActionConfig.reach_fraction,
) -> Location:
    """
    Checks one grasp the caller already chose: where can the robot stand to reach it?

    .. note::
        - *Grasp*: given by the caller as ``grasp_pose``; no other grasp is ever tried.
        - *Result*: standing poses only.
        - To have the grasp chosen as well, use :func:`grasping_location`.

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
    :returns: Standing poses from which ``grasp_pose`` can be reached, or released at
        ``destination``.
    """
    body_T_grasp = grasp_pose or Pose(reference_frame=body)
    target_pose = destination or body.global_pose
    releases_the_body = destination is not None
    occupancy_costmap = OccupancyCostmap.default_map(
        context=context, target=target_pose
    )
    ring_costmap = RingCostmap.from_arm_reach_distance(
        context=context, arm=arm, origin=target_pose, reach_fraction=reach_fraction
    )
    final_costmap = occupancy_costmap & ring_costmap
    return Location(
        context=context,
        target_pose=target_pose,
        generator=final_costmap,
        validator=AreReachableBy.for_grasp(
            grasp_pose=target_pose.to_homogeneous_matrix() @ body_T_grasp,
            arm=arm,
            body_T_grasp=body_T_grasp,
            context=context,
            reverse=releases_the_body,
            approach_clearance=approach_clearance,
            retreat_distance=retreat_distance,
        ),
    )


def grasping_location(
    graspable: HasGraspPoses,
    context: Context,
    arm: Arm,
    approach_clearance: float = ActionConfig.approach_clearance,
    retreat_distance: float = ActionConfig.retreat_distance,
) -> Location:
    """
    Chooses the grasp as well as the standing pose: only the object is given, and every
    grasp it offers is tried.

    .. note::
        - *Grasp*: chosen here, from the grasps ``graspable`` offers; the one chosen
          for the current pose is on the validator's
          :attr:`~coraplex.locations.pose_validator.IsObjectReachableBy.reachable_grasp`.
        - *Result*: standing poses, each paired with the grasp it was found for.
        - For a grasp the caller already chose, use :func:`reachability_location`.

    :param graspable: The annotation of the object that should be grasped.
    :param context: The context in which to create the location.
    :param arm: The arm with which to grasp the object.
    :param approach_clearance: The gap left between the object and the gripper before
        the final approach.
    :param retreat_distance: How far the gripper rises after closing on the object.
    :returns: Standing poses from which at least one of the object's grasps can be
        reached.
    """
    target_pose = graspable.root.global_pose
    occupancy_costmap = OccupancyCostmap.default_map(
        context=context, target=target_pose
    )
    ring_costmap = RingCostmap.from_arm_reach_distance(
        context=context, arm=arm, origin=target_pose
    )
    final_costmap = occupancy_costmap & ring_costmap
    return Location(
        context=context,
        target_pose=target_pose,
        generator=final_costmap,
        validator=IsObjectReachableBy(
            context=context,
            arm=arm,
            graspable=graspable,
            approach_clearance=approach_clearance,
            retreat_distance=retreat_distance,
        ),
    )


@dataclass
class ReachableGrasps(Iterable[Pose]):
    """
    The grasps of an object that some standing pose reaches, worked out when they are
    asked for rather than when the plan is built.

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

    arm: Arm[EndEffector]
    """
    The arm that should do the grasping.

    Written with its end effector type, since a bound generic is what the ORM maps a
    field of; an unparameterized one is skipped and the arm is then not persisted.
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
            graspable=self.graspable,
            context=self.context,
            arm=self.arm,
            approach_clearance=self.approach_clearance,
            retreat_distance=self.retreat_distance,
        )
        for _ in location:
            yield location.validator.reachable_grasp


def accessing_location(
    container: Union[Drawer, Cabinet], context: Context, arm: Arm
) -> Location:
    """
    Where the robot can stand to open or close a container by its handle.

    The same question :func:`reachability_location` answers about the handle, asked from
    the closer stand-off distance that pulling a container needs.

    :param container: The container to be opened or closed.
    :param context: The context in which to create the location.
    :param arm: The arm that works the handle.
    :returns: A location from which the handle can be reached.
    """
    return reachability_location(
        body=container.handle.root,
        context=context,
        arm=arm,
        reach_fraction=ActionConfig.accessing_reach_fraction,
    )


def visibility_location(target: Union[Pose, Body], context: Context) -> Location:
    """
    Where the robot can stand to see a target with its camera.

    :param target: The pose or body that should be visible.
    :param context: The context in which to create the location.
    :returns: A location from which the target is in view.
    """
    target_pose, target_body = (
        (target.global_pose, target) if isinstance(target, Body) else (target, None)
    )

    camera = context.robot.get_default_camera()
    occupancy_costmap = OccupancyCostmap.default_map(
        context=context, target=target_pose
    )
    visibility_costmap = VisibilityCostmap(
        minimum_height=camera.minimal_height,
        maximum_height=camera.maximal_height,
        world=context.world,
        width=200,
        height=200,
        resolution=0.02,
        origin=target_pose,
    )
    final_costmap = occupancy_costmap & visibility_costmap
    return Location(
        context=context,
        target_pose=target_pose,
        generator=final_costmap,
        validator=IsVisibleBy(
            context=context,
            target_pose=target_pose,
            target_body=target_body,
        ),
    )


def giskard_reachability_location(
    body: Body,
    context: Context,
    arm: Arm,
    grasp_pose: Optional[Pose] = None,
    destination: Optional[Pose] = None,
    approach_clearance: float = ActionConfig.approach_clearance,
    retreat_distance: float = ActionConfig.retreat_distance,
) -> Location:
    """
    Checks one grasp the caller already chose, like :func:`reachability_location`, but
    finds the standing poses by letting full-body control drive the robot to each
    candidate and offering where it arrived.

    .. note::
        - *Grasp*: given by the caller as ``grasp_pose``; no other grasp is ever tried.
        - *Result*: standing poses only.

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
    :returns: Standing poses from which ``grasp_pose`` can be reached, or released at
        ``destination``.
    """
    body_T_grasp = grasp_pose or Pose(reference_frame=body)
    target_pose = destination or body.global_pose
    grasp_frame = target_pose.to_homogeneous_matrix() @ body_T_grasp
    releases_the_body = destination is not None

    backend = GiskardLocationBackend(
        target_pose=target_pose,
        arm=arm,
        grasp_pose=grasp_frame,
        robot=context.robot,
        world=context.world,
        body_T_grasp=body_T_grasp,
        contact_bodies=[body],
        reverse=releases_the_body,
        approach_clearance=approach_clearance,
        retreat_distance=retreat_distance,
    )

    return Location(
        context=context,
        target_pose=target_pose,
        generator=backend,
        validator=AreReachableBy.for_grasp(
            grasp_pose=grasp_frame,
            arm=arm,
            body_T_grasp=body_T_grasp,
            context=context,
            reverse=releases_the_body,
            approach_clearance=approach_clearance,
            retreat_distance=retreat_distance,
        ),
    )
