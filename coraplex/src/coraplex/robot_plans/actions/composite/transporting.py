from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import Self

from krrood.entity_query_language.factories import a, variable
from coraplex.datastructures.dataclasses import Context
from coraplex.locations.locations import ReachabilityLocation
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.mixins import HasApproachesGraspPoses
from coraplex.robot_plans.actions.composite.facing import FaceAndLookAtAction
from coraplex.robot_plans.actions.core.container import OpenAction
from coraplex.robot_plans.actions.core.navigation import (
    FaceAtAction,
    LookAtAction,
    NavigateAction,
)
from coraplex.querying.predicates import IsAmongTheClosestGraspsTo
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from krrood.entity_query_language.query.match import Match
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.semantic_annotations.mixins import HasGraspCandidates
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle
from semantic_digital_twin.spatial_types.spatial_types import Pose


@dataclass
class TransportAction(ActionDescription):
    """
    Picks an object up with one step and puts it down with another.
    """

    pick_up: MoveAndPickUpAction
    """
    The step that picks the object up.
    """

    place: MoveAndPlaceAction
    """
    The step that puts down what :attr:`pick_up` picked up.
    """

    @classmethod
    def from_graspable_by_closest_grasps(
        cls,
        graspable: HasGraspCandidates,
        target_location: Pose,
        arm: Arm,
        context: Context,
        number_of_grasps: int = IsAmongTheClosestGraspsTo.number_of_grasps,
    ) -> Self:
        """
        A transport that takes `graspable` to `target_location`, standing wherever each
        step can be carried out from and taking the object by the grasps closest to the
        robot there.

        :param graspable: The object to transport.
        :param target_location: Where to put the object down.
        :param arm: The arm that carries the object.
        :param context: The context the standing poses are sampled in.
        :param number_of_grasps: How many of the object's grasps closest to a standing
            pose are tried from there.
        :return: The transport, standing near the object to pick it up and near the
            target to place it.
        """
        return cls(
            pick_up=MoveAndPickUpAction.from_graspable_by_closest_grasps(
                graspable=graspable,
                arm=arm,
                context=context,
                number_of_grasps=number_of_grasps,
            ),
            place=a(MoveAndPlaceAction)(
                navigate=a(NavigateAction)(
                    target_location=variable(
                        Pose,
                        domain=ReachabilityLocation(
                            target_pose=target_location, arm=arm, context=context
                        ),
                    )
                ),
                face_and_look_at=a(FaceAndLookAtAction)(
                    face_at=a(FaceAtAction)(target=target_location),
                    look_at=a(LookAtAction)(target=target_location),
                ),
                place=a(PlaceAction)(
                    object_designator=graspable, target_location=target_location
                ),
            ),
        )

    @property
    def _action_plan(self) -> PlanNode:
        return sequential(
            [
                ParkArmsAction(self.robot.all_arms),
                self.pick_up,
                ParkArmsAction(self.robot.all_arms),
                self.place,
                ParkArmsAction(self.robot.all_arms),
            ]
        )


@dataclass
class PickAndPlaceAction(ActionDescription):
    """
    Picks an object up with one step and puts it down with another, without moving the
    base of the robot.
    """

    pick_up: PickUpAction
    """
    The step that picks the object up.
    """

    place: PlaceAction
    """
    The step that puts down what :attr:`pick_up` picked up.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return sequential([self.pick_up, self.place])


@dataclass
class MoveAndPlaceAction(ActionDescription):
    """
    Navigates to where the robot stands, faces the target and places the object there.
    """

    navigate: NavigateAction
    """
    The step to where the robot stands while placing.
    """

    face_and_look_at: FaceAndLookAtAction
    """
    The turn towards the target and the look at it.
    """

    place: PlaceAction
    """
    The step that puts the object down.
    """

    @classmethod
    def from_standing_position(
        cls,
        standing_position: Pose,
        target_location: Pose,
        object_designator: HasGraspCandidates,
    ) -> Self:
        """
        :param standing_position: Where the robot stands while placing.
        :param target_location: Where to put the object down.
        :param object_designator: The object to put down.
        :return: The step placing the object from `standing_position`.
        """
        return cls(
            navigate=NavigateAction(standing_position),
            face_and_look_at=FaceAndLookAtAction(
                face_at=FaceAtAction(target_location),
                look_at=LookAtAction(target_location),
            ),
            place=PlaceAction(
                object_designator=object_designator, target_location=target_location
            ),
        )

    @property
    def _action_plan(self) -> PlanNode:
        return sequential([self.navigate, self.face_and_look_at, self.place])


@dataclass
class MoveAndPickUpAction(ActionDescription):
    """
    Navigates to where the robot stands, faces the object and picks it up.
    """

    navigate: NavigateAction
    """
    The step to where the robot stands while picking up.
    """

    face_and_look_at: FaceAndLookAtAction
    """
    The turn towards the object and the look at it.
    """

    pick_up: PickUpAction
    """
    The step that picks the object up.
    """

    @classmethod
    def from_standing_position(
        cls,
        standing_position: Pose,
        grasp: GraspCandidate,
        arm: Arm,
        approach_clearance: float = HasApproachesGraspPoses.approach_clearance,
        retreat_distance: float = HasApproachesGraspPoses.retreat_distance,
    ) -> Self:
        """
        :param standing_position: Where the robot stands while picking up.
        :param grasp: The grasp to take hold by, which also names the object.
        :param arm: The arm to pick up with.
        :param approach_clearance: How far from the grasp the gripper approaches from.
        :param retreat_distance: How far the gripper retreats with the object.
        :return: The step picking the object up from `standing_position`.
        """
        object_pose = Pose(reference_frame=grasp.graspable.root)
        return cls(
            navigate=NavigateAction(standing_position),
            face_and_look_at=FaceAndLookAtAction(
                face_at=FaceAtAction(object_pose), look_at=LookAtAction(object_pose)
            ),
            pick_up=PickUpAction(
                grasp=grasp,
                arm=arm,
                approach_clearance=approach_clearance,
                retreat_distance=retreat_distance,
            ),
        )

    @classmethod
    def from_graspable_by_closest_grasps(
        cls,
        graspable: HasGraspCandidates,
        arm: Arm,
        context: Context,
        number_of_grasps: int = IsAmongTheClosestGraspsTo.number_of_grasps,
    ) -> Match:
        """
        A pick-up of `graspable`, standing wherever it can be reached from and taking it
        by the grasps closest to the robot there.

        The closeness is a ``where`` condition on the returned match,
        :class:`~coraplex.querying.predicates.IsAmongTheClosestGraspsTo`.

        :param graspable: The object to pick up.
        :param arm: The arm to pick up with.
        :param context: The context the standing poses are sampled in.
        :param number_of_grasps: How many of the object's grasps closest to a standing
            pose are tried from there.
        :return: The pick-up, with the standing pose and the grasp left open.
        """
        grasps = graspable.grasp_candidates()
        object_pose = Pose(reference_frame=graspable.root)
        step = a(cls)(
            navigate=a(NavigateAction)(
                target_location=variable(
                    Pose,
                    domain=ReachabilityLocation(
                        target_pose=object_pose, arm=arm, context=context
                    ),
                )
            ),
            face_and_look_at=a(FaceAndLookAtAction)(
                face_at=a(FaceAtAction)(target=object_pose),
                look_at=a(LookAtAction)(target=object_pose),
            ),
            pick_up=a(PickUpAction)(
                grasp=variable(GraspCandidate, domain=grasps), arm=arm
            ),
        )
        return step.where(
            IsAmongTheClosestGraspsTo(
                step.pick_up.grasp,
                step.navigate.target_location,
                grasps,
                number_of_grasps,
            )
        )

    @property
    def _action_plan(self) -> PlanNode:
        return sequential([self.navigate, self.face_and_look_at, self.pick_up])


@dataclass
class MoveAndOpenAction(ActionDescription):
    """
    Navigates to where the robot stands, faces the handle and opens its container.
    """

    navigate: NavigateAction
    """
    The step to where the robot stands while opening.
    """

    face_and_look_at: FaceAndLookAtAction
    """
    The turn towards the handle and the look at it.
    """

    open_container: OpenAction
    """
    The step that opens the container.
    """

    @classmethod
    def from_standing_position(
        cls, standing_position: Pose, handle: Handle, arm: Arm
    ) -> Self:
        """
        :param standing_position: Where the robot stands while opening.
        :param handle: The handle of the container to open.
        :param arm: The arm to open with.
        :return: The step opening the container from `standing_position`.
        """
        handle_pose = Pose(reference_frame=handle.root)
        return cls(
            navigate=NavigateAction(standing_position),
            face_and_look_at=FaceAndLookAtAction(
                face_at=FaceAtAction(handle_pose), look_at=LookAtAction(handle_pose)
            ),
            open_container=OpenAction(handle=handle, arm=arm),
        )

    @property
    def _action_plan(self) -> PlanNode:
        return sequential([self.navigate, self.face_and_look_at, self.open_container])
