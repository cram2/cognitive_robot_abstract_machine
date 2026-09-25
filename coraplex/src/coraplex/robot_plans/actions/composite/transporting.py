from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import List

from typing_extensions import Any, Self

from krrood.entity_query_language.factories import (
    a,
    an,
    entity,
    variable,
)
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ReachFraction
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
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from krrood.entity_query_language.query.match import Match
from krrood.patterns.field_metadata import JSONMetadata
from semantic_digital_twin.reasoning.predicates import InsideOf
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.semantic_annotations.mixins import GraspPose, HasGraspPoses
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Handle,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class LimitsItsCandidates:
    """
    Adds a limit on how many candidates a step tries.

    A candidate is tried by running the step with it, so a step that succeeds with none
    would otherwise try every one it is offered.
    """

    candidates_to_try: int = field(default=50, kw_only=True)
    """
    How many candidates a step tries before giving up.
    """

    def _bound_candidates(self, *steps: Any) -> None:
        """
        Limit every step that tries candidates to :attr:`candidates_to_try` of them.

        :param steps: The steps.
        """
        for step in steps:
            if isinstance(step, Match):
                step.expression.limit(self.candidates_to_try)


@dataclass
class TransportAction(ActionDescription, LimitsItsCandidates):
    """
    Picks an object up with one step and puts it down with another.
    """

    pick_up: MoveAndPickUpAction = field(
        metadata=JSONMetadata(serialize=False).as_dict()
    )
    """
    The step that picks the object up.
    """

    place: MoveAndPlaceAction = field(metadata=JSONMetadata(serialize=False).as_dict())
    """
    The step that puts down what :attr:`pick_up` picked up.
    """

    @classmethod
    def from_grasp(
        cls, grasp: GraspPose, target_location: Pose, arm: Arm, context: Context
    ) -> Self:
        """
        A transport that takes an object by `grasp` to `target_location`, standing
        wherever each step can be carried out from.

        :param grasp: The grasp to take the object by.
        :param target_location: Where to put the object down.
        :param arm: The arm that carries the object.
        :param context: The context the standing poses are drawn in.
        :return: The transport, standing near the object to pick it up and near the
            target to place it.
        """
        object_pose = grasp.graspable.root.global_pose
        return cls(
            pick_up=a(MoveAndPickUpAction)(
                navigate=a(NavigateAction)(
                    target_location=variable(
                        Pose,
                        domain=ReachabilityLocation(
                            Pose(reference_frame=grasp.graspable.root),
                            arm,
                            context=context,
                        ),
                    )
                ),
                face_and_look_at=a(FaceAndLookAtAction)(
                    face_at=a(FaceAtAction)(target=object_pose),
                    look_at=a(LookAtAction)(target=object_pose),
                ),
                pick_up=a(PickUpAction)(grasp=grasp, arm=arm),
            ),
            place=a(MoveAndPlaceAction)(
                navigate=a(NavigateAction)(
                    target_location=variable(
                        Pose,
                        domain=ReachabilityLocation(
                            target_location, arm, context=context
                        ),
                    )
                ),
                face_and_look_at=a(FaceAndLookAtAction)(
                    face_at=a(FaceAtAction)(target=target_location),
                    look_at=a(LookAtAction)(target=target_location),
                ),
                place=a(PlaceAction)(
                    object_designator=grasp.graspable, target_location=target_location
                ),
            ),
        )

    @property
    def _action_plan(self) -> PlanNode:
        self._bound_candidates(self.pick_up, self.place)
        return sequential(
            [
                ParkArmsAction(self.robot.get_arms()),
                self.pick_up,
                ParkArmsAction(self.robot.get_arms()),
                self.place,
                ParkArmsAction(self.robot.get_arms()),
            ]
        )


@dataclass
class PickAndPlaceAction(ActionDescription, LimitsItsCandidates):
    """
    Picks an object up with one step and puts it down with another, without moving the
    base of the robot.
    """

    pick_up: PickUpAction = field(metadata=JSONMetadata(serialize=False).as_dict())
    """
    The step that picks the object up.
    """

    place: PlaceAction = field(metadata=JSONMetadata(serialize=False).as_dict())
    """
    The step that puts down what :attr:`pick_up` picked up.
    """

    @property
    def _action_plan(self) -> PlanNode:
        self._bound_candidates(self.pick_up, self.place)
        return sequential(
            [
                ParkArmsAction(self.robot.get_arms()),
                self.pick_up,
                ParkArmsAction(self.robot.get_arms()),
                self.place,
                ParkArmsAction(self.robot.get_arms()),
            ]
        )


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
        object_designator: HasGraspPoses,
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
                FaceAtAction(target_location), LookAtAction(target_location)
            ),
            place=PlaceAction(object_designator, target_location),
        )

    @property
    def _action_plan(self) -> PlanNode:
        return sequential([self.navigate, self.face_and_look_at, self.place])


@dataclass
class MoveAndPickUpAction(ActionDescription, LimitsItsCandidates):
    """
    Navigates to where the robot stands, faces the object and picks it up, opening the
    drawer it is in first.
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
        grasp: GraspPose,
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
        object_pose = grasp.graspable.root.global_pose
        return cls(
            navigate=NavigateAction(standing_position),
            face_and_look_at=FaceAndLookAtAction(
                FaceAtAction(object_pose), LookAtAction(object_pose)
            ),
            pick_up=PickUpAction(
                grasp,
                arm,
                approach_clearance=approach_clearance,
                retreat_distance=retreat_distance,
            ),
        )

    @property
    def _action_plan(self) -> PlanNode:
        children = []
        for container in self._containers_around_the_object():
            children.extend(self._make_open_container_actions(container))
        children.extend([self.navigate, self.face_and_look_at, self.pick_up])
        return sequential(children)

    def _containers_around_the_object(self) -> List[Body]:
        """
        :return: The bodies the object to pick up lies inside of.
        """
        object_body = self.pick_up.grasp.graspable.root
        return [
            body
            for body in self.world.bodies
            if body != object_body
            and InsideOf(object_body, body).compute_containment_ratio() > 0.9
        ]

    def _make_open_container_actions(self, container: Body) -> List[Match]:
        """
        :param container: A body the object lies inside of.
        :return: The step opening it, from a standing pose tried together with the
            opening, or nothing if the container is not a known drawer.
        """
        drawer_annotation = an(
            entity(
                drawer := variable(Drawer, domain=self.world.semantic_annotations)
            ).where(drawer.root == container)
        )
        drawer_annotation = list(drawer_annotation.evaluate())
        if len(drawer_annotation) == 0:
            return []
        handle = drawer_annotation[0].handle
        arm = self.pick_up.arm
        handle_pose = handle.root.global_pose
        open_the_drawer = a(MoveAndOpenAction)(
            navigate=a(NavigateAction)(
                target_location=variable(
                    Pose,
                    domain=ReachabilityLocation(
                        Pose(reference_frame=handle.root),
                        arm,
                        ReachFraction.ACCESSING,
                        context=self.context,
                    ),
                )
            ),
            face_and_look_at=a(FaceAndLookAtAction)(
                face_at=a(FaceAtAction)(target=handle_pose),
                look_at=a(LookAtAction)(target=handle_pose),
            ),
            open_container=a(OpenAction)(handle=handle, arm=arm),
        )
        self._bound_candidates(open_the_drawer)
        return [open_the_drawer]


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
        handle_pose = handle.root.global_pose
        return cls(
            navigate=NavigateAction(standing_position),
            face_and_look_at=FaceAndLookAtAction(
                FaceAtAction(handle_pose), LookAtAction(handle_pose)
            ),
            open_container=OpenAction(handle, arm),
        )

    @property
    def _action_plan(self) -> PlanNode:
        return sequential([self.navigate, self.face_and_look_at, self.open_container])
