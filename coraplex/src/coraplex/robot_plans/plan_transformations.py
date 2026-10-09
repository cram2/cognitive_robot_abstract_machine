from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING, Generic, List, Optional, cast

from coraplex.datastructures.enums import (
    DetectionTechnique,
    InsertionPosition,
    ReachFraction,
)
from coraplex.exceptions import ReachHasNoFinalApproach
from coraplex.locations.locations import ReachabilityLocation
from coraplex.plans.plan_node import ActionLike, ActionNode, MotionNode, PlanNode
from coraplex.plans.underspecified import UnderspecifiedNode
from coraplex.plans.factories import make_node
from coraplex.plans.plan_transformation import (
    InsertionTransformation,
    MatchedType,
    PlanTransformation,
)
from coraplex.robot_plans import MoveToolCenterPointMotion
from coraplex.robot_plans.actions.composite.facing import FaceAndLookAtAction
from coraplex.robot_plans.actions.composite.transporting import (
    MoveAndOpenAction,
    MoveAndPickUpAction,
    PickAndPlaceAction,
)
from coraplex.robot_plans.actions.core.container import OpenAction
from coraplex.robot_plans.actions.core.misc import DetectAction
from coraplex.robot_plans.actions.core.navigation import (
    FaceAtAction,
    LookAtAction,
    NavigateAction,
)
from coraplex.robot_plans.actions.core.pick_up import PickUpAction, ReachAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import a, variable
from krrood.entity_query_language.query.match import Match
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from semantic_digital_twin.reasoning.predicates import InsideOf
from semantic_digital_twin.grasping.grasp_candidates import (
    GraspCandidate,
    CanBeGrasped,
)
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer
from semantic_digital_twin.spatial_types.spatial_types import Pose, Pose2D

if TYPE_CHECKING:
    from coraplex.datastructures.dataclasses import Context
    from semantic_digital_twin.world import World


# %% perceiving before a grasp


@dataclass
class DetectBeforeGrasp(InsertionTransformation[ReachAction]):
    """
    Looks at the object and detects it before a reach makes its final approach, so that
    the approach acts on a freshly perceived pose instead of the one the world holds.
    """

    @property
    def position(self) -> InsertionPosition:
        return InsertionPosition.BEFORE

    def is_applicable(self, plan_node: PlanNode) -> bool:
        return True

    def final_approach(self, plan_node: ActionNode) -> MotionNode:
        """
        :param plan_node: The node of the reach
        :raises ReachHasNoFinalApproach: If no tool center point motion lies below the
            reach's node.
        :return: The reach's last tool center point motion, which brings the gripper
            onto the object.
        """
        motions = [
            node
            for node in plan_node.descendants
            if isinstance(node, MotionNode)
            and isinstance(node.motion, MoveToolCenterPointMotion)
        ]
        if not motions:
            raise ReachHasNoFinalApproach(plan_node)
        return motions[-1]

    def anchor(self, plan_node: ActionNode) -> PlanNode:
        return self.final_approach(plan_node)

    def nodes_to_insert(self, plan_node: ActionNode) -> List[ActionLike]:
        reach = cast(ReachAction, plan_node.action)
        return [
            LookAtAction(target=self.final_approach(plan_node).motion.target),
            DetectAction(
                DetectionTechnique.TYPES,
                object_sem_annotation=type(reach.grasp.graspable),
                accept_first_if_multiple=True,
            ),
        ]


# %% opening what the object lies in


@dataclass
class DrawerOpening(
    InsertionTransformation[MatchedType],
    Generic[MatchedType],
    SubClassSafeGeneric,
    ABC,
):
    """
    The shared part of the rewrites that open the drawers an object lies in.
    """

    minimum_containment_ratio: float = 0.9
    """
    How much of the object has to lie within a drawer for it to count as being in it.
    """

    minimum_opening_ratio: float = 0.9
    """
    How far along its travel a drawer has to stand pulled out to count as open already.
    """

    @property
    def position(self) -> InsertionPosition:
        return InsertionPosition.BEFORE

    def _closed_drawers_containing(
        self, annotation: HasRootBody, world: World
    ) -> List[Drawer]:
        """
        :param annotation: The object to locate
        :param world: The world the object and the drawers belong to
        :return: The drawers the object lies in that do not already stand open.
        """
        object_body = annotation.root
        return [
            drawer
            for drawer in world.get_semantic_annotations_by_type(Drawer)
            if InsideOf(object_body, drawer.root).compute_containment_ratio()
            > self.minimum_containment_ratio
            and drawer.opening_ratio < self.minimum_opening_ratio
        ]

    def opening_nodes(
        self, drawer: Drawer, arm: Arm, context: Context
    ) -> List[ActionLike]:
        """
        :param drawer: The drawer to open
        :param arm: The arm that opens it
        :param context: The context the standing pose is sampled in
        :return: The opening, from a standing pose tried together with it.
        """
        handle_pose = Pose(reference_frame=drawer.handle.root)
        open_the_drawer = a(MoveAndOpenAction)(
            navigate=a(NavigateAction)(
                target_location=variable(
                    Pose2D,
                    domain=ReachabilityLocation(
                        handle_pose, arm, ReachFraction.ACCESSING, context=context
                    ),
                )
            ),
            face_and_look_at=a(FaceAndLookAtAction)(
                face_at=a(FaceAtAction)(target=handle_pose),
                look_at=a(LookAtAction)(target=handle_pose),
            ),
            open_container=a(OpenAction)(handle=drawer.handle, arm=arm),
        )
        return [open_the_drawer]

    def anchor(self, plan_node: PlanNode) -> PlanNode:
        return plan_node


@dataclass
class OpenDrawerBeforePickUp(DrawerOpening[PickUpAction]):
    """
    Opens the drawers an object lies in before the robot picks it up, so that it reaches
    into an open drawer instead of a closed one.

    Nothing else positions the robot for a pick-up of its own, and opening a drawer
    leaves the robot standing at its handle, so the rewrite ends by parking and driving
    to a pose the object itself can be reached from.
    """

    def is_applicable(self, plan_node: ActionNode) -> bool:
        pick_up = cast(PickUpAction, plan_node.action)
        return bool(
            self._closed_drawers_containing(pick_up.grasp.graspable, pick_up.world)
        )

    def nodes_to_insert(self, plan_node: ActionNode) -> List[ActionLike]:
        pick_up = cast(PickUpAction, plan_node.action)
        graspable = pick_up.grasp.graspable
        nodes = []
        for drawer in self._closed_drawers_containing(graspable, pick_up.world):
            nodes.extend(self.opening_nodes(drawer, pick_up.arm, pick_up.context))
        drive_to_the_object = a(NavigateAction)(
            target_location=variable(
                Pose2D,
                # A location samples its poses only once the drive is grounded, by
                # which time the drawers this rewrite opens stand open.
                domain=ReachabilityLocation(
                    Pose(reference_frame=graspable.root),
                    pick_up.arm,
                    context=pick_up.context,
                ),
            ),
        )
        nodes.extend([ParkArmsAction(pick_up.robot.all_arms), drive_to_the_object])
        return nodes


@dataclass
class PickUpTarget:
    """
    The object a pick-up takes hold of, and the arm it takes hold with.
    """

    graspable: CanBeGrasped
    """
    The object that is picked up.
    """

    arm: Arm
    """
    The arm that picks it up.
    """


@dataclass
class OpenDrawerBeforeMoveAndPickUp(DrawerOpening[MoveAndPickUpAction]):
    """
    Opens the drawers an object lies in before the robot moves to it and picks it up.

    The opening precedes the whole move-and-pick-up, whose own drive then positions the
    robot at the object. When every candidate of a move-and-pick-up still to be grounded
    picks up the same object with the same arm, the drawer is opened once in front of
    it, so every candidate is grounded and tried with the drawer standing open.
    Otherwise each candidate gets its own opening, inside the sequence it is tried in.
    """

    def matches_node(self, plan_node: PlanNode) -> bool:
        if isinstance(plan_node, UnderspecifiedNode):
            return issubclass(plan_node.designator_type, MoveAndPickUpAction)
        return super().matches_node(plan_node)

    def is_applicable(self, plan_node: ActionNode | UnderspecifiedNode) -> bool:
        target = self._pick_up_target(plan_node)
        return target is not None and bool(
            self._closed_drawers_containing(target.graspable, plan_node.plan.world)
        )

    def nodes_to_insert(
        self, plan_node: ActionNode | UnderspecifiedNode
    ) -> List[ActionLike]:
        target = self._pick_up_target(plan_node)
        nodes = []
        for drawer in self._closed_drawers_containing(
            target.graspable, plan_node.plan.world
        ):
            nodes.extend(self.opening_nodes(drawer, target.arm, plan_node.context))
        if isinstance(plan_node, UnderspecifiedNode):
            # The candidates are grounded after the opening, which leaves the arms at
            # the handle, where they would stand in collision at every standing pose.
            nodes.append(ParkArmsAction(plan_node.plan.robot.all_arms))
        return nodes

    def _pick_up_target(
        self, plan_node: ActionNode | UnderspecifiedNode
    ) -> Optional[PickUpTarget]:
        """
        :param plan_node: A node this matches.
        :return: What the move-and-pick-up picks up and with which arm, or ``None`` if
            it is still to be grounded and its candidates differ in either.
        """
        if isinstance(plan_node, UnderspecifiedNode):
            return self._pick_up_target_shared_by_candidates_of(
                plan_node.underspecified_action
            )
        pick_up = cast(MoveAndPickUpAction, plan_node.action).pick_up
        return PickUpTarget(graspable=pick_up.grasp.graspable, arm=pick_up.arm)

    @staticmethod
    def _pick_up_target_shared_by_candidates_of(
        move_and_pick_up: Match[MoveAndPickUpAction],
    ) -> Optional[PickUpTarget]:
        """
        :param move_and_pick_up: A move-and-pick-up still to be grounded.
        :return: The object and arm every one of its candidates picks up with, or
            ``None`` if they are not the same for all of them, or not known before
            grounding.
        """
        grasp = move_and_pick_up.pick_up.grasp.apply_mapping_on_external_root(
            move_and_pick_up
        )
        arm = move_and_pick_up.pick_up.arm.apply_mapping_on_external_root(
            move_and_pick_up
        )
        grasps = grasp._domain_ if isinstance(grasp, Variable) else [grasp]
        if not isinstance(arm, Arm):
            return None
        if not all(isinstance(candidate, GraspCandidate) for candidate in grasps):
            return None
        graspables = {candidate.graspable for candidate in grasps}
        if len(graspables) != 1:
            return None
        [graspable] = graspables
        return PickUpTarget(graspable=graspable, arm=arm)


# %% parking around a pick-and-place


@dataclass
class ParkArmsAroundPickAndPlaceSteps(PlanTransformation[PickAndPlaceAction]):
    """
    Parks the robot's arms before the pick-up of a pick-and-place, between it and the
    place, and after the place, so that neither step starts with the arms wherever the
    one before it left them.
    """

    def is_applicable(self, plan_node: ActionNode) -> bool:
        return True

    def apply(self, plan_node: ActionNode) -> None:
        [steps] = plan_node.body_children
        for step in steps.children:
            plan_node.plan.insert_before(step, self._parking(plan_node))
        plan_node.plan.insert_after(steps.children[-1], self._parking(plan_node))

    @staticmethod
    def _parking(plan_node: ActionNode) -> PlanNode:
        """
        :param plan_node: The node of the pick-and-place.
        :return: A new node parking every arm of the robot running the plan.
        """
        return make_node(ParkArmsAction(plan_node.plan.robot.all_arms))


# %% parking before anything else


@dataclass
class ParkArmsBeforeFirstAction(InsertionTransformation[ActionNode]):
    """
    Parks the robot's arms in front of the first action of a plan.

    An action that is grounded against the world, such as a drive to a pose the object
    can be reached from, judges the robot in the configuration it is in. Arms left
    wherever an earlier plan dropped them stand in collision at every candidate pose,
    which rules out the whole location before it is ever checked for reachability.
    """

    @property
    def position(self) -> InsertionPosition:
        return InsertionPosition.BEFORE

    def is_applicable(self, plan_node: PlanNode) -> bool:
        return plan_node.plan.actions[0] is plan_node and not isinstance(
            plan_node.action, ParkArmsAction
        )

    def anchor(self, plan_node: PlanNode) -> PlanNode:
        return plan_node

    def nodes_to_insert(self, plan_node: PlanNode) -> List[ActionLike]:
        return [ParkArmsAction(plan_node.action.robot.all_arms)]
