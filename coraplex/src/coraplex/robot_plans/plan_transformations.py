from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING, Generic, List, cast

from coraplex.datastructures.enums import (
    DetectionTechnique,
    InsertionPosition,
    ReachFraction,
)
from coraplex.locations.locations import ReachabilityLocation
from coraplex.plans.plan_node import ActionLike, ActionNode, MotionNode, PlanNode
from coraplex.plans.plan_transformation import (
    InsertionTransformation,
    MatchedType,
)
from coraplex.robot_plans import MoveToolCenterPointMotion
from coraplex.robot_plans.actions.composite.facing import FaceAndLookAtAction
from coraplex.robot_plans.actions.composite.transporting import (
    LimitsItsCandidates,
    MoveAndOpenAction,
    MoveAndPickUpAction,
    TransportAction,
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
from krrood.entity_query_language.factories import a, variable
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from semantic_digital_twin.reasoning.predicates import InsideOf
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer
from semantic_digital_twin.spatial_types.spatial_types import Pose

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
        :return: The reach's last tool center point motion, which brings the gripper
            onto the object.
        """
        motions = [
            node
            for node in plan_node.descendants
            if isinstance(node, MotionNode)
            and isinstance(node.motion, MoveToolCenterPointMotion)
        ]
        return motions[-1]

    def anchor(self, plan_node: ActionNode) -> PlanNode:
        return self.final_approach(plan_node)

    def nodes_to_insert(self, plan_node: ActionNode) -> List[ActionLike]:
        reach = cast(ReachAction, plan_node.action)
        return [
            LookAtAction(self.final_approach(plan_node).motion.target),
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
    LimitsItsCandidates,
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
                    Pose,
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
        self._bound_candidates(open_the_drawer)
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
                Pose,
                # A location samples its poses only once the drive is grounded, by
                # which time the drawers this rewrite opens stand open.
                domain=ReachabilityLocation(
                    Pose(reference_frame=graspable.root),
                    pick_up.arm,
                    context=pick_up.context,
                ),
            ),
        )
        self._bound_candidates(drive_to_the_object)
        nodes.extend([ParkArmsAction(pick_up.robot.get_arms()), drive_to_the_object])
        return nodes


@dataclass
class OpenDrawerBeforeMoveAndPickUp(DrawerOpening[MoveAndPickUpAction]):
    """
    Opens the drawers an object lies in before the robot moves to it and picks it up.

    The opening precedes the whole move-and-pick-up, whose own drive then positions the
    robot at the object. A move-and-pick-up left underspecified is grounded, and tried,
    inside a sequence of its own, so the opening is tried and run together with it.
    """

    def is_applicable(self, plan_node: ActionNode) -> bool:
        move_and_pick_up = cast(MoveAndPickUpAction, plan_node.action)
        return bool(
            self._closed_drawers_containing(
                move_and_pick_up.pick_up.grasp.graspable, move_and_pick_up.world
            )
        )

    def nodes_to_insert(self, plan_node: ActionNode) -> List[ActionLike]:
        move_and_pick_up = cast(MoveAndPickUpAction, plan_node.action)
        pick_up = move_and_pick_up.pick_up
        nodes = []
        for drawer in self._closed_drawers_containing(
            pick_up.grasp.graspable, move_and_pick_up.world
        ):
            nodes.extend(
                self.opening_nodes(drawer, pick_up.arm, move_and_pick_up.context)
            )
        return nodes


@dataclass
class OpenDrawerBeforeTransport(DrawerOpening[TransportAction]):
    """
    Opens the drawers the transported object lies in before the transport starts.

    Every candidate of the transport's pick-up is tried against the world as it stands
    when the pick-up is grounded. Opened before the transport, the drawer stands open
    for all of them, rather than each candidate searching for an opening of its own.
    """

    def is_applicable(self, plan_node: ActionNode) -> bool:
        transport = cast(TransportAction, plan_node.action)
        return bool(
            self._closed_drawers_containing(
                transport.transported_object, transport.world
            )
        )

    def nodes_to_insert(self, plan_node: ActionNode) -> List[ActionLike]:
        transport = cast(TransportAction, plan_node.action)
        nodes = []
        for drawer in self._closed_drawers_containing(
            transport.transported_object, transport.world
        ):
            nodes.extend(
                self.opening_nodes(drawer, transport.carrying_arm, transport.context)
            )
        return nodes


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
        return [ParkArmsAction(plan_node.action.robot.get_arms())]
