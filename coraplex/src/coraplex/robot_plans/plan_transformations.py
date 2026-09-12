from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING, List, cast

from coraplex.datastructures.enums import Arms, DetectionTechnique, InsertionPosition
from coraplex.exceptions import PerceptionTargetMissing
from coraplex.locations.base import DeferredLocation
from coraplex.locations.factories import reachability_location
from coraplex.plans.plan_node import ActionLike, ActionNode, MotionNode, PlanNode
from coraplex.plans.plan_transformation import (
    ActionMatch,
    InsertionRewrite,
    PlanMatch,
)
from coraplex.robot_plans.actions.core.container import OpenAction
from coraplex.robot_plans.actions.core.misc import DetectAction
from coraplex.robot_plans.actions.core.navigation import LookAtAction, NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction, ReachAction
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from krrood.entity_query_language.factories import a, variable
from semantic_digital_twin.reasoning.predicates import InsideOf
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer
from semantic_digital_twin.spatial_types.spatial_types import Pose

if TYPE_CHECKING:
    from coraplex.datastructures.dataclasses import Context
    from semantic_digital_twin.world import World


# %% perceiving before a grasp


@dataclass
class DetectBeforeGrasp(InsertionRewrite, ActionMatch[ReachAction]):
    """
    Looks at the object and detects it before a reach makes its final approach, so that
    the approach acts on a freshly perceived pose instead of the one the world holds.
    """

    @property
    def position(self) -> InsertionPosition:
        return InsertionPosition.BEFORE

    def final_approach(self, plan_node: ActionNode) -> MotionNode:
        """
        :param plan_node: The node of the reach
        :return: The reach's last motion, which brings the tool center point onto the
            object.
        """
        motions = [
            node for node in plan_node.descendants if isinstance(node, MotionNode)
        ]
        return motions[-1]

    def anchor(self, plan_node: ActionNode) -> PlanNode:
        return self.final_approach(plan_node)

    def nodes_to_insert(self, plan_node: ActionNode) -> List[ActionLike]:
        reach = cast(ReachAction, plan_node.action)
        if reach.object_designator is None:
            raise PerceptionTargetMissing(reach)
        return [
            LookAtAction(self.final_approach(plan_node).motion.target),
            DetectAction(
                DetectionTechnique.TYPES,
                object_sem_annotation=type(reach.object_designator),
                accept_first_if_multiple=True,
            ),
        ]


# %% opening what the object lies in


@dataclass
class DrawerOpening(InsertionRewrite):
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

    def closed_drawers_holding(
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
        self, drawer: Drawer, arm: Arms, context: Context
    ) -> List[ActionLike]:
        """
        :param drawer: The drawer to open
        :param arm: The arm that opens it
        :param context: The context the drive to the handle is grounded against
        :return: The drive that makes the handle reachable and the opening itself.
        """
        handle = drawer.handle.root
        return [
            a(NavigateAction)(
                target_location=variable(
                    Pose,
                    domain=reachability_location(handle.global_pose, context, arm),
                ),
                keep_joint_states=True,
            ),
            OpenAction(handle, arm),
        ]

    def anchor(self, plan_node: PlanNode) -> PlanNode:
        return plan_node


@dataclass
class OpenDrawerBeforePickUp(DrawerOpening, ActionMatch[PickUpAction]):
    """
    Opens the drawers an object lies in before the robot picks it up, so that it reaches
    into an open drawer instead of a closed one.

    Nothing else positions the robot for a pick-up of its own, and opening a drawer
    leaves the robot standing at its handle, so the rewrite ends by parking and driving
    to a pose the object itself can be reached from.
    """

    def is_applicable(self, plan_node: PlanNode) -> bool:
        pick_up = cast(PickUpAction, plan_node.action)
        return bool(
            self.closed_drawers_holding(pick_up.object_designator, pick_up.world)
        )

    def nodes_to_insert(self, plan_node: PlanNode) -> List[ActionLike]:
        pick_up = cast(PickUpAction, plan_node.action)
        nodes = []
        for drawer in self.closed_drawers_holding(
            pick_up.object_designator, pick_up.world
        ):
            nodes.extend(self.opening_nodes(drawer, pick_up.arm, pick_up.context))
        nodes.extend(
            [
                ParkArmsAction(Arms.BOTH),
                a(NavigateAction)(
                    target_location=variable(
                        Pose,
                        # Built when the drive is grounded rather than now: the drawers
                        # this rewrite opens stand open by then, and a pose the object
                        # can be reached from only exists once they do.
                        domain=DeferredLocation(
                            lambda: reachability_location(
                                pick_up.object_designator.root,
                                pick_up.context,
                                pick_up.arm,
                                pick_up.grasp_description,
                            )
                        ),
                    ),
                    keep_joint_states=True,
                ),
            ]
        )
        return nodes


@dataclass
class OpenDrawerBeforeTransport(DrawerOpening, ActionMatch[TransportAction]):
    """
    Opens the drawers the transported object lies in before the transport starts.

    A transport drives to the object before picking it up, and that drive is grounded
    against the world it finds: with the drawer still shut there is no pose the object
    can be reached from. The opening therefore precedes the whole transport rather than
    the pick-up inside it, and the transport's own drive is what positions the robot
    afterwards.
    """

    def is_applicable(self, plan_node: PlanNode) -> bool:
        transport = cast(TransportAction, plan_node.action)
        return bool(
            self.closed_drawers_holding(transport.object_designator, transport.world)
        )

    def nodes_to_insert(self, plan_node: PlanNode) -> List[ActionLike]:
        transport = cast(TransportAction, plan_node.action)
        nodes = []
        for drawer in self.closed_drawers_holding(
            transport.object_designator, transport.world
        ):
            nodes.extend(self.opening_nodes(drawer, transport.arm, transport.context))
        return nodes


# %% parking before anything else


@dataclass
class ParkArmsBeforeFirstAction(InsertionRewrite, PlanMatch[ActionNode]):
    """
    Parks the arms in front of the first action of a plan.

    An action that is grounded against the world, such as a drive to a pose the object
    can be reached from, judges the robot in the configuration it is in. Arms left
    wherever an earlier plan dropped them stand in collision at every candidate pose,
    which rules out the whole location before it is ever checked for reachability.
    """

    arm: Arms = Arms.BOTH
    """
    The arms that are parked.
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
        return [ParkArmsAction(self.arm)]
