from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import List, cast

from coraplex.datastructures.enums import DetectionTechnique
from coraplex.exceptions import PerceptionTargetMissing
from coraplex.locations.factories import reachability_location
from coraplex.plans.plan_node import ActionLike, ActionNode, MotionNode, PlanNode
from coraplex.plans.plan_transformation import (
    ActionTransformation,
    InsertionTransformation,
)
from coraplex.robot_plans.actions.core.container import OpenAction
from coraplex.robot_plans.actions.core.misc import DetectAction
from coraplex.robot_plans.actions.core.navigation import LookAtAction, NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction, ReachAction
from krrood.entity_query_language.factories import a, variable
from semantic_digital_twin.reasoning.predicates import InsideOf
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer
from semantic_digital_twin.spatial_types.spatial_types import Pose

# %% perceiving before a grasp


@dataclass
class DetectBeforeGrasp(InsertionTransformation, ActionTransformation[ReachAction]):
    """
    Looks at the object and detects it before a reach makes its final approach, so that
    the approach acts on a freshly perceived pose instead of the one the world holds.
    """

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
class OpenDrawerBeforePickUp(
    InsertionTransformation, ActionTransformation[PickUpAction]
):
    """
    Opens the drawers an object lies in before the robot picks it up, so that it reaches
    into an open drawer instead of a closed one.
    """

    minimum_containment_ratio: float = 0.9
    """
    How much of the object has to lie within a drawer for it to count as being in it.
    """

    def containing_drawers(self, pick_up: PickUpAction) -> List[Drawer]:
        """
        :param pick_up: The pick-up whose object to locate
        :return: The drawers the object lies in.
        """
        object_body = pick_up.object_designator.root
        return [
            drawer
            for drawer in pick_up.world.get_semantic_annotations_by_type(Drawer)
            if InsideOf(object_body, drawer.root).compute_containment_ratio()
            > self.minimum_containment_ratio
        ]

    def anchor(self, plan_node: ActionNode) -> PlanNode:
        return plan_node

    def nodes_to_insert(self, plan_node: ActionNode) -> List[ActionLike]:
        pick_up = cast(PickUpAction, plan_node.action)
        nodes = []
        for drawer in self.containing_drawers(pick_up):
            handle = drawer.handle.root
            nodes.extend(
                [
                    a(NavigateAction)(
                        target_location=variable(
                            Pose,
                            domain=reachability_location(
                                handle.global_pose, pick_up.context, pick_up.arm
                            ),
                        ),
                        keep_joint_states=True,
                    ),
                    OpenAction(handle, pick_up.arm),
                ]
            )
        return nodes
