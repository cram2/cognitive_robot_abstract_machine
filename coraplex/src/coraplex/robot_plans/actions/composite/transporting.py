from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import List

from typing_extensions import Optional, Any

from krrood.entity_query_language.factories import (
    a,
    an,
    entity,
    variable,
)
from coraplex.config.action_conf import ActionConfig
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.locations.base import DeferredLocation
from coraplex.locations.factories import reachability_location
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.composite.facing import FaceAtAction
from coraplex.robot_plans.actions.core.container import OpenAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction
from coraplex.robot_plans.parameter_mixins import (
    GraspParameters,
    MobileManipulationParameters,
    TargetLocationMovedTo,
    UsedGraspDescription,
)
from coraplex.view_manager import ViewManager
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.reasoning.predicates import InsideOf
from semantic_digital_twin.semantic_annotations.mixins import IsGraspable
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class TransportAction(
    ActionDescription,
    GraspParameters,
    TargetLocationMovedTo,
):
    """
    Transports an object to a position using an arm.
    """

    target_object: IsGraspable = field(repr=False, kw_only=True)
    """
    The graspable annotation describing the object that should be transported.
    """

    grasp_description: Optional[GraspDescription] = field(default=None, kw_only=True)
    """
    Grasp Description that should be used for picking up the object.
    """

    def inside_container(self) -> List[Body]:
        bodies = []
        target_body = self.target_object.root
        for body in self.world.bodies:
            if body == target_body:
                continue
            if InsideOf(target_body, body).compute_containment_ratio() > 0.9:
                bodies.append(body)
        return bodies

    def _make_open_container_actions(self, container: Body) -> List:
        """
        :param container: The container body in which the object is located.
        :return: The actions needed to open the given container, empty if the container is not a known drawer.
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

        return [
            a(NavigateAction)(
                target_location=variable(
                    Pose,
                    domain=reachability_location(
                        handle.root.global_pose, self.context, self.arm
                    ),
                ),
                keep_joint_states=True,
            ),
            OpenAction(handle=handle, arm=self.arm),
        ]

    @property
    def _action_plan(self) -> PlanNode:
        self.grasp_description = self.grasp_description or GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            ViewManager.get_end_effector_view(self.arm, self.robot),
        )

        children = []
        for container in self.inside_container():
            children.extend(self._make_open_container_actions(container))

        children.extend(
            [
                ParkArmsAction(arm=Arms.BOTH),
                # Tries to find a pick-up position for the robot that uses the given arm
                a(NavigateAction)(
                    target_location=variable(
                        Pose,
                        domain=DeferredLocation(
                            lambda: reachability_location(
                                self.target_object.root,
                                self.context,
                                self.arm,
                                self.grasp_description,
                            )
                        ),
                    ),
                    keep_joint_states=True,
                ),
                a(PickUpAction)(
                    target_object=self.target_object,
                    arm=self.arm,
                    grasp_description=self.grasp_description,
                ),
                ParkArmsAction(arm=Arms.BOTH),
                MoveTorsoAction(torso_state=TorsoState.HIGH),
                self._make_navigate_action_for_placing(self.grasp_description),
                a(PlaceAction)(
                    target_object=self.target_object,
                    target_location=self.target_location,
                    arm=self.arm,
                ),
                ParkArmsAction(arm=Arms.BOTH),
            ]
        )

        return sequential(children)

    def _make_navigate_action_for_placing(self, grasp_description: GraspDescription):
        """
        :param grasp_description: The grasp description that should be used for placing the object.
        :return: The navigate action that will be used to place the object.
        """
        return a(NavigateAction)(
            target_location=variable(
                Pose,
                domain=reachability_location(
                    self.target_location, self.context, self.arm, grasp_description
                ),
            ),
            keep_joint_states=True,
        )


@dataclass
class PickAndPlaceAction(
    ActionDescription,
    GraspParameters,
    TargetLocationMovedTo,
):
    """
    Transports an object to a position using an arm without moving the base of the
    robot.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return sequential(
            [
                ParkArmsAction(arm=Arms.BOTH),
                PickUpAction(
                    target_object=self.target_object,
                    arm=self.arm,
                    grasp_description=self.grasp_description,
                ),
                ParkArmsAction(arm=Arms.BOTH),
                PlaceAction(
                    target_object=self.target_object,
                    target_location=self.target_location,
                    arm=self.arm,
                ),
                ParkArmsAction(arm=Arms.BOTH),
            ]
        )


@dataclass
class MoveAndPlaceAction(
    ActionDescription,
    MobileManipulationParameters,
    TargetLocationMovedTo,
):
    """
    Navigate to `standing_position`, then turn towards the target and place the object.
    """

    keep_joint_states: bool = field(
        default=ActionConfig.navigate_keep_joint_states, kw_only=True
    )
    """
    Keep the joint states of the robot the same during the navigation.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return sequential(
            [
                NavigateAction(
                    target_location=self.standing_position,
                    keep_joint_states=self.keep_joint_states,
                ),
                FaceAtAction(
                    look_at_target=self.target_location,
                    keep_joint_states=self.keep_joint_states,
                ),
                PlaceAction(
                    target_object=self.target_object,
                    target_location=self.target_location,
                    arm=self.arm,
                ),
            ]
        )


@dataclass
class MoveAndPickUpAction(
    ActionDescription,
    MobileManipulationParameters,
    UsedGraspDescription,
):
    """
    Navigate to `standing_position`, then turn towards the object and pick it up.
    """

    keep_joint_states: bool = field(
        default=ActionConfig.navigate_keep_joint_states, kw_only=True
    )
    """
    Keep the joint states of the robot the same during the navigation.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return sequential(
            [
                NavigateAction(
                    target_location=self.standing_position,
                    keep_joint_states=self.keep_joint_states,
                ),
                FaceAtAction(
                    look_at_target=self.target_object.root.global_pose,
                    keep_joint_states=self.keep_joint_states,
                ),
                PickUpAction(
                    target_object=self.target_object,
                    arm=self.arm,
                    grasp_description=self.grasp_description,
                ),
            ]
        )
