from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing_extensions import Optional, Any

from krrood.entity_query_language.factories import a, variable
from coraplex.config.action_conf import ActionConfig
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.locations.base import DeferredLocation
from coraplex.locations.factories import reachability_location
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.composite.facing import FaceAtAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction
from coraplex.view_manager import ViewManager
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class TransportAction(ActionDescription):
    """
    Transports an object to a position using an arm.
    """

    object_designator: HasRootBody = field(repr=False)
    """
    The annotation of the object that should be transported.
    """

    target_location: Pose
    """
    Target Location to which the object should be transported.
    """

    arm: Arms
    """
    Arm that should be used.
    """

    grasp_description: Optional[GraspDescription] = None
    """
    Grasp Description that should be used for picking up the object.
    """

    @property
    def _action_plan(self) -> PlanNode:
        self.grasp_description = self.grasp_description or GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            ViewManager.get_end_effector_view(self.arm, self.robot),
        )

        children = [
            ParkArmsAction(Arms.BOTH),
            # Tries to find a pick-up position for the robot that uses the given arm
            a(NavigateAction)(
                target_location=variable(
                    Pose,
                    domain=DeferredLocation(
                        lambda: reachability_location(
                            self.object_designator.root,
                            self.context,
                            self.arm,
                            self.grasp_description,
                        )
                    ),
                ),
                keep_joint_states=True,
            ),
            a(PickUpAction)(
                object_designator=self.object_designator,
                arm=self.arm,
                grasp_description=self.grasp_description,
            ),
            ParkArmsAction(Arms.BOTH),
            MoveTorsoAction(TorsoState.HIGH),
            self._make_navigate_action_for_placing(self.grasp_description),
            a(PlaceAction)(
                object_designator=self.object_designator.root,
                target_location=self.target_location,
                arm=self.arm,
            ),
            ParkArmsAction(Arms.BOTH),
        ]

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
class PickAndPlaceAction(ActionDescription):
    """
    Transports an object to a position using an arm without moving the base of
    the robot.
    """

    object_designator: HasRootBody
    """
    The annotation of the object that should be transported.
    """

    target_location: Pose
    """
    Target Location to which the object should be transported.
    """

    arm: Arms
    """
    Arm that should be used.
    """
    grasp_description: GraspDescription
    """
    Description of the grasp to pick up the target.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return sequential(
            [
                ParkArmsAction(Arms.BOTH),
                PickUpAction(
                    self.object_designator,
                    self.arm,
                    grasp_description=self.grasp_description,
                ),
                ParkArmsAction(Arms.BOTH),
                PlaceAction(
                    self.object_designator.root, self.target_location, self.arm
                ),
                ParkArmsAction(Arms.BOTH),
            ]
        )


@dataclass
class MoveAndPlaceAction(ActionDescription):
    """
    Navigate to `standing_position`, then turn towards the target and place the
    object.
    """

    standing_position: Pose
    """
    The pose to stand before trying to pick up the object.
    """
    object_designator: Body
    """
    The object to pick up.
    """
    target_location: Pose
    """
    The location to place the object.
    """
    arm: Arms
    """
    The arm to use.
    """

    keep_joint_states: bool = ActionConfig.navigate_keep_joint_states
    """
    Keep the joint states of the robot the same during the navigation.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return sequential(
            [
                NavigateAction(self.standing_position, self.keep_joint_states),
                FaceAtAction(self.target_location, self.keep_joint_states),
                PlaceAction(self.object_designator, self.target_location, self.arm),
            ]
        )


@dataclass
class MoveAndPickUpAction(ActionDescription):
    """
    Navigate to `standing_position`, then turn towards the object and pick it
    up.
    """

    standing_position: Pose
    """
    The pose to stand before trying to pick up the object.
    """
    object_designator: HasRootBody
    """
    The annotation of the object to pick up.
    """
    arm: Arms
    """
    The arm to use.
    """
    grasp_description: GraspDescription
    """
    The grasp to use.
    """

    keep_joint_states: bool = ActionConfig.navigate_keep_joint_states
    """
    Keep the joint states of the robot the same during the navigation.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return sequential(
            [
                NavigateAction(self.standing_position, self.keep_joint_states),
                FaceAtAction(
                    self.object_designator.root.global_pose, self.keep_joint_states
                ),
                PickUpAction(self.object_designator, self.arm, self.grasp_description),
            ]
        )
