from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import List

import numpy as np

from krrood.entity_query_language.factories import (
    an,
    entity,
    variable,
    underspecified,
)
from pycram.locations.factories import reachability_location
from pycram.plans.factories import sequential, execute_single
from pycram.robot_plans.actions.composite.facing import FaceAtAction
from pycram.robot_plans.actions.core.container import OpenAction
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.robot_plans.actions.core.pick_up import PickUpAction
from pycram.robot_plans.actions.core.placing import PlaceAction
from pycram.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction
from pycram.view_manager import ViewManager
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.reasoning.predicates import InsideOf, allclose
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Optional, Any

from pycram.config.action_conf import ActionConfig
from pycram.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription, GraspPose

from pycram.plans.failures import ConfigurationNotReached, BodyUnfetchable
from pycram.robot_plans.actions.base import ActionDescription


@dataclass
class TransportAction(ActionDescription):
    """
    Transports an object to a position using an arm
    """

    object_designator: Body = field(repr=False)
    """
    Object designator_description describing the object that should be transported.
    """

    target_location: Pose
    """
    Target Location to which the object should be transported
    """

    arm: Arms
    """
    Arm that should be used
    """

    grasp_description: Optional[GraspDescription] = None
    """
    Grasp Description that should be used for picking up the object
    """

    def inside_container(self) -> List[Body]:
        bodies = []
        for body in self.world.bodies:
            if body == self.object_designator:
                continue
            if InsideOf(self.object_designator, body).compute_containment_ratio() > 0.9:
                bodies.append(body)
        return bodies

    def open_container(self, container: Body):

        drawer_annotation = an(
            entity(
                drawer := variable(Drawer, domain=self.world.semantic_annotations)
            ).where(drawer.root == container)
        )
        drawer_annotation = list(drawer_annotation.evaluate())
        if len(drawer_annotation) == 0:
            return
        handle = drawer_annotation[0].handle.root

        self.add_subplan(
            sequential(
                [
                    NavigateAction(
                        reachability_location(
                            handle.global_pose, self.context, self.arm
                        ).ground(),
                        True,
                    ),
                    OpenAction(handle, self.arm),
                ]
            )
        ).perform()

    def execute(self) -> None:
        self.grasp_description = self.grasp_description or GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            ViewManager.get_end_effector_view(self.arm, self.robot),
        )

        for container in self.inside_container():
            self.open_container(container)

        self.add_subplan(execute_single(ParkArmsAction(Arms.BOTH))).perform()

        pickup_loc = reachability_location(
            self.object_designator.global_pose,
            self.context,
            self.arm,
            self.grasp_description,
        )
        # Tries to find a pick-up position for the robot that uses the given arm

        pickup_pose = pickup_loc.ground()

        if not pickup_pose:
            raise BodyUnfetchable(self.object_designator, self.arm)

        self.add_subplan(
            sequential(
                [
                    NavigateAction(pickup_pose, True),
                    PickUpAction(
                        self.object_designator,
                        self.arm,
                        grasp_description=self.grasp_description,
                    ),
                    ParkArmsAction(Arms.BOTH),
                    MoveTorsoAction(TorsoState.HIGH),
                ]
            )
        ).perform()

        self.add_subplan(self._make_place_plan()).perform()

    def _make_place_plan(self):

        return sequential(
            children=[
                self._make_navigate_action_for_placing(self.grasp_description),
                PlaceAction(self.object_designator, self.target_location, self.arm),
                ParkArmsAction(Arms.BOTH),
            ]
        )

    def _make_navigate_action_for_placing(self, grasp_description: GraspDescription):
        """
        :param grasp_description: The grasp description that should be used for placing the object.
        :return: The navigate action that will be used to place the object.
        """
        return underspecified(NavigateAction)(
            target_location=variable(
                Pose,
                domain=reachability_location(
                    self.target_location, self.context, self.arm, self.grasp_description
                ),
            ),
            keep_joint_states=True,
        )

    def validate(
        self, result: Optional[Any] = None, max_wait_time: Optional[timedelta] = None
    ):
        # The validation of each core action is done in the action itself, so no more validation needed here.
        pass


@dataclass
class PickAndPlaceAction(ActionDescription):
    """
    Transports an object to a position using an arm without moving the base of the robot
    """

    object_designator: Body
    """
    Object designator_description describing the object that should be transported.
    """
    target_location: Pose
    """
    Target Location to which the object should be transported
    """
    arm: Arms
    """
    Arm that should be used
    """
    grasp_description: GraspDescription
    """
    Description of the grasp to pick up the target
    """

    def execute(self) -> None:
        self.add_subplan(
            sequential(
                [
                    ParkArmsAction(Arms.BOTH),
                    PickUpAction(
                        self.object_designator,
                        self.arm,
                        grasp_description=self.grasp_description,
                    ),
                    ParkArmsAction(Arms.BOTH),
                    PlaceAction(self.object_designator, self.target_location, self.arm),
                    ParkArmsAction(Arms.BOTH),
                ]
            )
        ).perform()

    def validate(
        self, result: Optional[Any] = None, max_wait_time: Optional[timedelta] = None
    ):
        if self.object_designator.pose.__eq__(self.target_location):
            pass
        else:
            raise ValueError("Object not moved to the target location")


@dataclass
class MoveAndPlaceAction(ActionDescription):
    """
    Navigate to `standing_position`, then turn towards the object and pick it up.
    """

    standing_position: Pose
    """
    The pose to stand before trying to pick up the object
    """

    object_designator: Body
    """
    The object to pick up
    """

    target_location: Pose
    """
    The location to place the object.
    """

    arm: Arms
    """
    The arm to use
    """

    keep_joint_states: bool = ActionConfig.navigate_keep_joint_states
    """
    Keep the joint states of the robot the same during the navigation.
    """

    def execute(self):
        self.add_subplan(
            sequential(
                [
                    NavigateAction(self.standing_position, self.keep_joint_states),
                    FaceAtAction(self.target_location, self.keep_joint_states),
                    PlaceAction(self.object_designator, self.target_location, self.arm),
                ]
            )
        ).perform()

    def validate(
        self, result: Optional[Any] = None, max_wait_time: Optional[timedelta] = None
    ):
        # The validation will be done in each of the core action perform methods so no need to validate here.
        pass


@dataclass
class MoveAndPickUpAction(ActionDescription):
    """
    Navigate to `standing_position`, then turn towards the object and pick it up.
    """

    standing_position: Pose
    """
    The pose to stand before trying to pick up the object
    """

    object_designator: Body
    """
    The object to pick up
    """

    arm: Arms
    """
    The arm to use
    """

    grasp_description: GraspDescription
    """
    The grasp to use
    """

    keep_joint_states: bool = ActionConfig.navigate_keep_joint_states
    """
    Keep the joint states of the robot the same during the navigation.
    """

    def execute(self):
        obj_pose = Pose.from_spatial_type(self.object_designator.global_transform)
        self.add_subplan(
            sequential(
                [
                    NavigateAction(self.standing_position, self.keep_joint_states),
                    FaceAtAction(obj_pose, self.keep_joint_states),
                    PickUpAction(
                        self.object_designator, self.arm, self.grasp_description
                    ),
                ]
            )
        ).perform()

    def validate(
        self, result: Optional[Any] = None, max_wait_time: Optional[timedelta] = None
    ):
        # The validation will be done in each of the core action perform methods so no need to validate here.
        pass
