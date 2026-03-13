from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from typing_extensions import Any, Dict

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.factories import or_, not_, and_
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import (
    Arms,
    ApproachDirection,
    VerticalAlignment,
)
from pycram.datastructures.grasp import GraspDescription
from pycram.datastructures.partial_designator import PartialDesignator
from pycram.datastructures.pose import PoseStamped
from pycram.language import SequentialPlan
from pycram.querying.predicates import GripperIsFree
from pycram.robot_plans.actions.base import ActionDescription, DescriptionType
from pycram.robot_plans.actions.core.pick_up import ReachActionDescription, PickUpAction
from pycram.robot_plans.motions.gripper import MoveTCPMotion, MoveGripperMotion
from pycram.view_manager import ViewManager
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.reasoning.robot_predicates import is_body_in_gripper
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class PlaceAction(ActionDescription):
    """
    Places an Object at a position using an arm.
    """

    object_designator: Body
    """
    Object designator_description describing the object that should be place
    """
    target_location: PoseStamped
    """
    Pose in the world at which the object should be placed
    """
    arm: Arms
    """
    Arm that is currently holding the object
    """
    _pre_perform_callbacks = []
    """
    List to save the callbacks which should be called before performing the action.
    """

    def execute(self) -> None:
        arm = ViewManager.get_arm_view(self.arm, self.robot_view)
        manipulator = arm.manipulator

        previous_pick = self.plan.get_previous_node_by_designator_type(
            self.plan_node, PickUpAction
        )
        previous_grasp = (
            previous_pick.designator_ref.grasp_description
            if previous_pick
            else GraspDescription(
                ApproachDirection.FRONT, VerticalAlignment.NoAlignment, manipulator
            )
        )

        SequentialPlan(
            self.context,
            ReachActionDescription(
                self.target_location,
                self.arm,
                previous_grasp,
                self.object_designator,
                reverse_reach_order=True,
            ),
            MoveGripperMotion(GripperState.OPEN, self.arm),
        ).perform()

        # Detaches the object from the robot
        world_root = self.world.root
        obj_transform = self.world.compute_forward_kinematics(
            world_root, self.object_designator
        )
        with self.world.modify_world():
            self.world.remove_connection(self.object_designator.parent_connection)
            connection = Connection6DoF.create_with_dofs(
                parent=world_root, child=self.object_designator, world=self.world
            )
            self.world.add_connection(connection)
            connection.origin = obj_transform

        _, _, retract_pose = previous_grasp._pose_sequence(
            self.target_location, self.object_designator, reverse=True
        )

        SequentialPlan(self.context, MoveTCPMotion(retract_pose, self.arm)).perform()

    @staticmethod
    def pre_condition(
        variables, context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression:
        manipulator = ViewManager.get_end_effector_view(variables["arm"], context.robot)
        return or_(
            not_(GripperIsFree(manipulator)),
            is_body_in_gripper(kwargs["object_designator"], manipulator) > 0.9,
        )

    @staticmethod
    def post_condition(
        variables, context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression:
        manipulator = ViewManager.get_end_effector_view(variables["arm"], context.robot)
        return and_(
            GripperIsFree(manipulator),
            is_body_in_gripper(kwargs["object_designator"], manipulator) < 0.1,
            np.allclose(
                kwargs["object_designator"].global_pose,
                kwargs["target_location"].to_spatial_type(),
                atol=0.03,
            ),
        )

    @classmethod
    def description(
        cls,
        object_designator: DescriptionType[Body],
        target_location: DescriptionType[PoseStamped],
        arm: DescriptionType[Arms],
    ) -> PartialDesignator[PlaceAction]:
        return PartialDesignator[PlaceAction](
            PlaceAction,
            object_designator=object_designator,
            target_location=target_location,
            arm=arm,
        )


PlaceActionDescription = PlaceAction.description
