from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from typing_extensions import Any, Dict

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.factories import and_
from pycram.config.action_conf import ActionConfig
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import (
    Arms,
    ApproachDirection,
    VerticalAlignment,
)
from pycram.datastructures.grasp import GraspDescription
from pycram.plans.factories import sequential
from pycram.pose_validator import reachability_validator
from pycram.querying.predicates import GripperIsFree
from pycram.robot_plans.actions.base import ActionDescription, DescriptionType
from pycram.robot_plans.actions.core.pick_up import GraspingAction
from pycram.robot_plans.motions.container import OpeningMotion, ClosingMotion
from pycram.robot_plans.motions.gripper import MoveGripperMotion
from pycram.view_manager import ViewManager
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.reasoning.robot_predicates import is_body_in_gripper
from semantic_digital_twin.robots.abstract_robot import HasMobileBase
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class OpenAction(ActionDescription):
    """
    Opens a container like object
    """

    object_designator: Body
    """
    Object designator_description describing the object that should be opened
    """
    arm: Arms
    """
    Arm that should be used for opening the container
    """
    grasping_prepose_distance: float = ActionConfig.grasping_prepose_distance
    """
    The distance in meters the gripper should be at in the x-axis away from the handle.
    """

    def execute(self) -> None:
        arm = ViewManager.get_arm_view(self.arm, self.robot)
        manipulator = arm.manipulator

        grasp_description = GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            manipulator,
        )

        self.add_subplan(
            sequential(
                [
                    GraspingAction(self.object_designator, self.arm, grasp_description),
                    OpeningMotion(self.object_designator, self.arm),
                    MoveGripperMotion(
                        GripperState.OPEN, self.arm, allow_gripper_collision=True
                    ),
                ]
            )
        ).perform()

    @staticmethod
    def pre_condition(
        variables, context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression:
        """
        The gripper with which to open the container has to be free and the handle has to be reachable.
        """
        manipulator = ViewManager.get_end_effector_view(variables["arm"], context.robot)
        test_world = deepcopy(context.world)

        return and_(
            GripperIsFree(manipulator),
            reachability_validator(
                kwargs["object_designator"].global_pose,
                manipulator.tool_frame,
                test_world.get_semantic_annotations_by_type(type(context.robot))[0],
                test_world,
                (
                    context.robot.full_body_controlled
                    if isinstance(context.robot, HasMobileBase)
                    else False
                ),
            ),
        )

    @staticmethod
    def post_condition(
        variables, context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression | bool:
        """
        The handle has to be in the gripper of the robot and the container has to be open.
        """
        manipulator = ViewManager.get_end_effector_view(kwargs["arm"], context.robot)
        parent_connection = kwargs[
            "object_designator"
        ].get_first_parent_connection_of_type(ActiveConnection1DOF)
        return (
            is_body_in_gripper(kwargs["object_designator"], manipulator) > 0.9
            or np.allclose(
                kwargs["object_designator"].global_pose.to_position(),
                ViewManager.get_end_effector_view(
                    kwargs["arm"], context.robot
                ).tool_frame.global_pose.to_position(),
                atol=3e-2,
            )
        ) and bool(parent_connection.position > 0.3)


@dataclass
class CloseAction(ActionDescription):
    """
    Closes a container like object.
    """

    object_designator: Body
    """
    Object designator_description describing the object that should be closed
    """
    arm: Arms
    """
    Arm that should be used for closing
    """
    grasping_prepose_distance: float = ActionConfig.grasping_prepose_distance
    """
    The distance in meters between the gripper and the handle before approaching to grasp.
    """

    def execute(self) -> None:
        arm = ViewManager.get_arm_view(self.arm, self.robot)
        manipulator = arm.manipulator

        grasp_description = GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            manipulator,
        )

        self.add_subplan(
            sequential(
                [
                    GraspingAction(self.object_designator, self.arm, grasp_description),
                    ClosingMotion(self.object_designator, self.arm),
                    MoveGripperMotion(
                        GripperState.OPEN, self.arm, allow_gripper_collision=True
                    ),
                ]
            )
        ).perform()

    @staticmethod
    def post_condition(
        variables, context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression | bool:
        """
        The container has to be closed
        """
        close_connection = kwargs[
            "object_designator"
        ].get_first_parent_connection_of_type(ActiveConnection1DOF)

        return bool(close_connection.position < 0.1)
