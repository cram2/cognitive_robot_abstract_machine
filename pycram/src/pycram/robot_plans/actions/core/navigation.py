from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
from typing_extensions import Union, Optional, Type, Any, Iterable, Dict

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.factories import and_
from semantic_digital_twin.reasoning.robot_predicates import is_pose_free_for_robot
from semantic_digital_twin.robots.abstract_robot import Camera
from pycram.robot_plans.actions.base import ActionDescription, DescriptionType
from pycram.robot_plans.motions.robot_body import LookingMotion
from pycram.robot_plans.motions.navigation import MoveMotion
from pycram.config.action_conf import ActionConfig
from ....datastructures.dataclasses import Context
from pycram.datastructures.partial_designator import PartialDesignator
from pycram.datastructures.pose import PoseStamped
from pycram.failures import LookAtGoalNotReached
from pycram.failures import NavigationGoalNotReachedError
from pycram.language import SequentialPlan
from pycram.validation.error_checkers import PoseErrorChecker


@dataclass
class NavigateAction(ActionDescription):
    """
    Navigates the Robot to a position.
    """

    target_location: PoseStamped
    """
    Location to which the robot should be navigated
    """

    keep_joint_states: bool = ActionConfig.navigate_keep_joint_states
    """
    Keep the joint states of the robot the same during the navigation.
    """

    def execute(self) -> None:
        return SequentialPlan(
            self.context, MoveMotion(self.target_location, self.keep_joint_states)
        ).perform()

    @staticmethod
    def pre_condition(
        variables, context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression:
        return and_(
            is_pose_free_for_robot(
                context.robot, variables["target_location"].to_spatial_type()
            ),
            context.robot.drive is not None,
        )

    @staticmethod
    def post_condition(
        variables, context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression:
        return and_(
            np.allclose(
                context.robot.root.global_pose,
                kwargs["target_location"].to_spatial_type(),
                atol=0.03,
            )
        )

    @classmethod
    def description(
        cls,
        target_location: DescriptionType[PoseStamped],
        keep_joint_states: DescriptionType[
            bool
        ] = ActionConfig.navigate_keep_joint_states,
    ) -> PartialDesignator[NavigateAction]:
        return PartialDesignator[NavigateAction](
            NavigateAction,
            target_location=target_location,
            keep_joint_states=keep_joint_states,
        )


@dataclass
class LookAtAction(ActionDescription):
    """
    Lets the robot look at a position.
    """

    target: PoseStamped
    """
    Position at which the robot should look, given as 6D pose
    """

    camera: Camera = None
    """
    Camera that should be looking at the target
    """

    def execute(self) -> None:
        camera = self.camera or self.robot_view.get_default_camera()
        SequentialPlan(
            self.context, LookingMotion(target=self.target, camera=camera)
        ).perform()

    @classmethod
    def description(
        cls,
        target: DescriptionType[PoseStamped],
        camera: DescriptionType[Camera] = None,
    ) -> PartialDesignator[LookAtAction]:
        return PartialDesignator[LookAtAction](
            LookAtAction, target=target, camera=camera
        )


NavigateActionDescription = NavigateAction.description
LookAtActionDescription = LookAtAction.description
