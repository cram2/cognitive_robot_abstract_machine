from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from typing_extensions import Optional, Any, Dict

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.factories import variable_from, and_
from pycram.config.action_conf import ActionConfig
from pycram.datastructures.dataclasses import Context
from pycram.plans.factories import execute_single
from pycram.robot_plans.actions.base import ActionDescription
from pycram.robot_plans.motions.navigation import MoveMotion
from pycram.robot_plans.motions.robot_body import LookingMotion
from semantic_digital_twin.reasoning.predicates import allclose
from semantic_digital_twin.reasoning.robot_predicates import is_pose_free_for_robot
from semantic_digital_twin.robots.abstract_robot import Camera
from semantic_digital_twin.spatial_types.spatial_types import Pose


@dataclass
class NavigateAction(ActionDescription):
    """
    Navigates the Robot to a position.
    """

    target_location: Pose
    """
    Location to which the robot should be navigated
    """

    keep_joint_states: bool = ActionConfig.navigate_keep_joint_states
    """
    Keep the joint states of the robot the same during the navigation.
    """

    def execute(self) -> None:
        self.add_subplan(
            execute_single(MoveMotion(self.target_location, self.keep_joint_states))
        ).perform()

    @staticmethod
    def pre_condition(
        variables, context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression:
        """
        The robot needs to have a drive and the target location needs to be free from obstacles
        """
        return True
        drive_variable = variable_from(context.robot.drive is not None)
        return and_(
            is_pose_free_for_robot(context.robot, variables["target_location"]),
            drive_variable,
        )

    @staticmethod
    def post_condition(
        variables, context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression:
        """
        The robot needs to be within 3 cm of the target location
        """

        return allclose(
            context.robot.root.global_pose,
            kwargs["target_location"],
            atol=0.03,
        )


@dataclass
class LookAtAction(ActionDescription):
    """
    Lets the robot look at a position.
    """

    target: Pose
    """
    Position at which the robot should look, given as 6D pose
    """

    camera: Camera = None
    """
    Camera that should be looking at the target
    """

    def execute(self) -> None:
        camera = self.camera or self.robot.get_default_camera()
        self.add_subplan(
            execute_single(LookingMotion(target=self.target, camera=camera))
        ).perform()
