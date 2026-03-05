from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
from typing_extensions import Union, Optional, Type, Any, Iterable

from semantic_digital_twin.robots.abstract_robot import Camera
from pycram.robot_plans.actions.base import ActionDescription
from pycram.robot_plans.motions.robot_body import LookingMotion
from pycram.robot_plans.motions.navigation import MoveMotion
from pycram.config.action_conf import ActionConfig
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

    def execute(self) -> None:
        return SequentialPlan(self.context, MoveMotion(self.target_location)).perform()

    @classmethod
    def description(
        cls,
        target_location: Union[Iterable[PoseStamped], PoseStamped],
    ) -> PartialDesignator[NavigateAction]:
        return PartialDesignator[NavigateAction](
            NavigateAction,
            target_location=target_location,
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

    def validate(
        self, result: Optional[Any] = None, max_wait_time: Optional[timedelta] = None
    ):
        """
        Check if the robot is looking at the target location by spawning a virtual object at the target location and
        creating a ray from the camera and checking if it intersects with the object.
        """
        return

    @classmethod
    def description(
        cls,
        target: Union[Iterable[PoseStamped], PoseStamped],
        camera: Optional[Union[Iterable[Camera], Camera]] = None,
    ) -> PartialDesignator[LookAtAction]:
        return PartialDesignator[LookAtAction](
            LookAtAction, target=target, camera=camera
        )


NavigateActionDescription = NavigateAction.description
LookAtActionDescription = LookAtAction.description
