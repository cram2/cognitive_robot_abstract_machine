from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from typing_extensions import Optional, Any

from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from pycram.config.action_conf import ActionConfig
from pycram.plans.factories import execute_single
from pycram.robot_plans.actions.base import ActionDescription
from pycram.robot_plans.motions.navigation import MoveMotion
from pycram.robot_plans.motions.robot_body import LookingMotion
from semantic_digital_twin.robots.abstract_robot import Camera, Manipulator
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

    def validate(
        self, result: Optional[Any] = None, max_wait_time: Optional[timedelta] = None
    ):
        """
        Check if the robot is looking at the target location by spawning a virtual object at the target location and
        creating a ray from the camera and checking if it intersects with the object.
        """
        return
