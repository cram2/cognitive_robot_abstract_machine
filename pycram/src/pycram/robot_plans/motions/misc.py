from dataclasses import dataclass

from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from pycram.perception import PerceptionQuery
from pycram.robot_plans.motions.base import BaseMotion
from semantic_digital_twin.robots.abstract_robot import Manipulator
from semantic_digital_twin.spatial_types.spatial_types import Pose


@dataclass
class DetectingMotion(BaseMotion):
    """
    Tries to detect an object in the FOV of the robot

    returns: ObjectDesignatorDescription.Object or Error: PerceptionObjectNotFound
    """

    query: PerceptionQuery
    """
    Query for the perception system that should be answered
    """

    def perform(self):
        return

    @property
    def _motion_chart(self):
        pass


@dataclass
class MoveManipulatorMotion(BaseMotion):
    """
    Moves the Tool center point (TCP) of the robot
    """

    target: Pose
    """
    Target pose to which the TCP should be moved
    """

    manipulator: Manipulator
    """
    The Manipulator to move to the target pose
    """

    allow_gripper_collision: bool = False
    """
    If the gripper can collide with something
    """

    @property
    def _motion_chart(self):
        root = self.world.root if self.robot.full_body_controlled else self.robot.root
        task = CartesianPose(
            root_link=root,
            tip_link=self.manipulator.tool_frame,
            goal_pose=self.target,
            name=self.__class__.__name__,
        )
        return task
