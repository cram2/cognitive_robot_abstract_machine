from dataclasses import dataclass

from giskardpy.motion_statechart.goals.open_close import Open, Close
from semantic_digital_twin.world_description.world_entity import Body

from coraplex.robot_plans.motions.base import BaseMotion
from coraplex.datastructures.enums import Arms
from coraplex.view_manager import ViewManager


@dataclass
class OpeningMotion(BaseMotion):
    """
    Designator for opening container.
    """

    object_part: Body
    """
    Object designator for the drawer handle
    """
    arm: Arms
    """
    Arm that should be used.
    """

    def perform(self):
        return

    @property
    def _motion_chart(self):
        tip = ViewManager().get_end_effector_view(self.arm, self.robot).tool_frame
        return Open(tip_link=tip, environment_link=self.object_part)


@dataclass
class ClosingMotion(BaseMotion):
    """
    Designator for closing a container.
    """

    object_part: Body
    """
    Object designator for the drawer handle.
    """

    arm: Arms
    """
    Arm that should be used.
    """

    def perform(self):
        return

    @property
    def _motion_chart(self):
        tip = ViewManager().get_end_effector_view(self.arm, self.robot).tool_frame
        return Close(
            tip_link=tip, environment_link=self.object_part, goal_joint_state=0.01
        )
