from dataclasses import dataclass

from giskardpy.motion_statechart.goals.open_close import Open, Close
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle

from coraplex.robot_plans.motions.base import BaseMotion
from coraplex.robot_plans.parameter_mixins import HandleOperationParameters
from coraplex.datastructures.enums import Arms
from coraplex.view_manager import ViewManager


@dataclass
class OpeningMotion(BaseMotion, HandleOperationParameters):
    """
    Designator for opening container.
    """

    def perform(self):
        return

    @property
    def _motion_chart(self):
        tip = ViewManager().get_end_effector_view(self.arm, self.robot).tool_frame
        return Open(tip_link=tip, environment_link=self.handle.root)


@dataclass
class ClosingMotion(BaseMotion, HandleOperationParameters):
    """
    Designator for closing a container.
    """

    def perform(self):
        return

    @property
    def _motion_chart(self):
        tip = ViewManager().get_end_effector_view(self.arm, self.robot).tool_frame
        return Close(
            tip_link=tip, environment_link=self.handle.root, goal_joint_state=0.01
        )
