from dataclasses import dataclass

from giskardpy.motion_statechart.goals.open_close import Open, Close
from giskardpy.motion_statechart.goals.templates import Parallel
from pycram.datastructures.enums import Arms
from pycram.robot_plans.motions.base import BaseMotion
from pycram.view_manager import ViewManager
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class OpeningMotion(BaseMotion):
    """
    Designator for opening container
    """

    object_part: Body
    """
    Object designator for the drawer handle
    """
    arm: Arms
    """
    Arm that should be used
    """

    def perform(self):
        return

    @property
    def _motion_chart(self):
        tip = ViewManager().get_end_effector_view(self.arm, self.robot_view).tool_frame
        motion_state_chart_nodes = self._only_allow_gripper_collision_rules(self.arm)
        motion_state_chart_nodes.append(
            Open(tip_link=tip, environment_link=self.object_part)
        )
        return Parallel(motion_state_chart_nodes)


@dataclass
class ClosingMotion(BaseMotion):
    """
    Designator for closing a container
    """

    object_part: Body
    """
    Object designator for the drawer handle
    """
    arm: Arms
    """
    Arm that should be used
    """

    def perform(self):
        return

    @property
    def _motion_chart(self):
        tip = ViewManager().get_end_effector_view(self.arm, self.robot_view).tool_frame
        motion_state_chart_nodes = self._only_allow_gripper_collision_rules(self.arm)
        motion_state_chart_nodes.append(
            Close(
                tip_link=tip, environment_link=self.object_part, goal_joint_state=0.01
            )
        )
        return Parallel(motion_state_chart_nodes)
