from dataclasses import dataclass

from giskardpy.motion_statechart.goals.open_close import Open
from pycram.config.action_conf import ActionConfig
from pycram.plans.factories import sequential
from pycram.robot_plans import OpeningMotion, MoveGripperMotion, BaseMotion
from pycram.robot_plans.actions.base import ActionDescription
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.abstract_robot import Manipulator
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class OpenWithHandleMotion(BaseMotion):
    """
    Designator for opening container
    """

    handle: Body

    manipulator: Manipulator

    @property
    def _motion_chart(self):
        return Open(tip_link=self.manipulator.tool_frame, environment_link=self.handle)


@dataclass
class OpenWithHandleAction(ActionDescription):
    """
    Opens a container like object
    """

    handle: Handle
    manipulator: Manipulator

    def execute(self) -> None: ...
