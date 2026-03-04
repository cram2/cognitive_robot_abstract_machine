from dataclasses import dataclass

from pycram.perception import PerceptionQuery
from pycram.robot_plans.motions.base import BaseMotion


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
