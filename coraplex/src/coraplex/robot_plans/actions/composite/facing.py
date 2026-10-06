from __future__ import annotations

from dataclasses import dataclass

from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.navigation import FaceAtAction, LookAtAction


@dataclass
class FaceAndLookAtAction(ActionDescription):
    """
    Turns the robot's base towards a target, then looks at it.
    """

    face_at: FaceAtAction
    """
    The turn of the base towards the target.
    """

    look_at: LookAtAction
    """
    The look at the target once the base faces it.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return sequential([self.face_at, self.look_at])
