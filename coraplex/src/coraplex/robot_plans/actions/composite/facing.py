from __future__ import annotations

from dataclasses import dataclass


from cramph.node import StatechartNode
from coraplex.robot_plans.actions.base import SequentialAction
from coraplex.robot_plans.actions.core.navigation import FaceAtAction, LookAtAction


@dataclass(eq=False, repr=False, kw_only=True)
class FaceAndLookAtAction(SequentialAction):
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

    def create_steps(self) -> list[StatechartNode]:
        return [self.face_at, self.look_at]
