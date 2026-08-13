from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

from coraplex.plans.plan import Plan
from coraplex.plans.plan_entity import PlanEntity
from coraplex.plans.plan_node import PlanNode

if TYPE_CHECKING:
    from giskardpy.motion_statechart.motion_statechart import MotionStatechart


@dataclass
class PlanCallback(PlanEntity):
    """
    Observer of a plan's execution, notified as its nodes start, end, and tick.

    Subclasses override the events they care about; every event defaults to a no-op.
    """

    def on_start(self, node: PlanNode): ...

    def on_end(self, node: PlanNode): ...

    def on_motion_tick(self, statechart: MotionStatechart): ...
