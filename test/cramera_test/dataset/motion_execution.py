"""
Native plan and history fixtures for browser execution observers.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from coraplex.plans.plan import Plan
from coraplex.plans.plan_node import MotionNode, PlanNode
from coraplex.robot_plans.motions.base import BaseMotion
from giskardpy.motion_statechart.data_types import LifeCycleValues
from giskardpy.motion_statechart.graph_node import Goal
from giskardpy.motion_statechart.motion_statechart import (
    MotionStatechart,
    StateHistoryItem,
)

from cramera.live.bridge import Bridge
from cramera.live.visualization import BridgePlanCallback

# %% native motion execution fixture


@dataclass
class MotionExecution:
    """
    A plan and its native motion chart sharing one visualization callback.
    """

    plan: Plan
    """The plan whose root and motion delimit publication."""

    motion: MotionNode
    """
    The motion bound to the chart before its start notification.
    """

    chart: MotionStatechart
    """The native chart that records lifecycle changes."""

    bridge: Bridge
    """
    The published plan and chart state.
    """

    callback: BridgePlanCallback
    """The subscriber observing this plan's motion history."""

    def record(self, state: LifeCycleValues) -> None:
        """
        Record one lifecycle state using native snapshot copying.

        :param state: The state assigned to every chart node.
        """
        self.chart.life_cycle_state.data[:] = state
        self.chart.history.append(
            StateHistoryItem(
                control_cycle=len(self.chart.history),
                life_cycle_state=self.chart.life_cycle_state,
                observation_state=self.chart.observation_state,
            )
        )


@pytest.fixture()
def motion_execution() -> MotionExecution:
    """
    Build a native plan and chart without starting a motion controller.
    """
    plan = Plan()
    motion = MotionNode(designator=BaseMotion())
    plan.add_edge(PlanNode(), motion)
    chart = MotionStatechart()
    chart.add_node(Goal(name="Transport"))
    motion.motion_statechart = chart
    bridge = Bridge()
    bridge.begin_plan(plan)
    callback = BridgePlanCallback(bridge=bridge, plan=plan)
    return MotionExecution(plan, motion, chart, bridge, callback)
