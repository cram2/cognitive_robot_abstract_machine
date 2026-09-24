"""
Native plan lifecycle values at the live and recording boundaries.
"""

from __future__ import annotations

import json

import pytest

from coraplex.plans.plan import Plan
from coraplex.plans.plan_node import PlanNode
from giskardpy.motion_statechart.data_types import LifeCycleValues

from cramera.live.bridge import Bridge

# %% lifecycle publication


@pytest.mark.parametrize("status", list(LifeCycleValues))
def test_plan_snapshot_keeps_the_native_lifecycle(status: LifeCycleValues) -> None:
    """
    Keep native states in memory and their names in serialized snapshots.

    :param status: The native lifecycle state to publish.
    """
    plan = Plan()
    node = PlanNode(status=status)
    plan.add_node(node)
    bridge = Bridge()

    bridge.begin_plan(plan)

    assert bridge.plan_state.nodes[0].status is status
    assert (
        json.loads(json.dumps(bridge.get_plan()))["nodes"][0]["status"] == status.name
    )
    assert (
        json.loads(json.dumps(bridge.plan_state.recorded_trees()))[0]["status"]
        == status.name
    )


@pytest.mark.parametrize("status", list(LifeCycleValues))
def test_parent_lifecycle_is_independent_of_finished_children(
    status: LifeCycleValues,
) -> None:
    """
    A parent's lifecycle remains its own when children have completed.

    :param status: The parent's current lifecycle, including a reset.
    """
    plan = Plan()
    parent = PlanNode(status=LifeCycleValues.RUNNING)
    plan.add_edge(parent, PlanNode(status=LifeCycleValues.SUCCEEDED))
    bridge = Bridge()
    bridge.begin_plan(plan)

    parent.status = status
    bridge.snapshot_plan()

    assert bridge.plan_state.nodes[0].status is status
    assert bridge.plan_state.nodes[0].derived is False
