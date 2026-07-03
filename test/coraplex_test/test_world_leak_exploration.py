"""Regression guards for World retention in the coraplex world/plan paths.

Background: the module-scoped ``count_worlds`` guard raises when more than 30 ``World`` objects are
live. Exploration showed the two paths that create Worlds per test -- deepcopying a world
(``mutable_model_world``) and executing a plan (``perform`` -> a simulated ``BulletCollisionDetector``
world) -- both *release* their Worlds again, so they do not accumulate across tests. These tests pin
that: a real regression (a retained World per test) would make them fail.
"""
import gc
from copy import deepcopy

import objgraph

from coraplex.datastructures.enums import Arms
from coraplex.motion_executor import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction
from semantic_digital_twin.datastructures.definitions import TorsoState


def _live_worlds() -> int:
    gc.collect()
    return objgraph.count("World")


def _describe_referrers(world) -> dict:
    """Group the direct referrers of a leaked world by type, for failure diagnosis."""
    kinds: dict = {}
    for referrer in gc.get_referrers(world):
        key = type(referrer).__qualname__
        kinds[key] = kinds.get(key, 0) + 1
    return kinds


def test_deepcopy_apartment_world_is_released(pr2_apartment_world):
    """Deep-copying a world and dropping the copy must not retain it."""
    before = _live_worlds()
    for _ in range(10):
        copy = deepcopy(pr2_apartment_world)
        del copy
    after = _live_worlds()
    assert after == before, (
        f"deepcopy retained {after - before} world(s); "
        f"referrers: {[_describe_referrers(w) for w in objgraph.by_type('World')[-3:]]}"
    )


def test_performing_a_plan_does_not_accumulate_worlds(mutable_model_world):
    """Executing a plan spawns a simulation world (a ``BulletCollisionDetector`` world); each one
    must be released before the next execution, so the pre-execution World count stays flat.
    """
    _world, _robot_view, context = mutable_model_world
    baselines = []
    for _ in range(3):
        baselines.append(_live_worlds())
        plan = sequential(
            [MoveTorsoAction(TorsoState.HIGH), ParkArmsAction(Arms.BOTH)],
            context=context,
        ).plan
        with simulated_robot:
            plan.perform()
        del plan

    assert max(baselines) - min(baselines) <= 1, (
        f"World count climbed across plan executions ({baselines}); an execution retained its "
        f"simulation world. referrers: "
        f"{[_describe_referrers(w) for w in objgraph.by_type('World')[-3:]]}"
    )
