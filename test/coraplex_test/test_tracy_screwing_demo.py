"""
Tests for the scenes and statecharts of ``coraplex_tracy_screwing_demo``.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from datetime import timedelta
from pathlib import Path
from types import ModuleType

import pytest

from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import KinematicallySimulatedPlanExecutor
from cramph.context import StatechartContext
from cramph.executor import NoPacing
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bolt
from semantic_digital_twin.world_description.connections import ScrewConnection

# %% the demo, loaded from its folder


@pytest.fixture(scope="module")
def demo() -> ModuleType:
    """
    The screwing demo module loaded from its folder with its folder on ``sys.path``.
    """
    folder = (
        Path(__file__).parents[2]
        / "coraplex"
        / "demos"
        / "coraplex_tracy_screwing_demo"
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(folder))
        specification = importlib.util.spec_from_file_location(
            "tracy_screwing_demo", folder / "demo.py"
        )
        module = importlib.util.module_from_spec(specification)
        patch.setitem(sys.modules, specification.name, module)
        specification.loader.exec_module(module)
    return module


def _run_scene(*, scene: object, time_limit: timedelta) -> Bolt:
    """
    Build ``scene``, execute its statechart kinematically, and return the spawned bolt.

    :param scene: Scene instance providing ``build_world`` and ``build_statechart``.
    :param time_limit: Simulated time budget per action.
    :return: The :class:`Bolt` annotation in the executed world.
    """
    world, tracy = scene.build_world()
    context = StatechartContext.with_extensions(
        world=world, extensions=[RobotAccess(robot=tracy)]
    )
    statechart = scene.build_statechart(context=context, tracy=tracy)
    executor = KinematicallySimulatedPlanExecutor(
        context=context,
        simulation_time_per_action=time_limit,
        pacer=NoPacing(),
    )
    try:
        executor.compile(statechart=statechart)
        executor.execute()
    finally:
        executor.close()
    [bolt] = world.get_semantic_annotations_by_type(Bolt)
    return bolt


# %% scene execution


def test_nut_scene_screws_bolt_into_plate(demo: ModuleType) -> None:
    """
    Verify :class:`NutScene` drives the bolt to the configured inward turn count.
    """
    scene = demo.NutScene()
    bolt = _run_scene(scene=scene, time_limit=timedelta(minutes=5))
    connection = bolt.root.get_first_parent_connection_of_type(ScrewConnection)
    assert connection is not None
    assert connection.position == pytest.approx(-2.0 * math.pi * scene.turns, abs=1e-2)


def test_airplane_scene_inserts_and_tightens_bolt_with_screwdriver(
    demo: ModuleType,
) -> None:
    """
    Verify :class:`AirplaneScene` inserts the bolt into the wing and tightens it with
    the screwdriver.
    """
    scene = demo.AirplaneScene()
    bolt = _run_scene(scene=scene, time_limit=timedelta(minutes=5))
    connection = bolt.root.get_first_parent_connection_of_type(ScrewConnection)
    assert connection is not None
    assert (
        bolt.root.parent_kinematic_structure_entity.name.name == scene.screw_parent_name
    )
    assert connection.position == pytest.approx(-2.0 * math.pi * scene.turns, abs=1e-2)
