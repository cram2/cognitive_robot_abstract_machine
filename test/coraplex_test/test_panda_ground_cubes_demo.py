"""
Coverage for the Panda ground-cubes demonstration.

The demonstration builds its own MuJoCo simulation and drives a real motion through
Giskard, so the one thing worth proving here is that the arm actually reaches the parked
joint state it asks for -- not merely that the run finishes without raising.
"""

import sys
from pathlib import Path

import pytest

from coraplex.datastructures.enums import ExecutionType
from coraplex.execution_environment import ExecutionEnvironment
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.robots.panda import Panda

try:
    from ament_index_python.packages import (
        PackageNotFoundError,
        get_package_share_directory,
    )

    get_package_share_directory("iai_franka_panda_description")
    panda_description_package_available = True
except (ImportError, ModuleNotFoundError, PackageNotFoundError):
    panda_description_package_available = False

PANDA_DEMO_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "coraplex" / "demos" / "coraplex_panda_demo"
)

if str(PANDA_DEMO_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PANDA_DEMO_DIRECTORY))


@pytest.mark.skipif(
    not panda_description_package_available,
    reason="iai_franka_panda_description is not available in this ROS workspace",
)
def test_park_arms_is_actually_reached(monkeypatch):
    """
    The plan must leave the arm at its parked joint state, not merely return without
    raising.

    The check happens right as the plan finishes rather than after
    :meth:`~coraplex.demonstrations.RobotDemonstration.run` returns: ``run`` holds the
    simulation open for a couple of seconds afterwards so a non-headless viewer can see
    the final pose, and with no active motion holding it there, the arm drifts under
    gravity during that window.

    ``real_time_factor`` is left at its default (unpaced) rather than the ``1.0`` the
    demo's ``main`` uses for a nicer-looking run: pacing the tick loop to the wall clock
    makes how many ticks land before the motion's completion monitor fires -- and
    therefore how tightly the last joint converges -- sensitive to thread-scheduling
    jitter, which flips this assertion between runs.
    """
    monkeypatch.setenv("CI", "true")
    from demo import PandaSimpleDemo

    demonstration = PandaSimpleDemo(
        used_robot=Panda,
        execution_type=ExecutionType.SIMULATED,
        prediction_horizon=20,
    )

    world = demonstration.acquire_world()
    try:
        demonstration.populate_scene(world)
        plan = demonstration.build_plan(demonstration.build_context(world))
        with ExecutionEnvironment(
            execution_type=demonstration.execution_type,
            real_time_factor=demonstration.real_time_factor,
            prediction_horizon=demonstration.prediction_horizon,
        ):
            plan.perform()

        panda = world.get_semantic_annotations_by_type(Panda)[0]
        arm = panda.get_arms()[0]
        park_state = arm.get_joint_state_by_type(StaticJointState.PARK)

        assert park_state.is_achieved()
    finally:
        demonstration.tear_down()
