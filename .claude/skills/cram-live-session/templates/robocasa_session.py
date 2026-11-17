# %% cell setup: a RoboCasa kitchen with a PR2 standing at a counter and an apple on it
from __future__ import annotations

import logging
from datetime import timedelta

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import PhysicallySimulatedPlanExecutor, PlanOutcome
from cramph.composites import Sequence
from cramph.context import StatechartContext
from cramph.node import StatechartNode
from cramph.statechart import Statechart
from experiments.live_session.session import LiveSession
from experiments.robocasa_kitchen_demo import (
    _spawn_robot_and_prepare_pick_up,
    load_kitchen,
)
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.adapters.robocasa_dataset.mujoco_compat import (
    robocasa_version_assertions_relaxed,
)
from semantic_digital_twin.robots.pr2 import PR2

with robocasa_version_assertions_relaxed():
    from robocasa.models.scenes.scene_registry import LayoutType, StyleType

logging.basicConfig(level=logging.INFO)
logging.getLogger("physics_simulators").setLevel(logging.WARNING)

world = load_kitchen(layout=LayoutType.LAYOUT001, style=StyleType.STYLE001)
_spawn_robot_and_prepare_pick_up(world=world)
(robot,) = world.get_semantic_annotations_by_type(PR2)
arm = robot.right_arm

# %% cell physics
simulation = MujocoSim(world=world, headless=True)
simulation.start_stepped_simulation()
simulation.step_simulation(timedelta(seconds=1))


def run(*actions: StatechartNode) -> PlanOutcome:
    """
    Perform actions one after another in the physics.

    :param actions: The actions.
    :return: How the plan ended.
    """
    executor = PhysicallySimulatedPlanExecutor(
        context=StatechartContext.with_extensions(
            world=world, extensions=[RobotAccess(robot=robot)]
        ),
        simulation=simulation,
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(node=Sequence(nodes=list(actions)))
    executor.compile(statechart=statechart)
    outcome = executor.attempt()
    executor.close()
    return outcome


# %% cell the live session
session = LiveSession(simulation=simulation)
session.take_snapshot(name="composed")
session.report(label="session started")
