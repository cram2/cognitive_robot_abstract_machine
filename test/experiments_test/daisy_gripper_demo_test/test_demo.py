"""
Plan structure of the DAiSy gripper demonstration.

The demonstration is a plan about gripper motions, so what a run commands is decided
entirely when the plan is built: which gripper is commanded, into which state, and
whether the two grippers move together. Only the simulated run is performed here; the
others need a controller.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from coraplex.griplink_grippers import RealDAiSyPlanExecutor
from coraplex.plans.executors import (
    KinematicallySimulatedPlanExecutor,
    RealRobotPlanExecutor,
)
from coraplex.robot_plans.actions.core.robot_body import (
    ParkArmsAction,
    SetGripperAction,
)
from cramph.composites import Parallel, Sequence
from cramph.context import StatechartContext
from cramph.monitors import CountSimulationTimeSeconds, Print
from cramph.node import StatechartNode
from experiments.daisy_gripper_demo.demo import (
    DAiSyGripperDemonstration,
    ExecutionChoice,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.daisy import (
    DAiSy,
    DAiSyLeftGripper,
    DAiSyRightGripper,
)
from semantic_digital_twin.utils import daisy_installed
from semantic_digital_twin.world import World

pytestmark = pytest.mark.skipif(not daisy_installed(), reason="DAiSy not installed")

# %% the demonstration


@pytest.fixture
def demonstration() -> DAiSyGripperDemonstration:
    return DAiSyGripperDemonstration(used_robot=DAiSy)


@pytest.fixture
def demonstration_world(demonstration: DAiSyGripperDemonstration) -> World:
    return demonstration.build_simulated_world()


@pytest.fixture
def plan(
    demonstration: DAiSyGripperDemonstration, demonstration_world: World
) -> Sequence:
    """
    The top-level node of the plan the demonstration performs.
    """
    context = StatechartContext.with_extensions(
        demonstration_world,
        demonstration.build_context_extensions(demonstration_world),
    )
    [plan] = demonstration.build_statechart(context).top_level_nodes
    return plan


def gripper_steps(plan: Sequence) -> list[list[StatechartNode]]:
    """
    :return: The steps of `plan` after it parks the arms, grouped into announcement,
        gripper motion and wait.
    """
    nodes = plan.nodes[1:]
    return [nodes[position : position + 3] for position in range(0, len(nodes), 3)]


def actions_of(motion: StatechartNode) -> list[SetGripperAction]:
    """
    :return: The gripper actions a motion step runs, both of them for a step moving the
        two grippers together.
    """
    if isinstance(motion, Parallel):
        return list(motion.nodes)
    return [motion]


# %% the world


def test_the_simulated_world_holds_daisy(demonstration_world: World) -> None:
    assert len(demonstration_world.get_semantic_annotations_by_type(DAiSy)) == 1


def test_the_scene_needs_no_population(
    demonstration: DAiSyGripperDemonstration, demonstration_world: World
) -> None:
    assert demonstration.is_scene_populated(demonstration_world)


# %% where it runs


def test_every_execution_choice_runs_the_plan_with_its_own_executor() -> None:
    assert [choice.executor_type for choice in ExecutionChoice] == [
        KinematicallySimulatedPlanExecutor,
        RealRobotPlanExecutor,
        RealDAiSyPlanExecutor,
    ]


# %% the plan


def test_the_plan_parks_the_arms_first(plan: Sequence) -> None:
    assert isinstance(plan.nodes[0], ParkArmsAction)


def test_every_gripper_step_announces_its_motion_and_waits_after_it(
    plan: Sequence, demonstration: DAiSyGripperDemonstration
) -> None:
    steps = gripper_steps(plan)

    assert [
        (type(announcement), type(wait), wait.seconds)
        for announcement, _, wait in steps
    ] == [
        (
            Print,
            CountSimulationTimeSeconds,
            demonstration.pause_after_motion.total_seconds(),
        )
    ] * len(steps)


def test_each_gripper_moves_alone_then_both_together_presets_before_flex(
    plan: Sequence,
) -> None:
    commands = [
        (
            type(action.configuration.end_effector),
            action.configuration.joint_state.state_type,
            isinstance(motion, Parallel),
        )
        for _, motion, _ in gripper_steps(plan)
        for action in actions_of(motion)
    ]

    assert commands == [
        (DAiSyLeftGripper, GripperState.OPEN, False),
        (DAiSyLeftGripper, GripperState.CLOSE, False),
        (DAiSyRightGripper, GripperState.OPEN, False),
        (DAiSyRightGripper, GripperState.CLOSE, False),
        (DAiSyLeftGripper, GripperState.OPEN, True),
        (DAiSyRightGripper, GripperState.OPEN, True),
        (DAiSyLeftGripper, GripperState.CLOSE, True),
        (DAiSyRightGripper, GripperState.CLOSE, True),
        (DAiSyLeftGripper, GripperState.FLEXOPEN, False),
        (DAiSyLeftGripper, GripperState.FLEXCLOSE, False),
        (DAiSyRightGripper, GripperState.FLEXOPEN, False),
        (DAiSyRightGripper, GripperState.FLEXCLOSE, False),
        (DAiSyLeftGripper, GripperState.FLEXOPEN, True),
        (DAiSyRightGripper, GripperState.FLEXOPEN, True),
        (DAiSyLeftGripper, GripperState.FLEXCLOSE, True),
        (DAiSyRightGripper, GripperState.FLEXCLOSE, True),
    ]


def test_each_announcement_names_the_grippers_and_the_state_of_its_motion(
    plan: Sequence,
) -> None:
    announcements = [announcement.message for announcement, _, _ in gripper_steps(plan)]

    assert announcements == [
        "Opening the left gripper.",
        "Closing the left gripper.",
        "Opening the right gripper.",
        "Closing the right gripper.",
        "Opening both grippers.",
        "Closing both grippers.",
        "Flex-opening the left gripper.",
        "Flex-closing the left gripper.",
        "Flex-opening the right gripper.",
        "Flex-closing the right gripper.",
        "Flex-opening both grippers.",
        "Flex-closing both grippers.",
    ]


# %% running it


def test_the_plan_performs_in_the_simulation() -> None:
    DAiSyGripperDemonstration(used_robot=DAiSy, pause_after_motion=timedelta()).run()
