"""
Building and running coraplex plans in tests: the context extensions a test robot's
plans read, and the simulated executor running them.
"""

from __future__ import annotations

from dataclasses import replace


from coraplex.plans.context_extensions import RobotAccess, StatementGrounding
from coraplex.plans.executors import (
    PhysicallySimulatedPlanExecutor,
    PlanExecutor,
    RealRobotPlanExecutor,
    KinematicallySimulatedPlanExecutor,
)
from cramph.context import ContextExtension, StatechartContext
from cramph.executor import SimulationPacer
from cramph.node import StatechartNode
from cramph.statechart import Statechart
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World
from .sampling import SAMPLING_SEED

# %% building executors


def robot_extensions(robot: AbstractRobot) -> list[ContextExtension]:
    """
    :param robot: The robot performing the plans.
    :return: The context extensions a plan of `robot` reads, sampling with
        :data:`SAMPLING_SEED`.
    """
    return [RobotAccess(robot), StatementGrounding(sampling_seed=SAMPLING_SEED)]


def with_grounding(
    extensions: list[ContextExtension], **changes
) -> list[ContextExtension]:
    """
    :param extensions: The context extensions of a plan.
    :param changes: The fields of its
        :class:`~coraplex.plans.context_extensions.StatementGrounding` to change.
    :return: `extensions`, its grounding replaced by one with `changes` applied.
    """
    return [
        (
            replace(extension, **changes)
            if isinstance(extension, StatementGrounding)
            else extension
        )
        for extension in extensions
    ]


def simulated_executor(
    extensions: list[ContextExtension], **options
) -> KinematicallySimulatedPlanExecutor:
    """
    :param extensions: The context extensions of the plan, holding a
        :class:`~coraplex.plans.context_extensions.RobotAccess`.
    :param options: Further arguments of the executor.
    :return: An executor simulating plans in the world of the robot in `extensions`.
    """
    return KinematicallySimulatedPlanExecutor(
        StatechartContext.with_extensions(world_of(extensions), extensions), **options
    )


def robot_executor(
    extensions: list[ContextExtension], **options
) -> RealRobotPlanExecutor:
    """
    :param extensions: The context extensions of the plan, holding a
        :class:`~coraplex.plans.context_extensions.RobotAccess`.
    :param options: Further arguments of the executor.
    :return: An executor sending plans to the real robot in `extensions`.
    """
    return RealRobotPlanExecutor(
        StatechartContext.with_extensions(world_of(extensions), extensions), **options
    )


# %% reading context extensions


def robot_of(extensions: list[ContextExtension]) -> AbstractRobot:
    """
    :param extensions: The context extensions of a plan, holding a
        :class:`~coraplex.plans.context_extensions.RobotAccess`.
    :return: The robot in `extensions`.
    """
    return next(
        extension.robot
        for extension in extensions
        if isinstance(extension, RobotAccess)
    )


def world_of(extensions: list[ContextExtension]) -> World:
    """
    :param extensions: The context extensions of a plan, holding a
        :class:`~coraplex.plans.context_extensions.RobotAccess`.
    :return: The world of the robot in `extensions`.
    """
    return robot_of(extensions)._world


def context_of(extensions: list[ContextExtension]) -> StatechartContext:
    """
    :param extensions: The context extensions of a plan, holding a
        :class:`~coraplex.plans.context_extensions.RobotAccess`.
    :return: A context holding `extensions`, over the world of their robot, for building
        the parts of a plan that read the context before the plan joins a statechart.
    """
    return StatechartContext.with_extensions(world_of(extensions), extensions)


# %% running plans


def statechart_of(executor: PlanExecutor, *plan_nodes: StatechartNode) -> Statechart:
    """
    :param executor: The executor the statechart is built for.
    :param plan_nodes: The top-level nodes of the plan.
    :return: A statechart in the context of `executor`, holding `plan_nodes`.
    """
    statechart = Statechart(context=executor.context)
    for plan_node in plan_nodes:
        statechart.add_node(plan_node)
    return statechart


def expand(plan: StatechartNode, extensions: list[ContextExtension]) -> StatechartNode:
    """
    Expand `plan` in a statechart of its own, without running it, so a test can read the
    nodes its steps expand into.

    :param plan: The plan to expand.
    :param extensions: The context extensions the plan is expanded with.
    :return: The plan, expanded.
    """
    statechart_of(simulated_executor(extensions), plan)
    return plan


def run_plan(
    plan: StatechartNode, extensions: list[ContextExtension], **options
) -> KinematicallySimulatedPlanExecutor:
    """
    Run `plan` simulated until it succeeded.

    :param plan: The plan to run.
    :param extensions: The context extensions of the plan.
    :param options: Further arguments of the executor.
    :return: The executor that ran the plan.
    """
    executor = simulated_executor(extensions, **options)
    executor.compile(statechart_of(executor, plan))
    executor.execute()
    return executor


def perform_in_physics(
    plan: StatechartNode, extensions: list[ContextExtension], simulation: MujocoSim
) -> None:
    """
    :param plan: The plan to perform.
    :param extensions: The context extensions of the plan.
    :param simulation: The running physics to perform it in.
    """
    executor = PhysicallySimulatedPlanExecutor(
        StatechartContext.with_extensions(world_of(extensions), extensions),
        simulation=simulation,
    )
    executor.compile(statechart_of(executor, plan))
    executor.execute()
