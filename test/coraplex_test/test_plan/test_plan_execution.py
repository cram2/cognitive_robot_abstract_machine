"""
Tests for compiling and executing a plan with a
:class:`~coraplex.plans.executors.PlanExecutor`.
"""

from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta

import pytest

from cramph.exceptions import EmptyStatechartError

from coraplex.exceptions import (
    MotionDidNotFinish,
    PlanNotCompiled,
)
from coraplex.plans.executors import KinematicallySimulatedPlanExecutor
from coraplex.plans.plan_transformation import PlanRewriting
from coraplex.plans.failures import (
    MotionExceededSimulationTimeLimit,
    MotionMadeNoProgress,
    MotionViolatedCollisionAvoidance,
)
from coraplex.robot_plans.actions.core.pick_up import ReachAction
from coraplex.robot_plans.plan_transformations import DetectBeforeGrasp
from coraplex.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    ParkArmsAction,
)
from cramph.composites import Sequence
from cramph.exceptions import (
    DuplicateContextExtensionError,
    StatechartOfDifferentContextError,
)
from cramph.data_types import LifeCycleValues, ObservationStateValues
from cramph.executor import Pacer, StatechartExecutor
from cramph.node import CancelStatechart
from cramph.nodes_for_testing import NodeFailingOnObservingFalse
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.exceptions import CollisionViolatedError
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
    SelfCollisionAvoidance,
)
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.progress_monitors import StillProgressing
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.robots.pr2 import PR2Joint
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.world import World
from ..conftest import physically_simulated_executor
from ...plan_running import (
    robot_extensions,
    run_plan,
    simulated_executor,
    statechart_of,
)

# %% counting the waits between ticks


@dataclass
class WaitCountingPacer(Pacer):
    """
    Counts how often a loop waited for its next cycle, without waiting.
    """

    wait_count: int = 0
    """
    How often the loop waited so far.
    """

    def sleep(self) -> None:
        self.wait_count += 1


# %% compiling


def test_compiling_puts_the_plan_and_its_end_in_one_statechart(
    pr2_apartment_context,
) -> None:
    world, robot_view, extensions = pr2_apartment_context
    plan = MoveTorsoAction(torso_state=TorsoState.HIGH)
    executor = simulated_executor(extensions)

    executor.compile(statechart_of(executor, plan))

    statechart = plan.statechart
    assert plan in statechart.top_level_nodes
    assert len(statechart.get_nodes_by_type(EndMotion)) == 1


def test_executing_before_compiling_is_refused(pr2_apartment_context) -> None:
    world, robot_view, extensions = pr2_apartment_context

    with pytest.raises(PlanNotCompiled):
        simulated_executor(extensions).execute()


# %% executing


def test_executing_runs_the_plan_until_it_succeeded(pr2_apartment_context) -> None:
    world, robot_view, extensions = pr2_apartment_context
    plan = MoveTorsoAction(torso_state=TorsoState.HIGH)
    executor = simulated_executor(extensions)

    executor.compile(statechart_of(executor, plan))
    executor.execute()

    assert plan.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_a_plan_that_fails_raises_motion_did_not_finish(pr2_apartment_context) -> None:
    world, robot_view, extensions = pr2_apartment_context
    plan = NodeFailingOnObservingFalse(observation=ObservationStateValues.FALSE)
    executor = simulated_executor(extensions)

    with pytest.raises(MotionDidNotFinish):
        executor.compile(statechart_of(executor, plan))
        executor.execute()


def test_preparing_a_plan_does_not_move_the_robot(pr2_apartment_context) -> None:
    world, robot_view, extensions = pr2_apartment_context
    torso = world.get_degree_of_freedom_by_name(PR2Joint.TORSO_LIFT).id
    position_before = world.state[torso].position
    executor = simulated_executor(extensions)

    executor.prepare(
        statechart_of(executor, MoveTorsoAction(torso_state=TorsoState.HIGH))
    )

    assert world.state[torso].position == position_before


def test_a_plan_of_several_top_level_nodes_ends_once_all_of_them_succeeded(
    pr2_apartment_context,
) -> None:
    """
    A statechart has no root node, so every node the caller puts at its top level is
    part of the plan.
    """
    world, robot_view, extensions = pr2_apartment_context
    torso = MoveTorsoAction(torso_state=TorsoState.HIGH)
    arms = ParkArmsAction(arms=robot_view.all_arms)
    executor = simulated_executor(extensions)

    executor.compile(statechart_of(executor, torso, arms))
    executor.execute()

    assert executor.plan_nodes == [torso, arms]
    assert torso.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert arms.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_a_statechart_without_a_plan_is_refused(pr2_apartment_context) -> None:
    world, robot_view, extensions = pr2_apartment_context
    executor = simulated_executor(extensions)

    with pytest.raises(EmptyStatechartError):
        executor.compile(statechart_of(executor))


def test_a_statechart_of_another_context_is_refused(pr2_apartment_context) -> None:
    world, robot_view, extensions = pr2_apartment_context
    statechart = statechart_of(
        simulated_executor(extensions), MoveTorsoAction(torso_state=TorsoState.HIGH)
    )

    with pytest.raises(StatechartOfDifferentContextError):
        simulated_executor(extensions).compile(statechart)


# %% how long a plan may take


def test_compiling_watches_the_plan_for_progress(pr2_apartment_context) -> None:
    """
    A stalled run has to end by itself, so the statechart carries a monitor watching the
    plan and an abort path wired to it.
    """
    world, robot_view, extensions = pr2_apartment_context
    plan = MoveTorsoAction(torso_state=TorsoState.HIGH)
    executor = simulated_executor(extensions)

    executor.compile(statechart_of(executor, plan))

    [progress_monitor] = plan.statechart.get_nodes_by_type(StillProgressing)
    assert progress_monitor.monitored_node is plan
    assert len(plan.statechart.get_nodes_by_type(CancelStatechart)) == 1


def test_a_plan_that_stops_approaching_its_goal_is_given_up_on(
    pr2_apartment_context,
) -> None:
    """
    Nothing bounds the tick loop but the monitor, so a reach the arm cannot close on has
    to end the run rather than tick forever.
    """
    world, robot_view, extensions = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    milk.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        2, 1.5, 50, reference_frame=milk.root.parent_connection.parent
    )
    plan = ReachAction(
        grasp=GraspCandidate.from_body_origin(milk), arm=robot_view.right_arm
    )
    executor = simulated_executor(extensions)

    with pytest.raises(MotionMadeNoProgress):
        executor.compile(statechart_of(executor, plan))
        executor.execute()


def test_a_plan_that_outlasts_the_simulation_time_limit_is_given_up_on(
    pr2_apartment_context,
) -> None:
    """
    A simulated plan is given up on once it exceeds the simulation time limit.
    """
    world, robot_view, extensions = pr2_apartment_context
    executor = simulated_executor(extensions, simulation_time_per_action=timedelta(0))

    with pytest.raises(MotionExceededSimulationTimeLimit):
        executor.compile(
            statechart_of(executor, MoveTorsoAction(torso_state=TorsoState.HIGH))
        )
        executor.execute()


def test_a_plan_is_given_up_on_at_the_simulation_time_limit_of_its_own_executor(
    pr2_apartment_context,
) -> None:
    world, robot_view, extensions = pr2_apartment_context
    executor = simulated_executor(
        extensions, simulation_time_per_action=timedelta(milliseconds=100)
    )

    with pytest.raises(MotionExceededSimulationTimeLimit) as error:
        executor.compile(
            statechart_of(executor, MoveTorsoAction(torso_state=TorsoState.HIGH))
        )
        executor.execute()

    assert error.value.time_limit == executor.simulation_time_limit


def test_a_plan_may_take_longer_the_more_actions_it_holds(
    pr2_apartment_context,
) -> None:
    """
    Each torso stroke fits the time one action is given, both strokes together only the
    time two actions are given.
    """
    world, robot_view, extensions = pr2_apartment_context
    plan = Sequence(
        nodes=[
            MoveTorsoAction(torso_state=TorsoState.HIGH),
            MoveTorsoAction(torso_state=TorsoState.LOW),
        ]
    )

    run_plan(plan, extensions, simulation_time_per_action=timedelta(seconds=25))

    assert plan.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_the_simulation_time_limit_grows_when_an_action_joins(
    pr2_apartment_context,
) -> None:
    """
    An action joining the running plan adds its time before it starts.
    """
    world, robot_view, extensions = pr2_apartment_context
    executor = simulated_executor(extensions)
    statechart = statechart_of(executor, MoveTorsoAction(torso_state=TorsoState.HIGH))
    executor.compile(statechart)

    with statechart.modify():
        statechart.add_node(MoveTorsoAction(torso_state=TorsoState.LOW))

    assert executor.simulation_time_limit == 2 * executor.simulation_time_per_action


def test_a_simulated_plan_waits_for_its_pacer_after_every_tick(
    pr2_apartment_context,
) -> None:
    world, robot_view, extensions = pr2_apartment_context
    pacer = WaitCountingPacer()
    executor = simulated_executor(extensions, pacer=pacer)

    executor.compile(
        statechart_of(executor, MoveTorsoAction(torso_state=TorsoState.HIGH))
    )
    executor.execute()

    assert pacer.wait_count == executor.tick_count


def test_a_plan_that_violates_collision_avoidance_fails_as_a_plan_failure(
    pr2_apartment_context, monkeypatch
) -> None:
    """
    A plan that brings the robot closer to something than collision avoidance allows did
    not work from where it started, so a surrounding plan can try another candidate
    instead of stopping.
    """
    world, robot_view, extensions = pr2_apartment_context
    violation = CollisionViolatedError(violated_collisions=[])

    def violate_collision_avoidance(executor: StatechartExecutor) -> None:
        raise violation

    executor = simulated_executor(extensions)
    executor.compile(
        statechart_of(executor, MoveTorsoAction(torso_state=TorsoState.HIGH))
    )
    monkeypatch.setattr(StatechartExecutor, "tick", violate_collision_avoidance)
    with pytest.raises(MotionViolatedCollisionAvoidance) as failure:
        executor.execute()

    assert failure.value.violation is violation


# %% avoiding collisions in a world with several robots


@pytest.fixture
def tiago_world_with_a_stretch(_tiago_world_setup) -> World:
    """
    A world holding a Tiago and, three metres in front of it, a Stretch.
    """
    world = deepcopy(_tiago_world_setup)
    RobotSpecification(
        Stretch, world_T_odom=HomogeneousTransformationMatrix.from_xyz_rpy(x=3)
    ).spawn(world)
    return world


def test_collision_avoidance_is_compiled_for_the_robot_performing_the_plan(
    tiago_world_with_a_stretch,
) -> None:
    [tiago] = tiago_world_with_a_stretch.get_semantic_annotations_by_type(Tiago)
    plan = MoveTorsoAction(torso_state=TorsoState.HIGH)
    executor = simulated_executor(robot_extensions(tiago), collision_avoidance=True)

    executor.compile(statechart_of(executor, plan))

    statechart = plan.statechart
    [external_collision_avoidance] = statechart.get_nodes_by_type(
        ExternalCollisionAvoidance
    )
    [self_collision_avoidance] = statechart.get_nodes_by_type(SelfCollisionAvoidance)
    assert external_collision_avoidance.robot is tiago
    assert self_collision_avoidance.robot is tiago


def test_a_plan_avoiding_collisions_succeeds_in_a_world_with_another_robot(
    tiago_world_with_a_stretch,
) -> None:
    [tiago] = tiago_world_with_a_stretch.get_semantic_annotations_by_type(Tiago)
    plan = MoveTorsoAction(torso_state=TorsoState.HIGH)
    executor = simulated_executor(robot_extensions(tiago), collision_avoidance=True)

    executor.compile(statechart_of(executor, plan))
    executor.execute()

    assert plan.life_cycle_state == LifeCycleValues.SUCCEEDED


# %% performing a plan in a physically simulated world


def test_a_physically_simulated_plan_steps_the_physics_one_control_period_per_tick(
    pr2_apartment_context,
) -> None:
    """
    The servos take every command as their set point, so the physics has to advance by
    exactly the time the controller planned the command for before the next tick reads
    the world back.
    """
    world, robot_view, extensions = pr2_apartment_context
    executor = physically_simulated_executor(extensions)

    executor.compile(
        statechart_of(executor, MoveTorsoAction(torso_state=TorsoState.HIGH))
    )
    executor.execute()

    control_time_step = executor.require_extension(
        MotionControl
    ).qp_controller_config.control_time_step
    assert executor.pacer.simulation.steps == [control_time_step] * executor.tick_count


def test_a_physically_simulated_plan_is_given_the_same_time_limit(
    pr2_apartment_context,
) -> None:
    """
    Both simulated executors tick the same loop, so a plan performed in physics is given
    up on by the same limit as one projected kinematically.
    """
    world, robot_view, extensions = pr2_apartment_context
    executor = physically_simulated_executor(
        extensions, simulation_time_per_action=timedelta(0)
    )

    with pytest.raises(MotionExceededSimulationTimeLimit):
        executor.compile(
            statechart_of(executor, MoveTorsoAction(torso_state=TorsoState.HIGH))
        )
        executor.execute()


# %% adapting a plan to its executor


def test_a_plan_is_adapted_after_the_transformations_it_was_given(
    pr2_apartment_context,
) -> None:
    """
    The executor's robot adaptations and then its plan adaptations adapt an action
    before the plan's transformations are offered it.
    """
    world, robot_view, extensions = pr2_apartment_context
    detecting = DetectBeforeGrasp()
    executor = physically_simulated_executor(
        [*extensions, PlanRewriting(transformations=[detecting])]
    )

    rewriting = executor.context.require_extension(PlanRewriting)

    assert rewriting.transformations == [
        *executor.robot_adaptations,
        *executor.plan_adaptations,
        detecting,
    ]


def test_a_plan_given_no_transformations_is_still_adapted(
    pr2_apartment_context,
) -> None:
    world, robot_view, extensions = pr2_apartment_context
    executor = physically_simulated_executor(extensions)

    rewriting = executor.context.require_extension(PlanRewriting)

    assert rewriting.transformations == [
        *executor.robot_adaptations,
        *executor.plan_adaptations,
    ]


def test_the_rewriting_a_plan_was_given_is_left_as_it_was(
    pr2_apartment_context,
) -> None:
    """
    The same context extensions may configure executors of different kinds, so one
    executor's adaptations must not reach another through them.
    """
    world, robot_view, extensions = pr2_apartment_context
    detecting = DetectBeforeGrasp()
    given = PlanRewriting(transformations=[detecting])

    physically_simulated_executor([*extensions, given])

    assert given.transformations == [detecting]


def test_a_second_executor_on_a_context_leaves_the_rewriting_of_the_first_alone(
    pr2_apartment_context,
) -> None:
    world, robot_view, extensions = pr2_apartment_context
    first = simulated_executor(extensions)
    rewriting = first.context.require_extension(PlanRewriting)
    transformations = list(rewriting.transformations)

    with pytest.raises(DuplicateContextExtensionError):
        KinematicallySimulatedPlanExecutor(
            first.context, plan_adaptations=[DetectBeforeGrasp()]
        )

    assert first.context.require_extension(PlanRewriting) is rewriting
    assert rewriting.transformations == transformations
