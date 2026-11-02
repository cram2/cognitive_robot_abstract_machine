import threading

import numpy as np
import pytest
from typing_extensions import Never

from cramph.data_types import LifeCycleValues, ObservationStateValues

from coraplex.datastructures.enums import DetectionTechnique

from coraplex.plans.failures import PlanFailure
from cramph.exceptions import PlanCancelled, RepetitionsExhausted
from cramph.composites import (
    CancelledWhenTrue,
    Parallel,
    RepeatUntil,
    Sequence,
    TryAll,
    TryInOrder,
)
from coraplex.robot_plans.actions.core.misc import DetectAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from cramph.monitors import CountNodeResets
from cramph.nodes_for_testing import ConstFalseNode, ConstTrueNode
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.spatial_types import Pose
from semantic_digital_twin.robots.pr2 import PR2Joint
from cramph.threaded_nodes import FunctionCall
from ...plan_running import run_plan


def test_factory_construction():
    act = NavigateAction(target_location=Pose())
    act2 = MoveTorsoAction(torso_state=TorsoState.HIGH)
    act3 = DetectAction(technique=DetectionTechnique.TYPES)

    root = Sequence(nodes=[act, act2, act3])
    assert isinstance(root, Sequence)
    assert len(root.children) == 3


def test_parallel_construction():
    act = NavigateAction(target_location=Pose())
    act2 = MoveTorsoAction(torso_state=TorsoState.HIGH)
    act3 = DetectAction(technique=DetectionTechnique.TYPES)

    root = Parallel(nodes=[act, act2, act3])
    assert isinstance(root, Parallel)
    assert len(root.children) == 3


def test_try_in_order_construction():
    act = NavigateAction(target_location=Pose())
    act2 = MoveTorsoAction(torso_state=TorsoState.HIGH)
    act3 = DetectAction(technique=DetectionTechnique.TYPES)

    root = TryInOrder(nodes=[act, act2, act3])
    assert isinstance(root, TryInOrder)
    assert len(root.children) == 3


def test_try_all_construction():
    act = NavigateAction(target_location=Pose())
    act2 = MoveTorsoAction(torso_state=TorsoState.HIGH)
    act3 = DetectAction(technique=DetectionTechnique.TYPES)

    root = TryAll(nodes=[act, act2, act3])
    assert isinstance(root, TryAll)
    assert len(root.children) == 3


def test_combination_construction():
    act = NavigateAction(target_location=Pose())
    act2 = MoveTorsoAction(torso_state=TorsoState.HIGH)
    act3 = DetectAction(technique=DetectionTechnique.TYPES)
    root = Parallel(nodes=[Sequence(nodes=[act, act2]), act3])
    assert isinstance(root, Parallel)
    assert len(root.children) == 2
    assert isinstance(root.children[0], Sequence)
    assert len(root.children[0].children) == 2


def test_perform_execute_single(pr2_apartment_context):
    world, robot_view, extensions = pr2_apartment_context
    act = NavigateAction(
        target_location=Pose.from_xyz_rpy(0.3, -1.3, 0, reference_frame=world.root)
    )
    act2 = MoveTorsoAction(torso_state=TorsoState.HIGH)
    act3 = ParkArmsAction(arms=robot_view.all_arms)

    plan = Sequence(nodes=[act, act2, act3])
    run_plan(plan, extensions)
    np.testing.assert_almost_equal(
        robot_view.root.global_transform.to_np()[:3, 3], [0.3, -1.3, 0], decimal=1
    )
    assert world.state[
        world.get_degree_of_freedom_by_name(PR2Joint.TORSO_LIFT).id
    ].position == pytest.approx(0.3, abs=0.1)


def test_perform_single_designator(pr2_apartment_context):
    world, robot_view, extensions = pr2_apartment_context

    plan = Sequence(nodes=[MoveTorsoAction(torso_state=TorsoState.HIGH)])
    run_plan(plan, extensions)

    assert world.state[
        world.get_degree_of_freedom_by_name(PR2Joint.TORSO_LIFT).id
    ].position == pytest.approx(0.3, abs=0.1)


def test_perform_parallel(pr2_apartment_context):
    world, robot_view, extensions = pr2_apartment_context

    def check_thread_id(main_id):
        assert main_id != threading.get_ident()

    main_thread_id = threading.get_ident()
    act = FunctionCall(function=lambda: check_thread_id(main_thread_id))
    act2 = FunctionCall(function=lambda: check_thread_id(main_thread_id))
    act3 = FunctionCall(function=lambda: check_thread_id(main_thread_id))

    plan = Parallel(nodes=[act, act2, act3])
    run_plan(plan, extensions)

    assert [node.life_cycle_state for node in plan.children] == [
        LifeCycleValues.SUCCEEDED
    ] * 3


def _repeat(task: Sequence, maximum_repetitions: int) -> RepeatUntil:
    """
    :return: `task` attempted until it succeeds, at most `maximum_repetitions` times,
        raising :class:`RepetitionsExhausted` once the attempts run out.
    """
    return RepeatUntil(
        task=task,
        stop_retry_monitor=CountNodeResets(node=task, target=maximum_repetitions),
        exception=RepetitionsExhausted(
            repeated_node=task, maximum_repetitions=maximum_repetitions
        ),
    )


def test_perform_repeat_runs_a_succeeding_motion_once(pr2_apartment_context):
    """
    Attempting stops as soon as the children succeed, so a motion that works first time
    is not repeated and the plan finishes normally.
    """
    world, robot_view, extensions = pr2_apartment_context

    plan = _repeat(Sequence(nodes=[MoveTorsoAction(torso_state=TorsoState.HIGH)]), 3)
    run_plan(plan, extensions)

    assert world.state[
        world.get_degree_of_freedom_by_name("torso_lift_joint").id
    ].position == pytest.approx(0.3, abs=0.05)
    assert plan.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_exception_sequential(pr2_apartment_context):
    world, robot_view, extensions = pr2_apartment_context

    def raise_except():
        raise PlanFailure()

    act = NavigateAction(
        target_location=Pose.from_xyz_rpy(1, -1, reference_frame=world.root)
    )
    act2 = FunctionCall(function=raise_except, failure_types=(PlanFailure,))

    plan = Sequence(nodes=[act, act2])

    def perform_plan():
        run_plan(plan, extensions)

    with pytest.raises(PlanFailure):
        perform_plan()
    assert len(plan.children) == 2
    assert plan.life_cycle_state == LifeCycleValues.FAILED


def test_exception_try_in_order(pr2_apartment_context):
    world, robot_view, extensions = pr2_apartment_context

    def raise_except():
        raise PlanFailure()

    act = NavigateAction(
        target_location=Pose.from_xyz_rpy(1, -1, reference_frame=world.root)
    )
    act2 = FunctionCall(function=raise_except, failure_types=(PlanFailure,))

    plan = TryInOrder(nodes=[act, act2])
    run_plan(plan, extensions)
    assert len(plan.children) == 2
    assert plan.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_exception_try_all(pr2_apartment_context):
    world, robot_view, extensions = pr2_apartment_context

    def raise_except() -> Never:
        raise PlanFailure()

    act = NavigateAction(
        target_location=Pose.from_xyz_rpy(x=-2, reference_frame=world.root)
    )
    act2 = FunctionCall(function=raise_except, failure_types=(PlanFailure,))

    plan = TryAll(nodes=[act, act2])
    run_plan(plan, extensions)

    assert type(plan) is TryAll
    assert plan.life_cycle_state == LifeCycleValues.SUCCEEDED


# %% children run only as part of the chart


def test_try_in_order_recovers_from_a_failing_code_step(pr2_apartment_context) -> None:
    """
    A code step that fails is one failed alternative, so the next one is tried.
    """
    world, robot_view, extensions = pr2_apartment_context

    def raise_except():
        raise PlanFailure()

    plan = TryInOrder(
        nodes=[
            FunctionCall(function=raise_except, failure_types=(PlanFailure,)),
            MoveTorsoAction(torso_state=TorsoState.HIGH),
        ]
    )
    run_plan(plan, extensions)

    [torso_up] = (
        robot_view.get_torso().get_joint_state_by_type(TorsoState.HIGH).target_values
    )
    assert _torso_position(world) == pytest.approx(torso_up, abs=0.05)
    assert plan.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_children_report_the_outcome_of_the_chart_they_ran_in(
    pr2_apartment_context,
) -> None:
    """
    A child is only run as part of its parent's chart, and reports how it ended there.
    """
    world, robot_view, extensions = pr2_apartment_context

    root = Sequence(
        nodes=[
            MoveTorsoAction(torso_state=TorsoState.HIGH),
            ParkArmsAction(arms=robot_view.all_arms),
        ]
    )
    run_plan(root, extensions)

    assert [child.life_cycle_state for child in root.children] == [
        LifeCycleValues.SUCCEEDED,
        LifeCycleValues.SUCCEEDED,
    ]


# %% monitored subtrees


def test_cancel_monitor_construction(pr2_apartment_context):
    world, robot_view, extensions = pr2_apartment_context
    act = ParkArmsAction(arms=robot_view.all_arms)
    act2 = MoveTorsoAction(torso_state=TorsoState.HIGH)

    never = ConstFalseNode(name="never")

    root = CancelledWhenTrue(
        monitor=never,
        monitored_node=Sequence(nodes=[act, act2]),
        exception=PlanCancelled(monitor=never),
    )
    assert isinstance(root, CancelledWhenTrue)
    assert root.monitored_node.nodes == [act, act2]


def _torso_position(world):
    return world.state[
        world.get_degree_of_freedom_by_name("torso_lift_joint").id
    ].position


def test_cancel_monitor_stops_the_motion_it_wraps(pr2_apartment_context):
    """
    A monitor that is true from the start stops the motion before it moves.
    """
    world, robot_view, extensions = pr2_apartment_context
    start_position = _torso_position(world)

    always = ConstTrueNode(name="always")

    plan = CancelledWhenTrue(
        monitor=always,
        monitored_node=Sequence(nodes=[MoveTorsoAction(torso_state=TorsoState.HIGH)]),
        exception=PlanCancelled(monitor=always),
    )
    with pytest.raises(PlanCancelled):
        run_plan(plan, extensions)

    assert _torso_position(world) == pytest.approx(start_position, abs=0.05)


def test_cancel_monitor_gives_up_on_the_plan_instead_of_stalling(pr2_apartment_context):
    """
    Cancelling reports that the plan has to be made again, rather than leaving the
    surrounding plan waiting for a subtree that will never succeed until the motion runs
    out of control cycles.
    """
    world, robot_view, extensions = pr2_apartment_context

    always = ConstTrueNode(name="always")

    plan = Sequence(
        nodes=[
            CancelledWhenTrue(
                monitor=always,
                monitored_node=Sequence(
                    nodes=[MoveTorsoAction(torso_state=TorsoState.HIGH)]
                ),
                exception=PlanCancelled(monitor=always),
            ),
            MoveTorsoAction(torso_state=TorsoState.LOW),
        ]
    )
    with pytest.raises(PlanCancelled):
        run_plan(plan, extensions)


def test_never_firing_cancel_monitor_leaves_the_motion_alone(pr2_apartment_context):
    """
    The control for the test above: the same plan with a monitor that never fires runs
    the motion to its target.
    """
    world, robot_view, extensions = pr2_apartment_context

    never = ConstFalseNode(name="never")

    plan = CancelledWhenTrue(
        monitor=never,
        monitored_node=Sequence(nodes=[MoveTorsoAction(torso_state=TorsoState.HIGH)]),
        exception=PlanCancelled(monitor=never),
    )
    run_plan(plan, extensions)

    assert _torso_position(world) == pytest.approx(0.3, abs=0.05)
    assert plan.last_observation_state == ObservationStateValues.TRUE
