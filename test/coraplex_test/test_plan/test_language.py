from datetime import timedelta
from functools import partial

import numpy as np
import pytest

from coraplex.datastructures.enums import (
    TaskStatus,
    DetectionTechnique,
)
from coraplex.execution_environment import simulated_robot
from coraplex.language import (
    CancelMonitor,
    SequentialNode,
    TryAllNode,
    ParallelNode,
    TryInOrderNode,
)
from coraplex.plans.factories import (
    sequential,
    parallel,
    try_in_order,
    try_all,
    cancel_when,
    repeat,
    code,
)
from coraplex.plans.failures import (
    AllChildrenFailed,
    PlanCancelled,
    PlanFailure,
    RepetitionsExhausted,
)
from coraplex.robot_plans import *
from coraplex.robot_plans.actions.core.misc import DetectAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from coraplex.robot_plans.motions.gripper import MoveToolCenterPointMotion
from giskardpy.motion_statechart.exceptions import NoConvergingTaskError
from giskardpy.motion_statechart.goals.templates import RepeatOnStall
from giskardpy.motion_statechart.graph_node import Task
from giskardpy.motion_statechart.nodes_for_testing.nodes_for_testing import (
    ConstFalseNode,
    ConstTrueNode,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.pr2 import PR2Joint
from semantic_digital_twin.spatial_types import Pose


def _torso_position(world):
    return world.state[
        world.get_degree_of_freedom_by_name("torso_lift_joint").id
    ].position


def test_factory_construction():
    act = NavigateAction(Pose())
    act2 = MoveTorsoAction(TorsoState.HIGH)
    act3 = DetectAction(DetectionTechnique.TYPES)

    root = sequential([act, act2, act3])
    assert isinstance(root, SequentialNode)
    assert len(root.children) == 3


def test_simplify_tree():
    act = NavigateAction(Pose())
    act2 = MoveTorsoAction(TorsoState.HIGH)
    act3 = DetectAction(DetectionTechnique.TYPES)
    act4 = DetectAction(DetectionTechnique.TYPES)

    root = sequential([act, sequential([act2, act3]), act4])
    root.plan.validate()
    assert [c.designator for c in root.children] == [act, act2, act3, act4]
    assert len(root.plan.nodes) == 5


def test_parallel_construction():
    act = NavigateAction(Pose())
    act2 = MoveTorsoAction(TorsoState.HIGH)
    act3 = DetectAction(DetectionTechnique.TYPES)

    root = parallel(
        [act, act2, act3],
    )
    root.plan.validate()
    assert isinstance(root, ParallelNode)
    assert len(root.children) == 3


def test_try_in_order_construction():
    act = NavigateAction(Pose())
    act2 = MoveTorsoAction(TorsoState.HIGH)
    act3 = DetectAction(DetectionTechnique.TYPES)

    root = try_in_order([act, act2, act3])
    root.plan.validate()
    assert isinstance(root, TryInOrderNode)
    assert len(root.children) == 3


def test_try_all_construction():
    act = NavigateAction(Pose())
    act2 = MoveTorsoAction(TorsoState.HIGH)
    act3 = DetectAction(DetectionTechnique.TYPES)

    root = try_all([act, act2, act3])
    root.plan.validate()
    assert isinstance(root, TryAllNode)
    assert len(root.children) == 3


def test_combination_construction():
    act = NavigateAction(Pose())
    act2 = MoveTorsoAction(TorsoState.HIGH)
    act3 = DetectAction(DetectionTechnique.TYPES)
    root = parallel([sequential([act, act2]), act3])
    assert isinstance(root, ParallelNode)
    assert len(root.children) == 2
    assert isinstance(root.children[0], SequentialNode)
    assert len(root.children[0].children) == 2


def test_repeat_construction():
    act = ParkArmsAction(Arms.BOTH)
    act2 = MoveTorsoAction(TorsoState.HIGH)

    root = repeat([act, act2], maximum_repetitions=10)
    assert len(root.children) == 2
    root.plan.validate()


def test_perform_execute_single(immutable_model_world):
    world, robot_view, context = immutable_model_world
    act = NavigateAction(Pose.from_xyz_rpy(0.3, -1.3, 0, reference_frame=world.root))
    act2 = MoveTorsoAction(TorsoState.HIGH)
    act3 = ParkArmsAction(Arms.BOTH)

    plan = sequential([act, act2, act3], context).plan
    with simulated_robot:
        plan.perform()
    np.testing.assert_almost_equal(
        robot_view.root.global_transform.to_np()[:3, 3], [0.3, -1.3, 0], decimal=1
    )
    assert world.state[
        world.get_degree_of_freedom_by_name(PR2Joint.TORSO_LIFT).id
    ].position == pytest.approx(0.3, abs=0.1)

    plan.validate()


def test_perform_single_designator(immutable_model_world):
    world, robot_view, context = immutable_model_world

    plan = sequential([MoveTorsoAction(TorsoState.HIGH)], context).plan
    with simulated_robot:
        plan.perform()

    assert world.state[
        world.get_degree_of_freedom_by_name(PR2Joint.TORSO_LIFT).id
    ].position == pytest.approx(0.3, abs=0.1)

    plan.validate()


def test_parallel_performs_all_its_children(immutable_model_world):
    """
    Every child of a parallel node ends up in the one motion state chart that is
    executed, so all of their targets are reached.
    """
    world, robot_view, context = immutable_model_world
    target = Pose.from_xyz_rpy(0.3, -1.3, 0, reference_frame=world.root)

    plan = parallel(
        [NavigateAction(target), MoveTorsoAction(TorsoState.HIGH)], context
    ).plan
    with simulated_robot:
        plan.perform()

    np.testing.assert_almost_equal(
        robot_view.root.global_transform.to_np()[:3, 3],
        target.to_np()[:3, 3],
        decimal=1,
    )
    [torso_up] = (
        robot_view.get_torso().get_joint_state_by_type(TorsoState.HIGH).target_values
    )
    assert _torso_position(world) == pytest.approx(torso_up, abs=0.05)
    assert plan.root.status == TaskStatus.SUCCEEDED
    plan.validate()


# %% expanding a plan does not execute it


@pytest.mark.parametrize("build_node", [sequential, try_in_order, try_all])
def test_a_language_node_expands_its_children_without_executing_them(
    immutable_model_world, build_node
):
    """
    ``notify`` expands the plan; what runs, and when, is decided afterwards by the goal
    the node becomes.

    A node that executes a child while expanding runs it before the actions in front of
    it, and again once the chart it was merged into is executed.
    """
    world, robot_view, context = immutable_model_world
    root = build_node([MoveTorsoAction(TorsoState.HIGH)], context)
    torso = world.get_degree_of_freedom_by_name(PR2Joint.TORSO_LIFT)
    position_before_expanding = world.state[torso.id].position

    with simulated_robot:
        root.notify()

    assert [child.status for child in root.children] == [TaskStatus.CREATED]
    assert world.state[torso.id].position == position_before_expanding


# %% running out of alternatives


@dataclass
class FailingMotion(BaseMotion):
    """
    A motion that never reaches its goal, standing in for any reason one fails.

    Lets a test say "this alternative does not work" without arranging the conditions
    that would make a real motion fail.
    """

    @property
    def _motion_chart(self) -> Task:
        return ConstFalseNode()


@pytest.mark.parametrize("build_node", [try_in_order, try_all])
def test_trying_alternatives_reports_that_every_one_of_them_failed(
    immutable_model_world, build_node
):
    """
    A plan that has run out of alternatives says so where it ran out, instead of leaving
    the chart running until it exhausts its control cycles.
    """
    world, robot_view, context = immutable_model_world
    plan = build_node([FailingMotion(), FailingMotion()], context).plan

    with simulated_robot, pytest.raises(AllChildrenFailed):
        plan.perform()


@pytest.mark.parametrize("build_node", [try_in_order, try_all])
def test_trying_alternatives_succeeds_once_one_of_them_works(
    immutable_model_world, build_node
):
    """
    An alternative that fails does not fail the plan: the one that works carries it, and
    it is that alternative's motion that is executed.
    """
    world, robot_view, context = immutable_model_world
    target = Pose.from_xyz_rpy(0.3, -1.3, 0, reference_frame=world.root)
    plan = build_node([FailingMotion(), NavigateAction(target)], context).plan

    with simulated_robot:
        plan.perform()

    np.testing.assert_almost_equal(
        robot_view.root.global_transform.to_np()[:3, 3],
        target.to_np()[:3, 3],
        decimal=1,
    )
    assert plan.root.status == TaskStatus.SUCCEEDED


def test_perform_repeat_runs_a_succeeding_motion_once(immutable_model_world):
    """
    Attempting stops as soon as the children succeed, so a motion that works first time
    is not repeated and the plan finishes normally.
    """
    world, robot_view, context = immutable_model_world

    plan = repeat(
        [MoveTorsoAction(TorsoState.HIGH)], maximum_repetitions=3, context=context
    ).plan
    with simulated_robot:
        plan.perform()

    assert world.state[
        world.get_degree_of_freedom_by_name("torso_lift_joint").id
    ].position == pytest.approx(0.3, abs=0.05)
    assert plan.root.status == TaskStatus.SUCCEEDED
    plan.validate()


def test_repeat_does_not_give_up_on_a_child_that_starts_at_its_goal(
    immutable_model_world,
):
    """
    A child that is already where it should be finishes without converging on anything,
    which must not be mistaken for an attempt that stalled.
    """
    world, robot_view, context = immutable_model_world
    with simulated_robot:
        sequential([MoveTorsoAction(TorsoState.HIGH)], context).plan.perform()

    plan = repeat(
        [MoveTorsoAction(TorsoState.HIGH), MoveTorsoAction(TorsoState.LOW)],
        maximum_repetitions=3,
        context=context,
    ).plan
    with simulated_robot:
        plan.perform()

    [torso_down] = (
        robot_view.get_torso().get_joint_state_by_type(TorsoState.LOW).target_values
    )
    assert _torso_position(world) == pytest.approx(torso_down, abs=0.05)
    assert plan.root.status == TaskStatus.SUCCEEDED


def test_exception_sequential(immutable_model_world):
    world, robot_view, context = immutable_model_world

    def raise_except():
        raise PlanFailure()

    act = NavigateAction(Pose.from_xyz_rpy(1, -1, reference_frame=world.root))
    act2 = code(raise_except)

    plan = sequential(
        [act, act2],
        context,
    ).plan

    def perform_plan():
        with simulated_robot:
            _ = plan.perform()

    with pytest.raises(PlanFailure):
        perform_plan()
    assert len(plan.root.children) == 2
    assert plan.root.status == TaskStatus.FAILED


# %% monitored subtrees


def test_cancel_monitor_construction():
    act = ParkArmsAction(Arms.BOTH)
    act2 = MoveTorsoAction(TorsoState.HIGH)

    root = cancel_when([act, act2], monitor=ConstFalseNode(name="never"))
    assert isinstance(root, CancelMonitor)
    assert len(root.children) == 2
    root.plan.validate()


def test_cancel_monitor_stops_the_motion_it_wraps(immutable_model_world):
    """
    A monitor that is true from the start stops the motion before it moves.
    """
    world, robot_view, context = immutable_model_world
    start_position = _torso_position(world)

    plan = cancel_when(
        [MoveTorsoAction(TorsoState.HIGH)],
        monitor=ConstTrueNode(name="always"),
        context=context,
    ).plan
    with pytest.raises(PlanCancelled):
        with simulated_robot:
            plan.perform()

    assert _torso_position(world) == pytest.approx(start_position, abs=0.05)


def test_cancel_monitor_gives_up_on_the_plan_instead_of_stalling(immutable_model_world):
    """
    Cancelling reports that the plan has to be made again, rather than leaving the
    surrounding plan waiting for a subtree that will never succeed until the motion runs
    out of control cycles.
    """
    world, robot_view, context = immutable_model_world

    plan = sequential(
        [
            cancel_when(
                [MoveTorsoAction(TorsoState.HIGH)],
                monitor=ConstTrueNode(name="always"),
            ),
            MoveTorsoAction(TorsoState.LOW),
        ],
        context=context,
    ).plan
    with pytest.raises(PlanCancelled):
        with simulated_robot:
            plan.perform()


def test_never_firing_cancel_monitor_leaves_the_motion_alone(immutable_model_world):
    """
    The control for the test above: the same plan with a monitor that never fires runs
    the motion to its target.
    """
    world, robot_view, context = immutable_model_world

    plan = cancel_when(
        [MoveTorsoAction(TorsoState.HIGH)],
        monitor=ConstFalseNode(name="never"),
        context=context,
    ).plan
    with simulated_robot:
        plan.perform()

    assert _torso_position(world) == pytest.approx(0.3, abs=0.05)
    assert plan.root.status == TaskStatus.SUCCEEDED


def test_repeat_raises_when_it_runs_out_of_attempts(immutable_model_world):
    """
    A motion that can never succeed is attempted the allowed number of times and then
    reported as a plan failure, rather than silently stalling until the motion runs out
    of control cycles.
    """
    world, robot_view, context = immutable_model_world
    unreachable = Pose.from_xyz_rpy(5, 0, 0, reference_frame=world.root)

    plan = repeat(
        [MoveToolCenterPointMotion(target=unreachable, arm=Arms.RIGHT)],
        maximum_repetitions=2,
        context=context,
        repeat_template=partial(RepeatOnStall, timeout=timedelta(seconds=1)),
    ).plan

    with pytest.raises(RepetitionsExhausted):
        with simulated_robot:
            plan.perform()


def test_repeat_of_a_non_converging_motion_is_rejected(immutable_model_world):
    """
    Retrying on a stall needs a task whose progress can be measured, so a repeat whose
    children never converge is reported when the motion state chart is compiled.
    """
    world, robot_view, context = immutable_model_world

    plan = repeat(
        [NavigateAction(Pose.from_xyz_rpy(1, -1, reference_frame=world.root))],
        maximum_repetitions=2,
        context=context,
    ).plan

    with pytest.raises(NoConvergingTaskError):
        with simulated_robot:
            plan.perform()
