import threading
import time
import pytest

from pycram.datastructures.enums import TaskStatus, MonitorBehavior
from pycram.failure_handling import RetryMonitor
from pycram.failures import PlanFailure, NotALanguageExpression
from pycram.fluent import Fluent
from pycram.language import (
    ParallelPlan,
    TryAllPLan,
    MonitorPlan,
    MonitorNode,
    SequentialNode,
    RepeatPlan,
    CodePlan,
    TryAllNode,
)
from pycram.process_module import simulated_robot
from pycram.robot_plans import *


def test_simplify_tree(immutable_model_world):
    world, robot_view, context = immutable_model_world
    act = NavigateActionDescription(PoseStamped())
    act2 = MoveTorsoActionDescription(TorsoState.HIGH)
    act3 = DetectActionDescription(DetectionTechnique.TYPES)

    plan = SequentialPlan(context, act, SequentialPlan(context, act2, act3))
    assert len(plan.root.children) == 3
    assert plan.root.children[0].children == []

    assert len(plan.nodes) == len(plan.all_nodes)
    assert len(plan.edges) == len(plan.all_nodes) - 1


def test_sequential_construction(immutable_model_world):
    world, robot_view, context = immutable_model_world
    act = NavigateActionDescription(PoseStamped())
    act2 = MoveTorsoActionDescription(TorsoState.HIGH)
    act3 = DetectActionDescription(DetectionTechnique.TYPES)

    plan = SequentialPlan(context, act, act2, act3)
    assert isinstance(plan, SequentialPlan)
    assert len(plan.root.children) == 3


def test_parallel_construction(immutable_model_world):
    world, robot_view, context = immutable_model_world
    act = NavigateActionDescription(PoseStamped())
    act2 = MoveTorsoActionDescription(TorsoState.HIGH)
    act3 = DetectActionDescription(DetectionTechnique.TYPES)

    plan = ParallelPlan(context, act, act2, act3)
    assert isinstance(plan, ParallelPlan)
    assert len(plan.root.children) == 3


def test_try_in_order_construction(immutable_model_world):
    world, robot_view, context = immutable_model_world
    act = NavigateActionDescription(PoseStamped())
    act2 = MoveTorsoActionDescription(TorsoState.HIGH)
    act3 = DetectActionDescription(DetectionTechnique.TYPES)

    plan = TryInOrderPlan(context, act, act2, act3)
    assert isinstance(plan, TryInOrderPlan)
    assert len(plan.root.children) == 3

    assert len(plan.nodes) == len(plan.all_nodes)
    assert len(plan.edges) == len(plan.all_nodes) - 1


def test_try_all_construction(immutable_model_world):
    world, robot_view, context = immutable_model_world
    act = NavigateActionDescription(PoseStamped())
    act2 = MoveTorsoActionDescription(TorsoState.HIGH)
    act3 = DetectActionDescription(DetectionTechnique.TYPES)

    plan = TryAllPLan(context, act, act2, act3)
    assert TryAllNode
    assert isinstance(plan, TryAllPLan)
    assert len(plan.root.children) == 3


def test_combination_construction(immutable_model_world):
    world, robot_view, context = immutable_model_world
    act = NavigateActionDescription(PoseStamped())
    act2 = MoveTorsoActionDescription(TorsoState.HIGH)
    act3 = DetectActionDescription(DetectionTechnique.TYPES)

    plan = ParallelPlan(context, SequentialPlan(context, act, act2), act3)
    assert isinstance(plan, ParallelPlan)
    assert len(plan.root.children) == 2
    assert isinstance(plan.root.children[0], SequentialNode)


def test_pickup_par_construction(immutable_model_world):
    world, robot_view, context = immutable_model_world
    act = NavigateActionDescription(PoseStamped())
    act2 = PickUpActionDescription(BelieveObject(names=["milk"]), ["left"], ["front"])

    with pytest.raises(AttributeError):
        ParallelPlan(context, act, act2)


def test_pickup_try_all_construction(immutable_model_world):
    world, robot_view, context = immutable_model_world
    act = NavigateActionDescription(PoseStamped())
    act2 = PickUpActionDescription(BelieveObject(names=["milk"]), ["left"], ["front"])

    with pytest.raises(AttributeError):
        TryAllPLan(context, act, act2)


def test_monitor_construction(immutable_model_world):
    world, robot_view, context = immutable_model_world
    act = ParkArmsActionDescription(Arms.BOTH)
    act2 = MoveTorsoActionDescription(TorsoState.HIGH)

    def monitor_func():
        time.sleep(1)
        return True

    plan = MonitorPlan(monitor_func, context, SequentialPlan(context, act, act2))
    assert len(plan.root.children) == 1
    assert isinstance(plan.root, MonitorNode)

    assert len(plan.nodes) == len(plan.all_nodes)
    assert len(plan.edges) == len(plan.all_nodes) - 1


def test_retry_monitor_construction(immutable_model_world):
    world, robot_view, context = immutable_model_world
    act = ParkArmsActionDescription(Arms.BOTH)
    act2 = MoveTorsoActionDescription(TorsoState.HIGH)

    def monitor_func():
        time.sleep(1)
        return True

    def recovery1():
        return

    recovery = {NotALanguageExpression: recovery1}

    subplan = MonitorPlan(monitor_func, context, SequentialPlan(context, act, act2))
    plan = RetryMonitor(subplan, max_tries=6, recovery=recovery)
    assert len(plan.recovery) == 1
    assert isinstance(plan.plan, MonitorPlan)


def test_retry_monitor_tries(immutable_model_world):
    world, robot_view, context = immutable_model_world

    def raise_failure():
        raise PlanFailure

    tries_counter = 0

    def monitor_func():
        nonlocal tries_counter
        tries_counter += 1
        return True

    act2 = MoveTorsoActionDescription([TorsoState.HIGH])
    fail = CodePlan(context, raise_failure)
    counter = CodePlan(context, monitor_func)

    subplan = SequentialPlan(context, counter, fail)
    plan = RetryMonitor(subplan, max_tries=6)
    with pytest.raises(PlanFailure):
        plan.perform()

    assert tries_counter == 6


def test_retry_monitor_recovery(immutable_model_world):
    world, robot_view, context = immutable_model_world
    recovery1_counter = 0
    recovery2_counter = 0

    def monitor_func():
        if not hasattr(monitor_func, "tries_counter"):
            monitor_func.tries_counter = 0
        if monitor_func.tries_counter % 2:
            monitor_func.tries_counter += 1
            raise NotALanguageExpression
        monitor_func.tries_counter += 1
        raise PlanFailure

    def recovery1():
        nonlocal recovery1_counter
        recovery1_counter += 1

    def recovery2():
        nonlocal recovery2_counter
        recovery2_counter += 1

    recovery = {NotALanguageExpression: recovery1, PlanFailure: recovery2}

    code = CodePlan(context, monitor_func)
    subplan = SequentialPlan(context, code)
    plan = RetryMonitor(subplan, max_tries=6, recovery=recovery)
    try:
        plan.perform()
    except PlanFailure:
        pass
    assert recovery1_counter == 2
    assert recovery2_counter == 3


def test_repeat_construction(immutable_model_world):
    world, robot_view, context = immutable_model_world
    act = ParkArmsActionDescription([Arms.BOTH])
    act2 = MoveTorsoActionDescription([TorsoState.HIGH])

    plan = RepeatPlan(context, 5, SequentialPlan(context, act, act2))
    assert len(plan.root.children) == 1
    assert isinstance(plan.root.children[0], SequentialNode)


def test_repeat_construction_error(immutable_model_world):
    world, robot_view, context = immutable_model_world
    act = ParkArmsActionDescription([Arms.BOTH])
    act2 = MoveTorsoActionDescription([TorsoState.HIGH])
    park = ParkArmsActionDescription([Arms.BOTH])

    with pytest.raises(AttributeError):
        RepeatPlan(context, park, SequentialPlan(context, act, act2))


def test_perform_desig(immutable_model_world):
    world, robot_view, context = immutable_model_world
    act = NavigateActionDescription(
        [PoseStamped.from_list([0.3, 0.3, 0], frame=world.root)]
    )
    act2 = MoveTorsoActionDescription([TorsoState.HIGH])
    act3 = ParkArmsActionDescription([Arms.BOTH])

    plan = SequentialPlan(context, act, act2, act3)
    with simulated_robot:
        plan.perform()
    np.testing.assert_almost_equal(
        robot_view.root.global_pose.to_np()[:3, 3], [0.3, 0.3, 0], decimal=1
    )
    assert world.state[
        world.get_degree_of_freedom_by_name("torso_lift_joint").id
    ].position == pytest.approx(0.3, abs=0.1)

    assert len(plan.nodes) == len(plan.all_nodes)
    assert len(plan.edges) == len(plan.all_nodes) - 1


def test_perform_single_designator(immutable_model_world):
    world, robot_view, context = immutable_model_world

    plan = SequentialPlan(context, MoveTorsoActionDescription([TorsoState.HIGH]))
    with simulated_robot:
        plan.perform()

    assert world.state[
        world.get_degree_of_freedom_by_name("torso_lift_joint").id
    ].position == pytest.approx(0.3, abs=0.1)

    assert len(plan.nodes) == len(plan.all_nodes)
    assert len(plan.edges) == len(plan.all_nodes) - 1


def test_perform_parallel(immutable_model_world):
    world, robot_view, context = immutable_model_world

    def check_thread_id(main_id):
        assert main_id != threading.get_ident()

    act = CodePlan(context, check_thread_id, {"main_id": threading.get_ident()})
    act2 = CodePlan(context, check_thread_id, {"main_id": threading.get_ident()})
    act3 = CodePlan(context, check_thread_id, {"main_id": threading.get_ident()})

    plan = ParallelPlan(context, act, act2, act3)
    with simulated_robot:
        plan.perform()


def test_perform_repeat(immutable_model_world):
    world, robot_view, context = immutable_model_world
    test_var = Fluent(0)

    def inc(var):
        var.set_value(var.get_value() + 1)

    plan = RepeatPlan(context, 10, CodePlan(context, lambda: inc(test_var)))
    with simulated_robot:
        plan.perform()
    assert test_var.get_value() == 10

    assert len(plan.nodes) == len(plan.all_nodes)
    assert len(plan.edges) == len(plan.all_nodes) - 1


def test_exception_sequential(immutable_model_world):
    world, robot_view, context = immutable_model_world

    def raise_except():
        raise PlanFailure

    act = NavigateActionDescription([PoseStamped().from_list(frame=world.root)])
    code = CodePlan(context, raise_except)

    plan = SequentialPlan(context, act, code)

    def perform_plan():
        with simulated_robot:
            _ = plan.perform()

    with pytest.raises(PlanFailure):
        perform_plan()
    assert len(plan.root.children) == 2
    assert plan.root.status == TaskStatus.FAILED


def test_exception_try_in_order(immutable_model_world):
    world, robot_view, context = immutable_model_world

    def raise_except():
        raise PlanFailure

    act = NavigateActionDescription([PoseStamped().from_list(frame=world.root)])
    code = CodePlan(context, raise_except)

    plan = TryInOrderPlan(context, act, code)
    with simulated_robot:
        _ = plan.perform()
    assert len(plan.root.children) == 2
    assert plan.root.status == TaskStatus.SUCCEEDED


def test_exception_parallel(immutable_model_world):
    world, robot_view, context = immutable_model_world

    def raise_except():
        raise PlanFailure

    act = NavigateActionDescription([PoseStamped()])
    code = CodePlan(context, raise_except)

    plan = ParallelPlan(context, act, code)
    with simulated_robot:
        _ = plan.perform()
    assert type(plan.root.reason) is PlanFailure
    assert plan.root.status == TaskStatus.FAILED


def test_exception_try_all(immutable_model_world):
    world, robot_view, context = immutable_model_world

    def raise_except():
        raise PlanFailure

    act = NavigateActionDescription([PoseStamped()])
    code = CodePlan(context, raise_except)

    plan = TryAllPLan(context, act, code)
    with simulated_robot:
        _ = plan.perform()

    assert type(plan.root) is TryAllNode
    assert plan.root.status == TaskStatus.SUCCEEDED


def test_monitor_resume(immutable_model_world):
    world, robot_view, context = immutable_model_world
    act = ParkArmsActionDescription(Arms.BOTH)
    act2 = MoveTorsoActionDescription(TorsoState.HIGH)

    def monitor_func():
        time.sleep(2)
        return True

    plan = MonitorPlan(
        monitor_func,
        context,
        SequentialPlan(context, act, act2),
        behavior=MonitorBehavior.RESUME,
    )
    with simulated_robot:
        plan.perform()
    assert len(plan.root.children) == 1
    assert isinstance(plan.root, MonitorNode)
    assert plan.root.status == TaskStatus.SUCCEEDED
