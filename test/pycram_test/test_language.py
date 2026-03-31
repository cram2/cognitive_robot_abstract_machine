import threading
import time

import numpy as np
import pytest

from pycram.datastructures.enums import TaskStatus, MonitorBehavior, DetectionTechnique

from pycram.plans.failures import PlanFailure
from pycram.fluent import Fluent
from pycram.language import (
    MonitorNode,
    SequentialNode,
    TryAllNode,
    ParallelNode,
    TryInOrderNode,
)
from pycram.motion_executor import simulated_robot
from pycram.plans.factories import (
    sequential,
    parallel,
    try_in_order,
    try_all,
    monitor,
    repeat,
    code,
)
from pycram.robot_plans import *
from pycram.robot_plans.actions.core.misc import DetectAction
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from semantic_digital_twin.datastructures.definitions import TorsoState


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


def test_monitor_construction():
    act = ParkArmsAction(Arms.BOTH)
    act2 = MoveTorsoAction(TorsoState.HIGH)

    def monitor_func():
        return True

    root = monitor(children=[sequential([act, act2])], condition=monitor_func)
    assert len(root.children) == 1
    assert isinstance(root, MonitorNode)
    root.plan.validate()


def test_repeat_construction():
    act = ParkArmsAction(Arms.BOTH)
    act2 = MoveTorsoAction(TorsoState.HIGH)

    root = repeat([act, act2], 10)
    assert len(root.children) == 2
    root.plan.validate()


def test_perform_execute_single(immutable_model_world):
    world, robot_view, context = immutable_model_world
    act = NavigateAction(Pose.from_xyz_rpy(0.3, 0.3, 0, reference_frame=world.root))
    act2 = MoveTorsoAction(TorsoState.HIGH)
    act3 = ParkArmsAction(Arms.BOTH)

    plan = sequential([act, act2, act3], context).plan
    with simulated_robot:
        plan.perform()
    np.testing.assert_almost_equal(
        robot_view.root.global_transform.to_np()[:3, 3], [0.3, 0.3, 0], decimal=1
    )
    assert world.state[
        world.get_degree_of_freedom_by_name("torso_lift_joint").id
    ].position == pytest.approx(0.3, abs=0.1)

    plan.validate()


def test_perform_single_designator(immutable_model_world):
    world, robot_view, context = immutable_model_world

    plan = sequential([MoveTorsoAction(TorsoState.HIGH)], context).plan
    with simulated_robot:
        plan.perform()

    assert world.state[
        world.get_degree_of_freedom_by_name("torso_lift_joint").id
    ].position == pytest.approx(0.3, abs=0.1)

    plan.validate()


def test_perform_parallel(immutable_model_world):
    world, robot_view, context = immutable_model_world

    def check_thread_id(main_id):
        assert main_id != threading.get_ident()

    act = code(lambda: check_thread_id(threading.get_ident()), context=context)
    act2 = code(lambda: check_thread_id(threading.get_ident()), context=context)
    act3 = code(lambda: check_thread_id(threading.get_ident()), context=context)

    plan = parallel([act, act2, act3], context).plan
    with simulated_robot:
        plan.perform()
    plan.validate()


def test_perform_repeat(immutable_model_world):
    world, robot_view, context = immutable_model_world
    test_var = Fluent(0)

    def inc(var):
        var.set_value(var.get_value() + 1)

    plan = repeat([code(lambda: inc(test_var))], 10, context=context).plan
    with simulated_robot:
        plan.perform()
    assert test_var.get_value() == 10
    plan.validate()


def test_exception_sequential(immutable_model_world):
    world, robot_view, context = immutable_model_world

    def raise_except():
        raise PlanFailure()

    act = NavigateAction(Pose(reference_frame=world.root))
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


def test_exception_try_in_order(immutable_model_world):
    world, robot_view, context = immutable_model_world

    def raise_except():
        raise PlanFailure()

    act = NavigateAction(Pose(reference_frame=world.root))
    act2 = code(raise_except)

    plan = try_in_order([act, act2], context).plan
    with simulated_robot:
        _ = plan.perform()
    assert len(plan.root.children) == 2
    assert plan.root.status == TaskStatus.SUCCEEDED


def test_exception_parallel(immutable_model_world):
    world, robot_view, context = immutable_model_world

    def raise_except():
        raise PlanFailure()

    act = NavigateAction(Pose(reference_frame=world.root))
    act2 = code(raise_except)

    plan = parallel([act, act2], context).plan
    with pytest.raises(PlanFailure):
        with simulated_robot:
            _ = plan.perform()
    assert type(plan.root.reason) is PlanFailure
    assert plan.root.status == TaskStatus.FAILED


def test_exception_try_all(immutable_model_world):
    world, robot_view, context = immutable_model_world

    def raise_except():
        raise PlanFailure()

    act = NavigateAction(Pose(reference_frame=world.root))
    act2 = code(raise_except)

    plan = try_all([act, act2], context).plan
    with simulated_robot:
        _ = plan.perform()

    assert type(plan.root) is TryAllNode
    assert plan.root.status == TaskStatus.SUCCEEDED


def test_monitor_resume(immutable_model_world):
    world, robot_view, context = immutable_model_world
    act = ParkArmsAction(Arms.BOTH)
    act2 = MoveTorsoAction(TorsoState.HIGH)

    def monitor_func():
        time.sleep(2)
        return True

    plan = monitor(
        [
            sequential([act, act2]),
        ],
        condition=monitor_func,
        behavior=MonitorBehavior.RESUME,
        context=context,
    ).plan
    with simulated_robot:
        plan.perform()
    assert len(plan.root.children) == 1
    assert isinstance(plan.root, MonitorNode)
    assert plan.root.status == TaskStatus.SUCCEEDED
