import pytest
import numpy as np

from pycram.datastructures.grasp import GraspDescription
from pycram.datastructures.partial_designator import PartialDesignator
from pycram.datastructures.pose import PoseStamped
from pycram.language import SequentialPlan
from pycram.robot_plans import (
    PickUpAction,
    PickUpAction,
    SetGripperAction,
    MoveTorsoAction,
    NavigateAction,
    MoveTorsoActionDescription,
    NavigateActionDescription,
    PickUpActionDescription,
)
from pycram.designators.object_designator import BelieveObject
from pycram.datastructures.enums import (
    Arms,
    Grasp,
    ApproachDirection,
    VerticalAlignment,
)
from pycram.utils import is_iterable, lazy_product
from pycram.process_module import simulated_robot
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState


def test_partial_desig_construction():
    test_object = BelieveObject(names=["milk"])
    partial_desig = PartialDesignator(PickUpAction, test_object, arm=Arms.RIGHT)
    assert partial_desig.performable == PickUpAction
    assert partial_desig.kwargs == {
        "arm": Arms.RIGHT,
        "object_designator": test_object,
        "grasp_description": None,
    }


def test_partial_desig_construction_none():
    partial_desig = PartialDesignator(PickUpAction, None, arm=Arms.RIGHT)
    assert partial_desig.performable == PickUpAction
    assert partial_desig.kwargs == {
        "arm": Arms.RIGHT,
        "object_designator": None,
        "grasp_description": None,
    }


def test_partial_desig_call():
    partial_desig = PartialDesignator(PickUpAction, None, arm=Arms.RIGHT)
    grasp_description = GraspDescription(
        ApproachDirection.FRONT, VerticalAlignment.NoAlignment, False
    )
    new_partial_desig = partial_desig(grasp_description=grasp_description)
    assert new_partial_desig.performable == PickUpAction
    assert {
        "arm": Arms.RIGHT,
        "grasp_description": grasp_description,
        "object_designator": None,
    } == new_partial_desig.kwargs


def test_partial_desig_missing_params():
    partial_desig = PartialDesignator(PickUpAction, None, arm=Arms.RIGHT)
    missing_params = partial_desig.missing_parameter()
    assert (
        "object_designator" in missing_params and "grasp_description" in missing_params
    )

    grasp_description = GraspDescription(
        ApproachDirection.FRONT, VerticalAlignment.NoAlignment, False
    )
    new_partial = partial_desig(grasp_description=grasp_description)
    missing_params = new_partial.missing_parameter()
    assert ["object_designator"] == missing_params


def test_is_iterable():
    assert is_iterable([1, 2, 3])
    assert not is_iterable(1)


def test_partial_desig_permutations():
    tp = PartialDesignator(
        SetGripperAction,
        [Arms.LEFT, Arms.RIGHT],
        motion=[GripperState.OPEN, GripperState.CLOSE],
    )
    permutations = tp.generate_permutations()
    assert [
        (Arms.LEFT, GripperState.OPEN),
        (Arms.LEFT, GripperState.CLOSE),
        (Arms.RIGHT, GripperState.OPEN),
        (Arms.RIGHT, GripperState.CLOSE),
    ] == [tuple(p.values()) for p in permutations]


def test_partial_desig_permutation_dict():
    tp = PartialDesignator(
        SetGripperAction,
        [Arms.LEFT, Arms.RIGHT],
        motion=[GripperState.OPEN, GripperState.CLOSE],
    )
    permutations = tp.generate_permutations()
    assert {"gripper": Arms.LEFT, "motion": GripperState.OPEN} == list(permutations)[0]


def test_partial_desig_iter(immutable_model_world):
    world, robot_view, context = immutable_model_world
    partial_desig = PartialDesignator(
        PickUpAction,
        world.get_body_by_name("milk.stl"),
        arm=[Arms.RIGHT, Arms.LEFT],
    )
    grasp_description_front = GraspDescription(
        ApproachDirection.FRONT, VerticalAlignment.NoAlignment, False
    )
    grasp_description_top = GraspDescription(
        ApproachDirection.FRONT, VerticalAlignment.NoAlignment, False
    )
    performables = list(
        partial_desig(
            grasp_description=[grasp_description_front, grasp_description_top]
        )
    )
    assert 4 == len(performables)
    assert all([isinstance(p, PickUpAction) for p in performables])
    assert [p.arm for p in performables] == [
        Arms.RIGHT,
        Arms.RIGHT,
        Arms.LEFT,
        Arms.LEFT,
    ]
    assert [p.grasp_description for p in performables] == [
        grasp_description_front,
        grasp_description_top,
        grasp_description_front,
        grasp_description_top,
    ]
    assert [p.object_designator for p in performables] == [
        world.get_body_by_name("milk.stl")
    ] * 4


def test_partial_movetorso_action():
    move1 = MoveTorsoActionDescription(TorsoState.HIGH).resolve()
    assert move1.torso_state == TorsoState.HIGH
    move2 = MoveTorsoActionDescription([TorsoState.HIGH, TorsoState.MID])
    for action in move2:
        assert action.torso_state in [TorsoState.HIGH, TorsoState.MID]


def test_partial_navigate_action_perform(immutable_model_world):
    world, robot_view, context = immutable_model_world
    with simulated_robot:
        move1 = SequentialPlan(
            context,
            NavigateActionDescription(
                PoseStamped.from_list([1, 0, 0], frame=world.root)
            ),
        )
        move1.perform()
        np.testing.assert_almost_equal(
            list(robot_view.root.global_pose.to_np()[:3, 3]),
            [1, 0, 0],
            decimal=1,
        )


def test_partial_navigate_action_multiple(immutable_model_world):
    world, robot_view, context = immutable_model_world
    nav = NavigateActionDescription(
        [
            PoseStamped.from_list([1, 0, 0], frame=world.root),
            PoseStamped.from_list([2, 0, 0], frame=world.root),
            PoseStamped.from_list([3, 0, 0], frame=world.root),
        ]
    )
    nav_goals = [[1, 0, 0], [2, 0, 0], [3, 0, 0]]
    for i, action in enumerate(nav):
        with simulated_robot:
            SequentialPlan(context, action).perform()
            np.testing.assert_almost_equal(
                robot_view.root.global_pose.to_np()[:3, 3],
                nav_goals[i],
                decimal=2,
            )


def test_partial_pickup_action(immutable_model_world):
    world, robot_view, context = immutable_model_world
    grasp_description = GraspDescription(
        ApproachDirection.FRONT, VerticalAlignment.NoAlignment, False
    )
    pick = PickUpActionDescription(
        world.get_body_by_name("milk.stl"),
        [Arms.LEFT, Arms.RIGHT],
        grasp_description,
    )
    pick_action = pick.resolve()
    assert pick_action.object_designator == world.get_body_by_name("milk.stl")
    assert pick_action.arm == Arms.LEFT
    assert pick_action.grasp_description == grasp_description


def test_partial_pickup_action_insert_param(immutable_model_world):
    world, robot_view, context = immutable_model_world
    grasp_description = GraspDescription(
        ApproachDirection.FRONT, VerticalAlignment.NoAlignment, False
    )
    pick = PickUpActionDescription(
        world.get_body_by_name("milk.stl"), [Arms.LEFT, Arms.RIGHT]
    )
    pick_action = pick(grasp_description=grasp_description).resolve()
    assert pick_action.grasp_description == grasp_description


# Lazy product utility tests (do not require world)


def test_lazy_product_result():
    l1 = [0, 1]
    l2 = [3, 4]
    assert list(lazy_product(l1, l2)) == [(0, 3), (0, 4), (1, 3), (1, 4)]


def test_lazy_product_single_input():
    l1 = [0, 1]
    assert list(lazy_product(l1)) == [(0,), (1,)]


def test_lazy_product_lazy_evaluate():
    def bad_generator():
        for i in range(10):
            if i == 5:
                raise RuntimeError()
            yield i

    l1 = iter(bad_generator())
    l2 = iter(bad_generator())
    res = next(lazy_product(l1, l2))
    assert res == (0, 0)


def test_lazy_product_error():
    def bad_generator():
        for i in range(10):
            if i == 5:
                raise RuntimeError()
            yield i

    with pytest.raises(RuntimeError):
        list(lazy_product(bad_generator()))


def test_correct_error():
    def bad_generator():
        if False:
            yield 1

    with pytest.raises(RuntimeError) as cm:
        list(lazy_product(bad_generator()))
    assert "bad_generator" in str(cm.value)
