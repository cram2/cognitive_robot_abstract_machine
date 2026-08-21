"""
Tests for the kitchen fridge demonstration's container handling: finding what has to be
opened to reach an object, and putting the opening and shutting into a plan that only
describes the transport.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.plans.plan_node import ActionNode, UnderspecifiedNode
from coraplex.robot_plans.actions.core.container import CloseAction, OpenAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from coraplex.view_manager import ViewManager
from krrood.entity_query_language.factories import an, entity, the, variable
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    CounterTop,
    Fridge,
    Milk,
    ShelfLayer,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Mesh
from semantic_digital_twin.world_description.world_entity import Body

# %% loading the demo


def load_demo():
    """
    :return: The demo module, which lives outside any package and so cannot be imported.
    """
    spec = importlib.util.spec_from_file_location(
        "kitchen_fridge_demo",
        Path(__file__).resolve().parents[2]
        / "coraplex"
        / "demos"
        / "coraplex_kitchen_fridge_demo"
        / "demo.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


demo = load_demo()


# %% the scene the demo acts in


@pytest.fixture(scope="module")
def demonstration() -> "demo.KitchenFridgeDemonstration":
    return demo.KitchenFridgeDemonstration(used_robot=PR2)


@pytest.fixture(scope="module")
def fridge_world(demonstration) -> World:
    world = demonstration.build_simulated_world()
    demonstration.populate_scene(world)
    return world


@pytest.fixture(scope="module")
def fridge_context(demonstration, fridge_world) -> Context:
    """
    The plan context, built without a ROS node so nothing here needs a running
    controller.
    """
    robot = variable(demonstration.used_robot, domain=fridge_world.semantic_annotations)
    return Context(
        world=fridge_world,
        robot=the(entity(robot)).first(),
        evaluate_conditions=False,
    )


@pytest.fixture(scope="module")
def milk_body(fridge_world) -> Body:
    milk = variable(Milk, domain=fridge_world.semantic_annotations)
    return the(entity(milk)).first().root


@pytest.fixture(scope="module")
def fridge_handle(fridge_world) -> Body:
    """
    The handle of the fridge door, which is what has to be pulled to reach the milk.
    """
    fridge = variable(Fridge, domain=fridge_world.semantic_annotations)
    return the(entity(fridge)).first().doors[0].handle.root


@pytest.fixture(scope="module")
def milk_grasp(demonstration, fridge_context) -> GraspDescription:
    return GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        ViewManager.get_end_effector_view(demonstration.milk_arm, fridge_context.robot),
    )


# %% building the scene


def test_the_kitchen_is_furnished_before_the_milk_arrives(demonstration):
    """
    Building the world stands the shelf layer in the fridge; the milk only turns up once
    the scene is populated.
    """
    world = demonstration.build_simulated_world()
    fridge = the(entity(variable(Fridge, domain=world.semantic_annotations))).first()
    shelf_layer = the(
        entity(variable(ShelfLayer, domain=world.semantic_annotations))
    ).first()

    assert shelf_layer in fridge.shelf_layers
    assert (
        next(
            an(entity(variable(Milk, domain=world.semantic_annotations))).evaluate(),
            None,
        )
        is None
    )
    assert not demonstration.is_scene_populated(world)

    demonstration.populate_scene(world)

    assert demonstration.is_scene_populated(world)


def test_the_milk_is_spawned_from_its_mesh(demonstration, milk_body):
    """
    The carton is the mesh the resources ship, not a box standing in for it.
    """
    [shape] = milk_body.collision.shapes

    assert isinstance(shape, Mesh)
    assert Path(shape.filename) == demonstration.milk_mesh_path


def test_the_milk_stands_on_the_shelf_layer(fridge_world, milk_body):
    """
    The mesh's own origin lies below the carton's centre, so the carton has to be stood
    up against its geometry rather than against its frame.
    """
    shelf_layer = the(
        entity(variable(ShelfLayer, domain=fridge_world.semantic_annotations))
    ).first()

    milk_bounds = milk_body.collision.as_bounding_box_collection_in_frame(
        fridge_world.root
    ).bounding_box()
    shelf_bounds = shelf_layer.root.collision.as_bounding_box_collection_in_frame(
        fridge_world.root
    ).bounding_box()

    assert milk_bounds.min_z == pytest.approx(shelf_bounds.max_z)


def test_the_milk_is_put_down_on_a_counter_top(demonstration, fridge_world):
    """
    The surface the demonstration places onto is one the reasoner recognises as a
    counter top, rather than an anonymous body it only knows by name.
    """
    counter_tops = fridge_world.get_semantic_annotations_by_type(CounterTop)

    assert demonstration.island_counter_top(fridge_world) in counter_tops


# %% finding what has to be opened


def test_handle_of_a_body_behind_a_door(
    demonstration, fridge_world, milk_body, fridge_handle
):
    """
    The milk stands in the fridge, so it is the fridge door's handle that has to be
    pulled.
    """
    assert (
        demonstration.handle_of_enclosing_container(milk_body, fridge_world)
        is fridge_handle
    )


def test_body_standing_in_the_open_has_no_handle(demonstration, fridge_world):
    """
    The kitchen island surface is in nothing that opens, so there is nothing to pull.
    """
    island_surface = demonstration.island_counter_top(fridge_world).root

    assert (
        demonstration.handle_of_enclosing_container(island_surface, fridge_world)
        is None
    )


# %% adding the opening and the shutting


def test_container_steps_enclose_the_given_steps(
    demonstration, fridge_context, milk_body, milk_grasp, fridge_handle
):
    """
    The opening goes in front of the given steps and the shutting behind them, leaving
    those steps themselves untouched.
    """
    transport = [
        PickUpAction(milk_body, demonstration.milk_arm, milk_grasp),
        ParkArmsAction(Arms.BOTH),
    ]

    steps = demonstration.add_container_opening_and_closing(
        transport, milk_body, fridge_context
    )

    assert steps[3:-3] == transport
    assert steps[3] is transport[0]
    assert steps[0].factory is NavigateAction
    assert steps[-3].factory is NavigateAction
    assert isinstance(steps[2], ParkArmsAction)
    assert isinstance(steps[-1], ParkArmsAction)

    opening = steps[1]
    assert opening.factory is OpenAction
    assert opening.kwargs["object_designator"] is fridge_handle
    assert opening.kwargs["goal_joint_state"] > 0.0

    shutting = steps[-2]
    assert shutting.factory is CloseAction
    assert shutting.kwargs["object_designator"] is fridge_handle
    assert shutting.kwargs["goal_joint_state"] < opening.kwargs["goal_joint_state"]


def test_container_steps_leave_their_arm_open(demonstration, fridge_context, milk_body):
    """
    Neither the opening nor the shutting says which hand pulls the handle, so the plan
    settles that against a hand that can reach it from where it ended up standing.
    """
    steps = demonstration.add_container_opening_and_closing(
        [ParkArmsAction(Arms.BOTH)], milk_body, fridge_context
    )

    assert steps[1].kwargs["arm"] is Ellipsis
    assert steps[-2].kwargs["arm"] is Ellipsis


def test_the_opening_and_the_shutting_pull_from_the_same_side():
    """
    The demonstration names no approach direction, so the opening and the shutting come
    at the handle from the same side only through their own defaults.
    """
    assert CloseAction.approach_direction is OpenAction.approach_direction


def test_nothing_is_added_for_a_body_in_the_open(
    demonstration, fridge_world, fridge_context
):
    """
    A body that stands in nothing that opens leaves the plan as it was.
    """
    island_surface = demonstration.island_counter_top(fridge_world).root
    transport = [ParkArmsAction(Arms.BOTH)]

    assert (
        demonstration.add_container_opening_and_closing(
            transport, island_surface, fridge_context
        )
        is transport
    )


# %% the whole plan


def test_plan_opens_the_fridge_and_shuts_it_again(demonstration, fridge_context):
    """
    The plan describes the transport only, and comes out with the fridge opened before
    the robot drives up to the milk and shut again once the milk is put down.
    """
    plan = demonstration.build_plan(fridge_context)

    actions = [
        (
            type(node.designator)
            if isinstance(node, ActionNode)
            else node.underspecified_action.factory
        )
        for node in plan.children
    ]

    assert actions == [
        ParkArmsAction,
        MoveTorsoAction,
        NavigateAction,
        OpenAction,
        ParkArmsAction,
        NavigateAction,
        PickUpAction,
        ParkArmsAction,
        NavigateAction,
        PlaceAction,
        ParkArmsAction,
        NavigateAction,
        CloseAction,
        ParkArmsAction,
    ]


def test_pick_up_leaves_its_approach_open(demonstration, fridge_context):
    """
    The pick-up says how the gripper is held but not which side it comes from, so the
    plan settles that once it is standing in front of the milk.
    """
    plan = demonstration.build_plan(fridge_context)
    pick_up = next(
        node
        for node in plan.children
        if isinstance(node, UnderspecifiedNode)
        and node.underspecified_action.factory is PickUpAction
    ).underspecified_action

    grasp = pick_up.kwargs["grasp_description"]

    assert grasp.factory is GraspDescription
    assert grasp.kwargs["approach_direction"] is Ellipsis
    assert grasp.kwargs["vertical_alignment"] is VerticalAlignment.NoAlignment
