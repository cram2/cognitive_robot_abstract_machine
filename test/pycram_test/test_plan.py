import os
import time

import pytest

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import (
    variable_from,
    underspecified,
    variable,
)
from krrood.parametrization.model_registries import (
    DictRegistry,
    FullyFactorizedRegistry,
)
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from probabilistic_model.probabilistic_circuit.rx.helper import fully_factorized
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import TaskStatus
from pycram.exceptions import MotionDidNotFinish
from pycram.language import CodeNode
from pycram.motion_executor import simulated_robot
from pycram.orm.ormatic_interface import *  # type: ignore
from pycram.plans.factories import code, sequential, parallel, execute_single
from pycram.plans.plan import (
    Plan,
)
from pycram.plans.plan_node import (
    PlanNode,
    ActionNode,
    MotionNode,
    UnderspecifiedNode,
)
from pycram.robot_plans import *
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.robot_plans.actions.core.pick_up import PickUpAction
from pycram.robot_plans.actions.core.robot_body import MoveTorsoAction
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.orm.model import (
    Point3Mapping,
    QuaternionMapping,
    PoseMapping,
)
from semantic_digital_twin.robots.abstract_robot import (
    Manipulator,
)


@pytest.fixture(scope="session")
def urdf_context():
    """Build a fresh URDF-based world and context for plan graph unit tests."""
    Plan.current_plan = None
    world = URDFParser.from_file(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "pycram",
            "resources",
            "robots",
            "pr2.urdf",
        )
    ).parse()
    context = Context(world, None, None)
    return world, context


# ---- Plan graph tests (no robot/world side effects needed) ----


def test_plan_construction():
    node = PlanNode()
    plan = Plan()
    plan.add_node(node)
    assert node == plan.root
    assert len(plan.edges) == 0
    assert len(plan.nodes) == 1
    assert plan == node.plan


def test_add_edge():
    node = PlanNode()
    plan = Plan()
    plan.add_node(node)
    node2 = PlanNode()
    plan.add_edge(node, node2)
    assert node == plan.root
    assert node in plan.nodes
    assert len(plan.nodes) == 2
    assert len(plan.edges) == 1
    assert node2 in plan.nodes
    assert plan is node2.plan


def test_add_node():
    plan = Plan()
    node = PlanNode()
    node2 = PlanNode()
    plan.add_node(node)
    plan.add_node(node2)
    assert node in plan.all_nodes
    assert node2 in plan.all_nodes
    assert (node, node2) not in plan.edges
    assert plan is node2.plan


def test_add_edge_with_layer_index():
    """
    Test that the layer index is correctly set when adding an edge.
    """

    root = PlanNode()
    plan = Plan()
    plan.add_node(root)
    child1 = PlanNode()
    child2 = PlanNode()
    plan.add_edge(root, child1)
    plan.add_edge(root, child2)

    child3 = PlanNode()
    plan.add_edge(root, child3, 1)
    assert root.layer_index == 0
    assert child1.layer_index == 0
    assert child2.layer_index == 2
    assert child3.layer_index == 1


def test_plan_all_parents():
    plan = Plan()
    node = PlanNode()
    node2 = PlanNode()
    plan.add_edge(node, node2)
    node3 = PlanNode()
    plan.add_edge(node2, node3)

    assert node.path == []
    assert node2.path == [node]
    assert node3.path == [node2, node]


def test_plan_node_children():

    plan = Plan()
    node = PlanNode()
    plan.add_node(node)
    assert [] == node.children

    node2 = PlanNode()
    plan.add_edge(node, node2)
    assert [node2] == node.children

    node3 = PlanNode()
    plan.add_edge(node, node3)
    assert [node2, node3] == node.children


def test_plan_node_recursive_children():
    node = PlanNode()
    plan = Plan()
    plan.add_node(node)

    assert [] == node.descendants

    node2 = PlanNode()
    plan.add_edge(node, node2)
    assert [node2] == node.descendants

    node3 = PlanNode()
    plan.add_edge(node2, node3)
    assert [node2, node3] == node.descendants


def test_plan_node_is_leaf():
    node = PlanNode()
    plan = Plan()
    node2 = PlanNode()
    plan.add_edge(node, node2)

    assert not node.is_leaf
    assert node2.is_leaf


def test_plan_node_subtree():
    node = PlanNode()
    node2 = PlanNode()
    node3 = PlanNode()
    plan = Plan()
    plan.add_edge(node, node2)
    plan.add_edge(node2, node3)

    assert node.descendants == [node2, node3]
    assert node2.descendants == [node3]


def test_plan_layers():

    node = PlanNode()
    node1 = PlanNode()
    node2 = PlanNode()
    node3 = PlanNode()
    plan = Plan()
    plan.add_edge(node, node1)
    plan.add_edge(node, node2)
    plan.add_edge(node2, node3)

    layers = plan.layers
    assert len(layers) == 3
    assert node in layers[0]
    assert node2 in layers[1]
    assert node3 in layers[2]

    assert layers[0] == [node]
    assert layers[1] == [node1, node2]
    assert layers[2] == [node3]


def test_depth_first_nodes_order():

    root = PlanNode()
    node1 = PlanNode()
    node2 = PlanNode()
    node3 = PlanNode()
    node4 = PlanNode()

    plan = Plan()

    plan.add_edge(root, node1)
    plan.add_edge(root, node3)
    plan.add_edge(node1, node2)
    plan.add_edge(node3, node4)

    assert len(plan.nodes) == 5

    assert plan.nodes == [root, node1, node2, node3, node4]


def test_layer_position():

    root = PlanNode()
    node1 = PlanNode()
    node2 = PlanNode()
    node3 = PlanNode()
    node4 = PlanNode()
    node5 = PlanNode()

    plan = Plan()
    plan.add_edge(root, node1)
    plan.add_edge(node1, node2)
    plan.add_edge(root, node3)
    plan.add_edge(node3, node4)
    plan.add_edge(node3, node5)

    assert root.layer_index == 0
    assert node1.layer_index == 0
    assert node3.layer_index == 1
    assert node2.layer_index == 0
    assert node4.layer_index == 0
    assert node5.layer_index == 1


def test_set_layer_index_insert_before():

    root = PlanNode()
    node1 = PlanNode()
    node2 = PlanNode()
    node3 = PlanNode()
    node4 = PlanNode()

    plan = Plan()
    plan.add_edge(root, node1)
    plan.add_edge(root, node2)
    plan.add_edge(root, node3)

    plan.add_edge(root, node4, node2.layer_index)
    assert node1.layer_index == 0
    assert node2.layer_index == 2
    assert node3.layer_index == 3
    assert node4.layer_index == 1


def test_get_previous_nodes():

    root = PlanNode()
    node1 = PlanNode()
    node2 = PlanNode()
    node3 = PlanNode()
    node4 = PlanNode()
    node5 = PlanNode()

    plan = Plan()
    plan.add_edge(root, node1)
    plan.add_edge(node1, node2)
    plan.add_edge(root, node3)
    plan.add_edge(node3, node4)
    plan.add_edge(node3, node5)

    assert node1.left_siblings == []
    assert node1.right_siblings == [node3]


# ---- Tests interacting with simulated robot/world ----


def test_interrupt_plan(immutable_model_world):
    world, robot_view, context = immutable_model_world

    def _interrupt_plan(code_node: CodeNode):
        code_node.plan.root.interrupt()

    act1 = MoveTorsoAction(TorsoState.HIGH)

    act2 = code(_interrupt_plan)
    act2.code = lambda: _interrupt_plan(act2)

    act3 = MoveTorsoAction(TorsoState.LOW)

    plan = sequential([act1, act2, act3], context=context).plan

    with simulated_robot:
        plan.perform()

    assert world.state[
        world.get_degree_of_freedom_by_name("torso_lift_joint").id
    ].position == pytest.approx(0.3, abs=0.1)


def test_pause_plan(immutable_model_world):
    world, robot_view, context = immutable_model_world

    def node_sleep():
        time.sleep(1)

    def pause_plan(node):
        node.pause()
        assert world.state[
            world.get_degree_of_freedom_by_name("torso_lift_joint").id
        ].position == pytest.approx(0.0, abs=0.1)
        node.resume()

        time.sleep(3)

        assert world.state[
            world.get_degree_of_freedom_by_name("torso_lift_joint").id
        ].position == pytest.approx(0.3, abs=0.1)

    code_node = code(function=lambda: None)
    code_node.code = lambda: pause_plan(code_node)
    sleep_node = code(lambda: node_sleep())
    robot_plan = sequential([sleep_node, MoveTorsoAction(TorsoState.HIGH)])
    plan = parallel([code_node, robot_plan], context=context).plan
    with simulated_robot:
        plan.perform()

    assert world.state[
        world.get_degree_of_freedom_by_name("torso_lift_joint").id
    ].position == pytest.approx(0.3, abs=0.1)


def test_algebra_sequential_plan(mutable_model_world):
    """
    Parameterize a SequentialPlan using krrood parameterizer, create a fully-factorized distribution and
    assert the correctness of sampled values after conditioning and truncation.
    """
    world, robot_view, context = mutable_model_world
    context.evaluate_conditions = False

    target_location = underspecified(PoseMapping.from_point_mapping_quaternion_mapping)(
        point_mapping=underspecified(Point3Mapping)(
            x=..., y=..., z=0.0, reference_frame=None
        ),
        quaternion_mapping=QuaternionMapping(x=0, y=0, z=0, w=1, reference_frame=None),
        reference_frame=variable_from([robot_view.root]),
    )

    navigate_action = underspecified(NavigateAction)(
        target_location=target_location,
    )
    navigate_action.resolve()

    context.query_backend = ProbabilisticBackend(
        model_registry=FullyFactorizedRegistry()
    )

    # resolved_navigate = next(pm_backend.evaluate(navigate_action))
    plan = sequential([MoveTorsoAction(TorsoState.LOW), navigate_action], context).plan

    with simulated_robot:
        plan.perform()

    assert isinstance(plan.root.children[1].children[0].designator, NavigateAction)
    assert len(plan.root.children[1].children) == 1


def test_parameterization_of_pick_up(mutable_model_world):
    world, robot_view, context = mutable_model_world
    context.evaluate_conditions = False

    milk = world.get_body_by_name("milk.stl")

    milk_variable = variable_from([milk])

    pick_up_description = underspecified(PickUpAction)(
        object_designator=milk_variable,
        arm=...,
        grasp_description=underspecified(GraspDescription)(
            approach_direction=...,
            vertical_alignment=...,
            rotate_gripper=...,
            manipulation_offset=0.05,
            manipulator=variable(Manipulator, world.semantic_annotations),
        ),
    )
    pick_up_description.resolve()

    parameters = UnderspecifiedParameters(pick_up_description)

    assert len(parameters.variables) == 7

    [manipulator_offset] = [
        v
        for v in parameters.variables.values()
        if v.name.endswith("manipulation_offset")
    ]

    assert parameters.assignments_for_conditioning[manipulator_offset] == 0.05

    context.query_backend = ProbabilisticBackend(
        model_registry=FullyFactorizedRegistry()
    )

    plan = execute_single(pick_up_description, context)

    with simulated_robot:
        try:
            plan.perform()
        except MotionDidNotFinish:
            pass
