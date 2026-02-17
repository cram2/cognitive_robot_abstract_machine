import os
import time

import pytest

from random_events.product_algebra import SimpleEvent, Event

from krrood.probabilistic_knowledge.parameterizer import Parameterizer
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import TaskStatus
from pycram.language import ParallelPlan, CodeNode
from pycram.plan import PlanNode, Plan, ActionDescriptionNode, ActionNode, MotionNode
from pycram.motion_executor import simulated_robot
from pycram.robot_plans import *
from pycram.language import SequentialPlan, ParallelPlan, CodeNode
from pycram.plan import PlanNode, Plan, ActionDescriptionNode, ActionNode, MotionNode
from semantic_digital_twin.spatial_types import Quaternion
from semantic_digital_twin.adapters.urdf import URDFParser
from pycram.orm.ormatic_interface import *


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


def test_plan_construction(urdf_context):
    world, context = urdf_context
    node = PlanNode()
    plan = Plan(node, context)
    assert node == plan.root
    assert len(plan.edges) == 0
    assert len(plan.nodes) == 1
    assert plan == node.plan
    assert Plan.current_plan is None


def test_add_edge(urdf_context):
    world, context = urdf_context
    node = PlanNode()
    plan = Plan(node, context)
    node2 = PlanNode()
    plan.add_edge(node, node2)
    assert node == plan.root
    assert node in plan.nodes
    assert len(plan.nodes) == 2
    assert len(plan.edges) == 1
    assert node2 in plan.nodes
    assert plan == node2.plan


def test_add_node(urdf_context):
    world, context = urdf_context
    node = PlanNode()
    plan = Plan(node, context)
    node2 = PlanNode()
    plan.add_node(node2)
    assert node == plan.root
    assert node in plan.all_nodes
    assert node2 in plan.all_nodes
    assert (node, node2) not in plan.edges
    assert plan == node2.plan


def test_mount(urdf_context):
    world, context = urdf_context
    plan1_node = PlanNode()
    plan1 = Plan(plan1_node, context)
    plan2_node = PlanNode()
    plan2 = Plan(plan2_node, context)

    plan1.mount(plan2)
    assert plan2_node in plan1.nodes
    assert plan1 == plan2_node.plan
    assert len(plan1.edges) == 1
    assert len(plan1.nodes) == 2


def test_mount_specific_node(urdf_context):
    world, context = urdf_context
    plan = Plan(PlanNode(), context)
    mount_node = PlanNode()
    plan.add_edge(plan.root, mount_node)

    plan2 = Plan(PlanNode(), context)
    plan.mount(plan2, mount_node)

    assert plan2.root in plan.nodes
    assert plan == plan2.root.plan
    assert (mount_node, plan2.root) in plan.edges
    assert len(plan.edges) == 2
    assert len(plan.nodes) == 3


def test_context_creation(urdf_context):
    world, context = urdf_context
    super_plan = Plan(PlanNode(), context)
    ctx = Context(world, 1, super_plan)
    plan = Plan(PlanNode(), ctx)
    assert ctx == plan.context
    assert plan.world == world
    assert plan.robot == 1
    assert plan.super_plan == super_plan


# ---- PlanNode tests (pure graph behavior) ----


def test_plan_node_creation(urdf_context):
    world, context = urdf_context
    node = PlanNode()
    assert isinstance(node, PlanNode)
    assert node.status == TaskStatus.CREATED
    assert node.plan is None
    assert node.start_time <= datetime.datetime.now()


def test_plan_node_parent(urdf_context):
    world, context = urdf_context
    node = PlanNode()
    plan = Plan(node, context)
    node2 = PlanNode()
    plan.add_edge(node, node2)

    assert node.parent is None
    assert node2.parent == node


def test_plan_all_parents(urdf_context):
    world, context = urdf_context
    node = PlanNode()
    plan = Plan(node, context)
    node2 = PlanNode()
    plan.add_edge(node, node2)
    node3 = PlanNode()
    plan.add_edge(node2, node3)

    assert node.all_parents == []
    assert node2.all_parents == [node]
    assert node3.all_parents == [node2, node]


def test_plan_node_children(urdf_context):
    world, context = urdf_context
    node = PlanNode()
    plan = Plan(node, context)

    assert [] == node.children

    node2 = PlanNode()
    plan.add_edge(node, node2)
    assert [node2] == node.children

    node3 = PlanNode()
    plan.add_edge(node, node3)
    assert [node2, node3] == node.children


def test_plan_node_recursive_children(urdf_context):
    world, context = urdf_context
    node = PlanNode()
    plan = Plan(node, context)

    assert [] == node.recursive_children

    node2 = PlanNode()
    plan.add_edge(node, node2)
    assert [node2] == node.recursive_children

    node3 = PlanNode()
    plan.add_edge(node2, node3)
    assert [node2, node3] == node.recursive_children


def test_plan_node_is_leaf(urdf_context):
    world, context = urdf_context
    node = PlanNode()
    plan = Plan(node, context)
    node2 = PlanNode()
    plan.add_edge(node, node2)

    assert not node.is_leaf
    assert node2.is_leaf


def test_plan_node_subtree(urdf_context):
    world, context = urdf_context
    node = PlanNode()
    node2 = PlanNode()
    node3 = PlanNode()
    plan = Plan(node, context)
    plan.add_edge(node, node2)
    plan.add_edge(node2, node3)

    sub_tree = node2.subtree
    assert node2 == sub_tree.root
    assert node2 in sub_tree.nodes
    assert node3 in sub_tree.nodes
    assert len(sub_tree.edges) == 1
    assert (node2, node3) in sub_tree.edges


def test_plan_layers(urdf_context):
    world, context = urdf_context

    node = PlanNode()
    node1 = PlanNode()
    node2 = PlanNode()
    node3 = PlanNode()
    plan = Plan(node, context)
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


def test_get_action_node_by_type(urdf_context):
    world, context = urdf_context

    plan = SequentialPlan(
        context,
    )
    nav_node = ActionDescriptionNode(
        designator_ref=NavigateActionDescription(None),
        designator_type=NavigateAction,
        kwargs={},
    )
    plan.add_edge(plan.root, nav_node)

    pick_node = ActionDescriptionNode(
        designator_ref=PickUpActionDescription(None, None, None),
        designator_type=PickUpAction,
        kwargs={},
    )
    plan.add_edge(plan.root, pick_node)
    place_node = ActionDescriptionNode(
        designator_ref=PlaceActionDescription(None, None, None),
        designator_type=PlaceAction,
        kwargs={},
    )

    plan.add_edge(plan.root, place_node)

    assert nav_node in plan.get_nodes_by_designator_type(NavigateAction)
    assert pick_node in plan.get_nodes_by_designator_type(PickUpAction)
    assert place_node in plan.get_nodes_by_designator_type(PlaceAction)

    assert nav_node not in plan.get_nodes_by_designator_type(PickUpAction)
    assert pick_node not in plan.get_nodes_by_designator_type(NavigateAction)
    assert place_node not in plan.get_nodes_by_designator_type(PickUpAction)

    assert nav_node == plan.get_node_by_designator_type(NavigateAction)
    assert pick_node == plan.get_node_by_designator_type(PickUpAction)
    assert place_node == plan.get_node_by_designator_type(PlaceAction)


def test_get_layer_node_by_type(urdf_context):
    world, context = urdf_context

    plan = SequentialPlan(
        context,
        NavigateActionDescription(None),
        PickUpActionDescription(None, None, None),
    )
    place_node = ActionNode(
        designator_ref=PlaceAction(None, None, None),
        designator_type=PlaceAction,
        kwargs={},
    )
    plan.add_edge(plan.root, place_node)

    pick_node = plan.get_node_by_designator_type(PickUpAction)

    query_pick = plan.get_previous_node_by_designator_type(place_node, PickUpAction)

    assert query_pick == pick_node


def test_depth_first_nodes_order(urdf_context):
    world, context = urdf_context

    root = PlanNode()
    node1 = PlanNode()
    node2 = PlanNode()
    node3 = PlanNode()
    node4 = PlanNode()

    plan = Plan(root, context)

    plan.add_edge(root, node1)
    plan.add_edge(root, node3)
    plan.add_edge(node1, node2)
    plan.add_edge(node3, node4)

    assert len(plan.nodes) == 5

    assert plan.nodes == [root, node1, node2, node3, node4]


def test_layer_position(urdf_context):
    world, context = urdf_context

    root = PlanNode()
    node1 = PlanNode()
    node2 = PlanNode()
    node3 = PlanNode()
    node4 = PlanNode()
    node5 = PlanNode()

    plan = Plan(root, context)
    plan.add_edge(root, node1)
    plan.add_edge(node1, node2)
    plan.add_edge(root, node3)
    plan.add_edge(node3, node4)
    plan.add_edge(node3, node5)

    assert root.layer_index == 0
    assert node1.layer_index == 0
    assert node3.layer_index == 1
    assert node2.layer_index == 0
    assert node4.layer_index == 1
    assert node5.layer_index == 2


def test_find_nodes_to_shift_index(urdf_context):
    world, context = urdf_context
    root = PlanNode()
    node1 = PlanNode()
    node2 = PlanNode()
    node3 = PlanNode()
    node4 = PlanNode()
    node5 = PlanNode()

    plan = Plan(root, context)

    assert plan._find_nodes_to_shift_index(root) == (0, [])

    plan.add_edge(root, node1)

    assert plan._find_nodes_to_shift_index(root) == (1, [])

    plan.add_edge(root, node2)
    assert plan._find_nodes_to_shift_index(root) == (2, [])
    plan.add_edge(root, node3)

    assert plan._find_nodes_to_shift_index(node2) == (0, [])

    plan.add_edge(node2, node4)

    assert plan._find_nodes_to_shift_index(node1) == (0, [node4])
    plan.add_edge(node1, node5)

    assert plan._find_nodes_to_shift_index(node1) == (1, [node4])


def test_set_layer_index_insert_before(urdf_context):
    world, context = urdf_context

    root = PlanNode()
    node1 = PlanNode()
    node2 = PlanNode()
    node3 = PlanNode()
    node4 = PlanNode()

    plan = Plan(root, context)
    plan.add_edge(root, node1)
    plan.add_edge(root, node2)
    plan.add_edge(root, node3)

    plan._set_layer_indices(root, node4, node_to_insert_before=node2)

    assert root.layer_index == 0
    assert node1.layer_index == 0
    assert node4.layer_index == 1
    assert node2.layer_index == 2
    assert node3.layer_index == 3


def test_set_layer_index_insert_after(urdf_context):
    world, context = urdf_context

    root = PlanNode()
    node1 = PlanNode()
    node2 = PlanNode()
    node3 = PlanNode()
    node4 = PlanNode()

    plan = Plan(root, context)
    plan.add_edge(root, node1)
    plan.add_edge(root, node2)
    plan.add_edge(root, node3)

    plan._set_layer_indices(root, node4, node_to_insert_after=node2)

    assert root.layer_index == 0
    assert node1.layer_index == 0
    assert node2.layer_index == 1
    assert node4.layer_index == 2
    assert node3.layer_index == 3


def test_set_layer_index(urdf_context):
    world, context = urdf_context
    root = PlanNode()
    node1 = PlanNode()
    node2 = PlanNode()
    node3 = PlanNode()
    node4 = PlanNode()
    node5 = PlanNode()

    plan = Plan(root, context)
    plan.add_edge(root, node1)
    plan.add_edge(root, node2)
    plan.add_edge(root, node3)
    plan.add_edge(node3, node4)

    plan._set_layer_indices(node2, node5)

    assert root.layer_index == 0
    assert node4.layer_index == 1
    assert node5.layer_index == 0

    plan.add_edge(node2, node5)

    layers = plan.layers
    assert len(layers) == 3
    assert layers[0] == [root]
    assert layers[1] == [node1, node2, node3]
    assert layers[2] == [node5, node4]


def test_get_layer_by_node(urdf_context):
    world, context = urdf_context

    root = PlanNode()
    node1 = PlanNode()
    node2 = PlanNode()
    node3 = PlanNode()
    node4 = PlanNode()

    plan = Plan(root, context)

    plan.add_edge(root, node1)
    plan.add_edge(root, node3)
    plan.add_edge(node1, node2)
    plan.add_edge(node3, node4)

    assert plan.get_layer_by_node(node1) == [node1, node3]
    assert plan.get_layer_by_node(node2) == [node2, node4]
    assert plan.get_layer_by_node(root) == [root]


def test_get_previous_nodes(urdf_context):
    world, context = urdf_context

    root = PlanNode()
    node1 = PlanNode()
    node2 = PlanNode()
    node3 = PlanNode()
    node4 = PlanNode()
    node5 = PlanNode()

    plan = Plan(root, context)
    plan.add_edge(root, node1)
    plan.add_edge(node1, node2)
    plan.add_edge(root, node3)
    plan.add_edge(node3, node4)
    plan.add_edge(node3, node5)

    assert plan.nodes == [root, node1, node2, node3, node4, node5]
    assert plan.get_previous_nodes(node3) == [root, node1, node2]
    assert plan.get_previous_nodes(node1) == [root]
    assert plan.get_previous_nodes(node4) == [root, node1, node2, node3]

    assert plan.get_previous_nodes(node3, on_layer=True) == [node1]
    assert plan.get_previous_nodes(node4, on_layer=True) == [node2]
    assert plan.get_previous_nodes(node5, on_layer=True) == [node2, node4]


def test_get_following_nodes(urdf_context):
    world, context = urdf_context

    root = PlanNode()
    node1 = PlanNode()
    node2 = PlanNode()
    node3 = PlanNode()
    node4 = PlanNode()
    node5 = PlanNode()

    plan = Plan(root, context)
    plan.add_edge(root, node1)
    plan.add_edge(node1, node2)
    plan.add_edge(root, node3)
    plan.add_edge(node3, node4)
    plan.add_edge(node3, node5)

    assert plan.nodes == [root, node1, node2, node3, node4, node5]
    assert plan.get_following_nodes(node3) == [node4, node5]
    assert plan.get_following_nodes(root) == [node1, node2, node3, node4, node5]
    assert plan.get_following_nodes(node1) == [node2, node3, node4, node5]
    assert plan.get_following_nodes(node3) == [node4, node5]

    assert plan.get_following_nodes(node4, on_layer=True) == [node5]
    assert plan.get_following_nodes(node2, on_layer=True) == [node4, node5]
    assert plan.get_following_nodes(node1, on_layer=True) == [node3]


def test_get_previous_node_by_type(urdf_context):
    world, context = urdf_context
    node1 = PlanNode()
    node2 = PlanNode()

    nav_node = ActionNode(
        designator_ref=NavigateActionDescription(None), designator_type=NavigateAction
    )

    move_node = MotionNode(designator_ref=MoveMotion(None), designator_type=MoveMotion)

    plan = SequentialPlan(context)
    root = plan.root
    plan.add_edge(root, node1)
    plan.add_edge(node1, nav_node)
    plan.add_edge(root, node2)
    plan.add_edge(node2, move_node)


def test_get_prev_node_by_designator_type(urdf_context):
    world, context = urdf_context

    plan = SequentialPlan(
        context,
        NavigateActionDescription(None),
        PickUpActionDescription(None, None, None),
    )
    place_node = ActionNode(
        designator_ref=PlaceAction(None, None, None),
        designator_type=PlaceAction,
        kwargs={},
    )
    plan.add_edge(plan.root, place_node)

    pick_node = plan.get_node_by_designator_type(PickUpAction)

    query_pick = plan.get_previous_node_by_designator_type(place_node, PickUpAction)

    assert query_pick == pick_node

    query_pick_layer = plan.get_previous_node_by_designator_type(
        place_node, PickUpAction, on_layer=True
    )

    assert query_pick_layer == pick_node


def test_get_nodes_by_designator_type(urdf_context):
    world, context = urdf_context

    plan = SequentialPlan(
        context,
        NavigateActionDescription(None),
    )

    place_node = ActionNode(
        designator_ref=PlaceAction(None, None, None),
        designator_type=PlaceAction,
    )

    place_node2 = ActionNode(
        designator_ref=PlaceAction(None, None, None), designator_type=PlaceAction
    )

    plan.add_edge(plan.root, place_node)
    plan.add_edge(place_node, place_node2)

    query_nav = plan.get_node_by_designator_type(NavigateAction)

    assert plan.nodes == [plan.root, query_nav, place_node, place_node2]

    assert plan.get_nodes_by_designator_type(PlaceAction) == [place_node, place_node2]


# ---- Tests interacting with simulated robot/world ----


def test_interrupt_plan(immutable_model_world):
    world, robot_view, context = immutable_model_world

    def _interrupt_plan():
        Plan.current_plan.root.interrupt()

    code_node = CodeNode(_interrupt_plan)
    with simulated_robot:
        SequentialPlan(
            context,
            MoveTorsoActionDescription(TorsoState.HIGH),
            Plan(code_node, context),
            MoveTorsoActionDescription([TorsoState.LOW]),
        ).perform()

        assert world.state[
            world.get_degree_of_freedom_by_name("torso_lift_joint").id
        ].position == pytest.approx(0.3, abs=0.1)


@pytest.mark.skip(
    reason="There is some weird error here that causes the interpreter to abort with exit code 134, something with thread handling. Needs more investigation"
)
def test_pause_plan(immutable_model_world):
    world, robot_view, context = immutable_model_world

    def node_sleep():
        time.sleep(1)

    def pause_plan():
        Plan.current_plan.root.pause()
        assert (
            world.state[
                world.get_degree_of_freedom_by_name("torso_lift_joint").name
            ].position
            == 0
        )
        Plan.current_plan.root.resume()
        time.sleep(3)
        assert (
            world.state[
                world.get_degree_of_freedom_by_name("torso_lift_joint").name
            ].position
            == 0.3
        )

    code_node = CodeNode(pause_plan)
    sleep_node = CodeNode(node_sleep)
    robot_plan = SequentialPlan(
        context,
        Plan(sleep_node, context),
        MoveTorsoActionDescription([TorsoState.HIGH]),
    )

    with simulated_robot:
        ParallelPlan(context, Plan(code_node, context), robot_plan).perform()

    assert (
        world.state[
            world.get_degree_of_freedom_by_name("torso_lift_joint").name
        ].position
        == 0.3
    )


def test_algebra_sequentialplan(immutable_model_world):
    """
    Parameterize a SequentialPlan using krrood parameterizer, create a fully-factorized distribution and
    assert the correctness of sampled values after conditioning and truncation.
    """
    world, robot_view, context = immutable_model_world

    sp = SequentialPlan(
        context,
        MoveTorsoActionDescription(TorsoState.LOW),
        NavigateActionDescription(None),
        MoveTorsoActionDescription(None),
    )

    parameterization = sp.parameterize()
    variables_map = {v.name: v for v in parameterization.variables}

    probabilistic_circuit = sp.create_fully_factorized_distribution()

    torso_1 = variables_map["MoveTorsoAction_0.torso_state"]
    torso_2 = variables_map["MoveTorsoAction_2.torso_state"]
    consistency_events = [
        SimpleEvent({torso_1: [state], torso_2: [state]}) for state in TorsoState
    ]
    restricted_distribution, _ = probabilistic_circuit.truncated(
        Event(*consistency_events)
    )
    restricted_distribution.normalize()

    navigate_action_constraints = {
        variables_map["NavigateAction_1.target_location.position.z"]: 0,
        variables_map["NavigateAction_1.target_location.rotation.x"]: 0,
        variables_map["NavigateAction_1.target_location.rotation.y"]: 0,
        variables_map["NavigateAction_1.target_location.rotation.z"]: 0,
        variables_map["NavigateAction_1.target_location.rotation.w"]: 1,
    }
    final_distribution, _ = restricted_distribution.conditional(
        navigate_action_constraints
    )
    final_distribution.normalize()

    nav_x = variables_map["NavigateAction_1.target_location.position.x"]
    nav_y = variables_map["NavigateAction_1.target_location.position.y"]
    nav_z = next(
        v
        for v in final_distribution.variables
        if v.name == "NavigateAction_1.target_location.position.z"
    )
    nav_ox_var = next(
        v
        for v in final_distribution.variables
        if v.name == "NavigateAction_1.target_location.rotation.x"
    )
    nav_oy_var = next(
        v
        for v in final_distribution.variables
        if v.name == "NavigateAction_1.target_location.rotation.y"
    )
    nav_oz_var = next(
        v
        for v in final_distribution.variables
        if v.name == "NavigateAction_1.target_location.rotation.z"
    )
    nav_ow_var = next(
        v
        for v in final_distribution.variables
        if v.name == "NavigateAction_1.target_location.rotation.w"
    )

    for sample_values in final_distribution.sample(10):
        sample = dict(zip(final_distribution.variables, sample_values))
        assert nav_x in sample
        assert nav_y in sample
        assert sample[nav_z] == 0.0
        assert sample[nav_ox_var] == 0.0
        assert sample[nav_oy_var] == 0.0
        assert sample[nav_oz_var] == 0.0
        assert sample[nav_ow_var] == 1.0


def test_algebra_parallelplan(immutable_model_world):
    """
    Parameterize a ParallelPlan using krrood parameterizer, create a fully-factorized distribution and
    assert the correctness of sampled values after truncation.
    """
    world, robot_view, context = immutable_model_world

    sp = ParallelPlan(
        context,
        MoveTorsoActionDescription(None),
        ParkArmsActionDescription(None),
    )

    parameterization = sp.parameterize()
    variables_map = {v.name: v for v in parameterization.variables}

    # Ensure expected variable names exist
    assert "MoveTorsoAction_0.torso_state" in variables_map
    assert "ParkArmsAction_1.arm" in variables_map

    probabilistic_circuit = sp.parameterizer.create_fully_factorized_distribution()

    arm_var = variables_map["ParkArmsAction_1.arm"]
    torso_var = variables_map["MoveTorsoAction_0.torso_state"]

    # Truncate distribution to force arm == Arms.BOTH
    restricted_dist, _ = probabilistic_circuit.truncated(
        Event(SimpleEvent({arm_var: [Arms.BOTH]}))
    )
    restricted_dist.normalize()

    for sample_values in restricted_dist.sample(5):
        sample = dict(zip(restricted_dist.variables, sample_values))
        assert sample[arm_var] == Arms.BOTH
        assert torso_var in sample


def test_parameterize_move_torse_navigate(immutable_model_world):
    """
    Test parameterization of a potential robot plan consisting of: MoveTorso - Navigate - MoveTorso.

    This test verifies:
    1. Parameterization of simple robot action plan
    2. Sampling from the constrained distribution and validation of constraints.
    """
    world, robot_view, context = immutable_model_world

    plan = SequentialPlan(
        context,
        MoveTorsoActionDescription(None),
        NavigateActionDescription(None),
        MoveTorsoActionDescription(None),
    )

    parameterization = plan.parameterize()

    variables = {v.name: v for v in parameterization.variables}

    expected_names = {
        "MoveTorsoAction_0.torso_state",
        "MoveTorsoAction_2.torso_state",
        "NavigateAction_1.keep_joint_states",
        "NavigateAction_1.target_location.position.x",
        "NavigateAction_1.target_location.position.y",
        "NavigateAction_1.target_location.position.z",
        "NavigateAction_1.target_location.rotation.w",
        "NavigateAction_1.target_location.rotation.x",
        "NavigateAction_1.target_location.rotation.y",
        "NavigateAction_1.target_location.rotation.z",
    }

    assert set(variables.keys()) == expected_names

    probabilistic_circuit = plan.parameterizer.create_fully_factorized_distribution()

    expected_distribution_names = expected_names - {
        "NavigateAction_1.target_location.header.sequence",
        "NavigateAction_1.target_location.header.frame_id.temp_collision_config.max_avoided_bodies",
        "NavigateAction_1.target_location.header.frame_id.collision_config.max_avoided_bodies",
    }
    assert {
        v.name for v in probabilistic_circuit.variables
    } == expected_distribution_names

    torso_1 = variables["MoveTorsoAction_0.torso_state"]
    torso_2 = variables["MoveTorsoAction_2.torso_state"]

    consistency_events = [
        SimpleEvent({torso_1: [state], torso_2: [state]}) for state in TorsoState
    ]
    restricted_distribution, _ = probabilistic_circuit.truncated(
        Event(*consistency_events)
    )
    restricted_distribution.normalize()

    pose_constraints = {
        variables["NavigateAction_1.target_location.position.x"]: 1.5,
        variables["NavigateAction_1.target_location.position.y"]: -2.0,
        variables["NavigateAction_1.target_location.rotation.x"]: 0.0,
        variables["NavigateAction_1.target_location.rotation.y"]: 0.0,
        variables["NavigateAction_1.target_location.rotation.z"]: 0.0,
        variables["NavigateAction_1.target_location.rotation.w"]: 1.0,
    }

    final_distribution, _ = restricted_distribution.conditional(pose_constraints)
    final_distribution.normalize()

    target_x, target_y = 1.5, -2.0
    nav_x = variables["NavigateAction_1.target_location.position.x"]
    nav_y = variables["NavigateAction_1.target_location.position.y"]

    for sample_values in final_distribution.sample(10):
        sample = dict(zip(final_distribution.variables, sample_values))
        assert sample[torso_1] == sample[torso_2]
        assert sample[nav_x] == target_x
        assert sample[nav_y] == target_y


def test_parameterize_pickup_navigate_place(immutable_model_world):
    """
    Test parameterization of a potential robot plan consisting of: PickUp - Navigate - Place.

    This test verifies:
    1. Parameterization of pick up, navigate, placing robot action plan
    2. Creating and sampling from a constrained distribution over the plan variables.
    """
    world, robot_view, context = immutable_model_world

    plan = SequentialPlan(
        context,
        PickUpActionDescription(None, None, None),
        NavigateActionDescription(None),
        PlaceActionDescription(None, None, None),
    )
    parameterization = plan.parameterize()
    variables = {v.name: v for v in parameterization.variables}

    expected_variables = {
        "NavigateAction_1.keep_joint_states",
        "NavigateAction_1.target_location.position.x",
        "NavigateAction_1.target_location.position.y",
        "NavigateAction_1.target_location.position.z",
        "NavigateAction_1.target_location.rotation.w",
        "NavigateAction_1.target_location.rotation.x",
        "NavigateAction_1.target_location.rotation.y",
        "NavigateAction_1.target_location.rotation.z",
        "PickUpAction_0.arm",
        "PickUpAction_0.grasp_description.approach_direction",
        "PickUpAction_0.grasp_description.manipulation_offset",
        "PickUpAction_0.grasp_description.manipulator.front_facing_axis.x",
        "PickUpAction_0.grasp_description.manipulator.front_facing_axis.y",
        "PickUpAction_0.grasp_description.manipulator.front_facing_axis.z",
        "PickUpAction_0.grasp_description.manipulator.front_facing_orientation.w",
        "PickUpAction_0.grasp_description.manipulator.front_facing_orientation.x",
        "PickUpAction_0.grasp_description.manipulator.front_facing_orientation.y",
        "PickUpAction_0.grasp_description.manipulator.front_facing_orientation.z",
        "PickUpAction_0.grasp_description.rotate_gripper",
        "PickUpAction_0.grasp_description.vertical_alignment",
        "PlaceAction_2.arm",
        "PlaceAction_2.target_location.position.x",
        "PlaceAction_2.target_location.position.y",
        "PlaceAction_2.target_location.position.z",
        "PlaceAction_2.target_location.rotation.w",
        "PlaceAction_2.target_location.rotation.x",
        "PlaceAction_2.target_location.rotation.y",
        "PlaceAction_2.target_location.rotation.z",
    }

    assert set(variables.keys()) == expected_variables

    probabilistic_distribution = (
        plan.parameterizer.create_fully_factorized_distribution()
    )

    expected_distribution = expected_variables - {
        "NavigateAction_1.target_location.header.sequence",
        "PlaceAction_2.target_location.header.sequence",
    }
    assert {
        v.name for v in probabilistic_distribution.variables
    } == expected_distribution

    arm_pickup = variables["PickUpAction_0.arm"]
    arm_place = variables["PlaceAction_2.arm"]

    arm_consistency_events = [
        SimpleEvent({arm_pickup: [arm], arm_place: [arm]}) for arm in Arms
    ]
    restricted_dist, _ = probabilistic_distribution.truncated(
        Event(*arm_consistency_events)
    )
    restricted_dist.normalize()

    nav_target_x = 2.0
    nav_target_y = 3.0
    pose_constraints = {
        variables["NavigateAction_1.target_location.position.x"]: nav_target_x,
        variables["NavigateAction_1.target_location.position.y"]: nav_target_y,
    }

    final_distribution, _ = restricted_dist.conditional(pose_constraints)
    final_distribution.normalize()

    v_nav_x = variables["NavigateAction_1.target_location.position.x"]
    v_nav_y = variables["NavigateAction_1.target_location.position.y"]

    for sample_values in final_distribution.sample(10):
        sample = dict(zip(final_distribution.variables, sample_values))
        assert sample[arm_pickup] == sample[arm_place]
        assert sample[v_nav_x] == nav_target_x
        assert sample[v_nav_y] == nav_target_y
