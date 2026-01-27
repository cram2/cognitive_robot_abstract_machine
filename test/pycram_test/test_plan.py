import os
import time
import pytest

from pycram.datastructures.pose import PyCramPose, PyCramQuaternion, PyCramVector3, Header
from random_events.product_algebra import SimpleEvent, Event
from random_events.variable import Integer
from krrood.class_diagrams.parameterizer import Parameterizer
from semantic_digital_twin.adapters.urdf import URDFParser

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import TaskStatus
from pycram.robot_plans import *
from pycram.language import SequentialPlan, ParallelPlan, CodeNode
from pycram.plan import PlanNode, Plan
from pycram.process_module import simulated_robot


@pytest.fixture
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
    assert node in plan.nodes
    assert node2 in plan.nodes
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
        MoveTorsoActionDescription(None),
        NavigateActionDescription(None),
        MoveTorsoActionDescription(None),
    )

    plan_classes = [
        MoveTorsoAction, NavigateAction, PyCramPose, PyCramVector3, PyCramQuaternion, Header,
        PoseStamped
    ]

    variables = sp.parameterize_plan(classes=plan_classes)
    variables_map = {v.name: v for v in variables}

    probabilistic_circuit = Parameterizer().create_fully_factorized_distribution(variables)

    torso_1 = variables_map["MoveTorsoAction_0.torso_state"]
    torso_2 = variables_map["MoveTorsoAction_2.torso_state"]
    consistency_events = [SimpleEvent({torso_1: [state], torso_2: [state]}) for state in TorsoState]
    restricted_distribution, _ = probabilistic_circuit.truncated(Event(*consistency_events))
    restricted_distribution.normalize()

    navigate_action_constraints = {
        variables_map["NavigateAction_1.target_location.pose.position.z"]: 0,
        variables_map["NavigateAction_1.target_location.pose.orientation.x"]: 0,
        variables_map["NavigateAction_1.target_location.pose.orientation.y"]: 0,
        variables_map["NavigateAction_1.target_location.pose.orientation.z"]: 0,
        variables_map["NavigateAction_1.target_location.pose.orientation.w"]: 1,
    }
    final_distribution, _ = restricted_distribution.conditional(navigate_action_constraints)
    final_distribution.normalize()

    nav_x = variables_map["NavigateAction_1.target_location.pose.position.x"]
    nav_y = variables_map["NavigateAction_1.target_location.pose.position.y"]
    nav_z = next(
        v for v in final_distribution.variables
        if v.name == "NavigateAction_1.target_location.pose.position.z"
    )
    nav_ox_var = next(
        v for v in final_distribution.variables
        if v.name == "NavigateAction_1.target_location.pose.orientation.x"
    )
    nav_oy_var = next(
        v for v in final_distribution.variables
        if v.name == "NavigateAction_1.target_location.pose.orientation.y"
    )
    nav_oz_var = next(
        v for v in final_distribution.variables
        if v.name == "NavigateAction_1.target_location.pose.orientation.z"
    )
    nav_ow_var = next(
        v for v in final_distribution.variables
        if v.name == "NavigateAction_1.target_location.pose.orientation.w"
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

    plan_classes = [
        MoveTorsoAction, ParkArmsAction, PoseStamped, PyCramPose,
        PyCramVector3, PyCramQuaternion, Header
    ]

    variables = sp.parameterize_plan(classes=plan_classes)
    variables_map = {v.name: v for v in variables}

    # Ensure expected variable names exist
    assert "MoveTorsoAction_0.torso_state" in variables_map
    assert "ParkArmsAction_1.arm" in variables_map

    probabilistic_circuit = Parameterizer().create_fully_factorized_distribution(variables)

    arm_var = variables_map["ParkArmsAction_1.arm"]
    torso_var = variables_map["MoveTorsoAction_0.torso_state"]

    # Truncate distribution to force arm == Arms.BOTH
    restricted_dist, _ = probabilistic_circuit.truncated(Event(SimpleEvent({arm_var: [Arms.BOTH]})))
    restricted_dist.normalize()

    for sample_values in restricted_dist.sample(5):
        sample = dict(zip(restricted_dist.variables, sample_values))
        assert sample[arm_var] == Arms.BOTH
        assert torso_var in sample
        assert sample[torso_var] in list(TorsoState)
