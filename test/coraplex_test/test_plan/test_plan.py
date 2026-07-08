import os
import time

import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    ApproachDirection,
    VerticalAlignment,
    Arms,
)
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import simulated_robot
from coraplex.language import CodeNode
from coraplex.orm.ormatic_interface import *  # type: ignore
from coraplex.plans.condition_nodes import ConditionNode
from coraplex.plans.executables import GiskardExecutable
from coraplex.plans.factories import code, sequential, parallel, execute_single
from coraplex.plans.failures import EmptyUnderspecified
from coraplex.plans.plan import Plan
from coraplex.plans.plan_node import PlanNode, ActionNode
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import (
    variable_from,
    underspecified,
    variable,
)
from krrood.parametrization.model_registries import (
    FullyFactorizedRegistry,
)
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.orm.model import (
    Point3Mapping,
    QuaternionMapping,
    PoseMapping,
)
from semantic_digital_twin.robots.robot_parts import (
    EndEffector,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose


@pytest.fixture(scope="session")
def urdf_context():
    """Build a fresh URDF-based world and context for plan graph unit tests."""
    Plan.current_plan = None
    world = URDFParser.from_file(
        os.path.join(
            os.path.dirname(__file__),
            "../../../..",
            "..",
            "coraplex",
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


def test_neighbours_at_edges_return_none():
    """
    The outermost siblings have no neighbour on the outer side, which must be
    reported as ``None`` rather than raising ``IndexError``.
    """
    root = PlanNode()
    plan = Plan()
    plan.add_node(root)
    left_child = PlanNode()
    right_child = PlanNode()
    plan.add_edge(root, left_child)
    plan.add_edge(root, right_child)

    assert left_child.left_neighbour is None
    assert right_child.right_neighbour is None
    assert left_child.right_neighbour is right_child
    assert right_child.left_neighbour is left_child


def test_simplify_keeps_designators_with_different_parameters():
    """
    ``DesignatorNode.simplify`` may only merge a child whose designator has the
    same type *and* the same parameters; differing parameters must be preserved.
    """
    plan = Plan()
    parent = ActionNode(designator=MoveTorsoAction(TorsoState.HIGH))
    different_child = ActionNode(designator=MoveTorsoAction(TorsoState.LOW))
    plan.add_node(parent)
    plan.add_edge(parent, different_child)

    parent.simplify()

    assert different_child in parent.children

    equal_child = ActionNode(designator=MoveTorsoAction(TorsoState.HIGH))
    plan.add_edge(parent, equal_child)
    parent.simplify()

    assert equal_child not in parent.children


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


def test_path_after_node_removal():
    """
    Removing a node leaves a hole in the rustworkx index space, so a remaining
    node can end up with an index that is no longer smaller than the node count.
    ``path``/``depth`` must keep working in that case (they previously relied on
    ``rx.all_shortest_paths``, which panics on non-contiguous indices).
    """

    class _Node(PlanNode):
        def notify(self):
            pass

    plan = Plan()
    root = _Node()
    # A removable leaf that takes a low index, so removing it leaves a hole
    # below the index of the deeper chain nodes.
    removable = _Node()
    n2 = _Node()
    n3 = _Node()
    n4 = _Node()

    plan.add_edge(root, removable)
    plan.add_edge(root, n2)
    plan.add_edge(n2, n3)
    plan.add_edge(n3, n4)

    plan.remove_node(removable)

    # The deepest node's index now exceeds the remaining node count.
    assert n4.index >= len(plan.all_nodes)

    assert n4.path == [n3, n2, root]
    assert n4.depth == 3
    assert root.path == []
    assert root.depth == 0


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

    act1 = MoveTorsoAction(TorsoState.HIGH)

    act3 = MoveTorsoAction(TorsoState.LOW)

    plan = sequential([act1, act3], context=context).plan

    plan.root.children[1].interrupt()

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


def _torso_position(world):
    return world.state[
        world.get_degree_of_freedom_by_name("torso_lift_joint").id
    ].position


def test_sequence_runs_all_motions_without_interrupt(immutable_model_world):
    """
    Control for the interrupt tests: without an interrupt every motion in the
    sequence is executed, so the torso ends at the target of the *last* motion.
    The robot starts in the LOW configuration, so a final HIGH motion proves the
    second motion actually ran.
    """
    world, robot_view, context = immutable_model_world

    plan = sequential(
        [MoveTorsoAction(TorsoState.LOW), MoveTorsoAction(TorsoState.HIGH)],
        context=context,
    ).plan
    with simulated_robot:
        plan.perform()

    assert _torso_position(world) == pytest.approx(0.3, abs=0.05)


def test_interrupt_finishes_active_motion_and_skips_the_rest(immutable_model_world):
    """
    Interrupting a plan lets the currently active motion finish but skips every
    subsequent one ("finish active, skip rest").

    The first motion (HIGH) is the active one and must complete (torso reaches the
    HIGH target), while the trailing LOW and MID motions must be skipped - if any
    of them ran, the torso would move away from the HIGH target.
    """
    world, robot_view, context = immutable_model_world

    def interrupt(node: CodeNode):
        node.plan.root.interrupt()

    trigger = code(lambda: None)
    trigger.code = lambda: interrupt(trigger)

    plan = sequential(
        [
            MoveTorsoAction(TorsoState.HIGH),
            trigger,
            MoveTorsoAction(TorsoState.LOW),
            MoveTorsoAction(TorsoState.MID),
        ],
        context=context,
    ).plan
    with simulated_robot:
        plan.perform()

    # active motion finished (reached HIGH) and the trailing motions were skipped
    assert _torso_position(world) == pytest.approx(0.3, abs=0.05)


def test_pause_holds_active_motion_until_resumed(immutable_model_world):
    """
    Pausing a node holds the motion it originates from: while paused the active
    motion does not progress, and once resumed it runs to completion.

    A leading delay gives the controller time to pause the motion's subtree
    *before* the motion starts ticking; the controller then waits well past the
    point at which the motion would otherwise have finished and checks that the
    torso has not moved, before resuming it.
    """
    world, robot_view, context = immutable_model_world
    start_position = _torso_position(world)
    observed = {}

    motion_subplan = sequential(
        [code(lambda: time.sleep(1.0)), MoveTorsoAction(TorsoState.HIGH)]
    )

    def control():
        # pause well before the (delayed) motion starts ticking
        time.sleep(0.5)
        motion_subplan.pause()
        # give the motion ample time to run, were it not paused
        time.sleep(2.0)
        observed["while_paused"] = _torso_position(world)
        motion_subplan.resume()

    controller = code(lambda: None)
    controller.code = control

    plan = parallel([controller, motion_subplan], context=context).plan
    with simulated_robot:
        plan.perform()

    # while paused (and past when it would have finished) the motion did not move
    assert observed["while_paused"] == pytest.approx(start_position, abs=0.05)
    # after resume the motion completed
    assert _torso_position(world) == pytest.approx(0.3, abs=0.1)


def test_algebra_sequential_plan(apartment_world_pr2_copy_with_context):
    """
    Parameterize a SequentialPlan using krrood parameterizer, create a fully-factorized distribution and
    assert the correctness of sampled values after conditioning and truncation.
    """
    world, robot_view, context = apartment_world_pr2_copy_with_context
    context.evaluate_conditions = False

    target_location = underspecified(PoseMapping.from_point_mapping_quaternion_mapping)(
        position=underspecified(Point3Mapping)(
            x=..., y=..., z=0.0, reference_frame=None
        ),
        orientation=QuaternionMapping(x=0, y=0, z=0, w=1, reference_frame=None),
        reference_frame=variable_from([robot_view.root]),
    )

    navigate_action = underspecified(NavigateAction)(
        target_location=target_location,
    )
    # navigate_action.resolve()

    context.query_backend = ProbabilisticBackend(
        model_registry=FullyFactorizedRegistry()
    )

    # resolved_navigate = next(pm_backend.evaluate(navigate_action))
    plan = sequential([MoveTorsoAction(TorsoState.LOW), navigate_action], context).plan

    with simulated_robot:
        plan.perform()

    assert isinstance(plan.root.children[1].children[0].designator, NavigateAction)
    assert len(plan.root.children[1].children) == 1


def test_parameterization_of_pick_up(apartment_world_pr2_copy_with_context):
    world, robot_view, context = apartment_world_pr2_copy_with_context
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
            end_effector=variable(EndEffector, world.semantic_annotations),
        ),
    )

    parameters = UnderspecifiedParameters(pick_up_description)

    [end_effector_offset] = [
        v
        for v in parameters.variables.values()
        if v.name.endswith("manipulation_offset")
    ]

    assert (
        parameters.conditioning_assignments_from_literal_values[end_effector_offset]
        == 0.05
    )

    context.query_backend = ProbabilisticBackend(
        model_registry=FullyFactorizedRegistry()
    )

    plan = execute_single(pick_up_description, context)

    with simulated_robot:
        try:
            plan.perform()
        except EmptyUnderspecified:
            pass


def test_conditions_reference_surviving_action_node_after_merge(immutable_model_world):
    """
    Expanding an action mounts a fresh action node whose conditions reference it,
    and simplification merges that node into the equivalent node already in the
    plan. After the merge every condition must reference the surviving node, not
    the discarded one, otherwise the dangling node leaks into serialization.
    """
    world, robot_view, context = immutable_model_world

    plan = sequential(
        [MoveTorsoAction(TorsoState.HIGH)],
        context=context,
    ).plan
    with simulated_robot:
        plan.perform()

    live_node_indices = {node.index for node in [plan.root, *plan.root.descendants]}
    condition_nodes = [
        node for node in plan.root.descendants if isinstance(node, ConditionNode)
    ]
    assert condition_nodes
    for condition_node in condition_nodes:
        assert condition_node.action_node.index in live_node_indices


def test_motion_order_pick_up(mutable_model_world):
    world, robot_view, context = mutable_model_world

    grasp_description = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        robot_view.left_arm.end_effector,
    )

    milk_body = world.get_body_by_name("milk.stl")
    milk_body.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1, -2, 0.6, reference_frame=world.root
    )
    robot_view.root.parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(
            0.3, -2.4, 0, reference_frame=world.root
        )
    )
    world.notify_state_change()

    root = sequential(
        [
            PickUpAction(
                world.get_body_by_name("milk.stl"), Arms.LEFT, grasp_description
            ),
        ],
        context,
    )

    all_motions = []

    def exec_wrapper(giskard_executable):
        all_motions.extend(giskard_executable.motion_mappings.values())

    original_execute = GiskardExecutable.execute
    GiskardExecutable.execute = exec_wrapper
    try:
        with simulated_robot:
            root.perform()
    finally:
        GiskardExecutable.execute = original_execute

    motion_names = [motion.name for motion in all_motions]

    assert motion_names == [
        "OpenGripper",
        "MoveTCP",
        "MoveTCP",
        "CloseGripper",
        "MoveTCP",
    ]


def test_motion_order_place(mutable_model_world):
    world, robot_view, context = mutable_model_world

    milk_body = world.get_body_by_name("milk.stl")
    milk_body.parent_connection.origin = world.get_body_by_name(
        "l_gripper_tool_frame"
    ).global_pose

    with world.modify_world():

        world.move_branch_with_fixed_connection(
            world.get_body_by_name("milk.stl"),
            world.get_body_by_name("l_gripper_tool_frame"),
        )

    robot_view.root.parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(
            0.3, -2.4, 0, reference_frame=world.root
        )
    )
    world.notify_state_change()

    root = sequential(
        [
            PlaceAction(
                world.get_body_by_name("milk.stl"),
                Pose.from_xyz_rpy(0.8, -1.9, 0.7, reference_frame=world.root),
                Arms.LEFT,
            ),
        ],
        context,
    )

    all_motions = []

    def exec_wrapper(giskard_executable):
        all_motions.extend(giskard_executable.motion_mappings.values())

    original_execute = GiskardExecutable.execute
    GiskardExecutable.execute = exec_wrapper
    try:
        with simulated_robot:
            root.perform()
    finally:
        GiskardExecutable.execute = original_execute

    motion_names = [motion.name for motion in all_motions]

    assert motion_names == [
        "MoveTCP",
        "MoveTCP",
        "OpenGripper",
        "MoveTCP",
    ]


def test_node_expansion(immutable_model_world):
    world, view, context = immutable_model_world

    plan = sequential(
        [
            PickUpAction(
                object_designator=world.get_body_by_name("milk.stl"),
                arm=Arms.RIGHT,
                grasp_description=GraspDescription(
                    ApproachDirection.FRONT,
                    vertical_alignment=VerticalAlignment.NoAlignment,
                    end_effector=view.right_arm.end_effector,
                ),
            )
        ],
        context=context,
    )

    pick_node = plan.children[0]
    pick_node.notify()

    expanded_children = pick_node.children
    assert len(expanded_children) == 3
    assert len(expanded_children[1].children) == 5


def test_expand_move_torso(immutable_model_world):
    world, view, context = immutable_model_world
    plan = sequential([MoveTorsoAction(TorsoState.HIGH)], context=context)

    plan.notify()

    node = plan.plan.get_nodes_by_designator_type(MoveTorsoAction)[0]

    assert len(node.children) == 3


def test_context_back_reference(immutable_model_world):
    world, view, context = immutable_model_world

    plan = sequential(
        [
            MoveTorsoAction(TorsoState.HIGH),
            PickUpAction(
                world.get_body_by_name("milk.stl"),
                Arms.RIGHT,
                GraspDescription(
                    ApproachDirection.FRONT,
                    VerticalAlignment.NoAlignment,
                    view.right_arm.end_effector,
                ),
            ),
        ],
        context=context,
    )

    plan.notify()

    assert plan.plan.context == context


def test_action_nodes_unequal(immutable_model_world):
    world, view, context = immutable_model_world

    plan = sequential(
        [
            ParkArmsAction(Arms.LEFT),
            PickUpAction(
                world.get_body_by_name("milk.stl"),
                Arms.LEFT,
                GraspDescription(
                    ApproachDirection.FRONT,
                    VerticalAlignment.NoAlignment,
                    view.right_arm.end_effector,
                ),
            ),
        ],
        context=context,
    )

    park_node = plan.children[0]
    pick_node = plan.children[1]

    assert not park_node == pick_node
