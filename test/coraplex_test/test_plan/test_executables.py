"""
Tests for the motion state chart a ``GiskardExecutable`` owns (see
``coraplex/src/coraplex/plans/executables.py``).

The chart is created once while the plan is parsed and only extended afterwards: parsing
adds a goal per plan node and a task per motion, and ``prepare_for_execution`` adds the
nodes that terminate the chart, which depend on the execution type.
"""

import pytest

from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
    SelfCollisionAvoidance,
)
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import CancelMotion, EndMotion, Task
from giskardpy.motion_statechart.monitors.payload_monitors import (
    ThreadedPredicateMonitor,
)
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose

from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.datastructures.enums import ExecutionType
from coraplex.execution_environment import (
    ExecutionEnvironment,
    real_robot,
    simulated_robot,
)
from coraplex.plans.executables import GiskardExecutable
from coraplex.plans.factories import execute_single
from coraplex.robot_plans.actions.core.pick_up import ReachAction
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk


@pytest.fixture
def reach_action_executable(immutable_model_world):
    """
    A real, 2-motion ``GiskardExecutable`` with pre-/post-conditions, built the same way
    ``test_merge_motions`` in ``test_graph_parsing.py`` does.
    """
    world, view, context = immutable_model_world
    milk_connection = world.get_body_by_name("milk.stl").parent_connection
    milk_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        2, 1.5, 0.7, 0, 0, 0, reference_frame=milk_connection.parent
    )
    plan = execute_single(
        ReachAction(
            Pose.from_xyz_rpy(2, 1.5, 0.7, reference_frame=world.root),
            Arms.RIGHT,
            GraspDescription(
                ApproachDirection.FRONT,
                VerticalAlignment.NoAlignment,
                view.right_arm.end_effector,
            ),
            world.get_semantic_annotations_by_type(Milk)[0],
        ),
        context=context,
    )
    plan.notify()
    return plan.parse()


# %% the chart is created once and extended


def test_motion_state_chart_is_created_once(reach_action_executable):
    """
    The chart is a field populated during parsing, not a property rebuilt per access.
    """
    assert (
        reach_action_executable.motion_state_chart
        is reach_action_executable.motion_state_chart
    )


def test_parsing_populates_the_chart_with_the_motions(reach_action_executable):
    """
    Every task is in the chart before execution begins, below the executable's root
    goal.
    """
    tasks = list(reach_action_executable.motion_mappings.values())
    chart = reach_action_executable.motion_state_chart

    assert len(tasks) == 2
    assert reach_action_executable.root_node in chart.nodes
    for task in tasks:
        assert task in chart.nodes
        # A reach that frees its gripper carries its Cartesian goal alongside the
        # collision rules, so the mapped node is the pair rather than the goal itself.
        assert (
            len([node for node in task.nodes if isinstance(node, CartesianPose)]) == 1
        )


def test_parsing_mirrors_the_plan_tree_as_nested_goals(reach_action_executable):
    """
    The action's motions live in a goal below the executable's root goal rather than
    flat in the chart.
    """
    tasks = list(reach_action_executable.motion_mappings.values())
    root_goal = reach_action_executable.root_node

    assert isinstance(root_goal, Sequence)
    assert root_goal.parent_node is None
    for task in tasks:
        assert task.parent_node is not None
        assert task.parent_node.parent_node is root_goal


def test_parsing_does_not_terminate_the_chart(reach_action_executable):
    """
    The nodes that end the motion are added by ``prepare_for_execution``, because they
    depend on the execution type.
    """
    chart = reach_action_executable.motion_state_chart

    assert chart.get_nodes_by_type(EndMotion) == []
    assert chart.get_nodes_by_type(CancelMotion) == []


# %% execution-type dependent extension


def test_prepare_for_execution_adds_a_single_end_motion(reach_action_executable):
    with real_robot:
        reach_action_executable.prepare_for_execution()

    chart = reach_action_executable.motion_state_chart
    assert len(chart.get_nodes_by_type(EndMotion)) == 1


@pytest.mark.parametrize("execution_environment", [real_robot, simulated_robot])
def test_execution_does_not_add_condition_monitors(
    reach_action_executable, execution_environment
):
    """
    Conditions are carried on the executable but stay out of the chart, whichever
    execution type the chart is prepared for.
    """
    assert reach_action_executable.pre_condition_node
    assert reach_action_executable.post_condition_node

    with execution_environment:
        reach_action_executable.prepare_for_execution()

    chart = reach_action_executable.motion_state_chart
    assert chart.get_nodes_by_type(ThreadedPredicateMonitor) == []
    assert chart.get_nodes_by_type(CancelMotion) == []


# %% wiring conditions into a chart

# Nothing calls _add_condition_monitors while conditions are kept out of the chart. These
# tests keep its wiring covered for whenever it is switched back on.


def test_condition_monitors_bring_their_own_abort_paths(reach_action_executable):
    assert reach_action_executable.pre_condition_node
    assert reach_action_executable.post_condition_node

    reach_action_executable._add_condition_monitors(
        reach_action_executable.root_node.observation_variable
    )

    chart = reach_action_executable.motion_state_chart
    # pre- and post-condition monitors
    assert len(chart.get_nodes_by_type(ThreadedPredicateMonitor)) == 2
    # abort paths for pre- and post-condition failing
    assert len(chart.get_nodes_by_type(CancelMotion)) == 2


def test_pre_condition_monitor_gates_the_root_goal(reach_action_executable):
    """
    The pre-condition monitor gates the whole motion, so it starts the root goal rather
    than an individual task.
    """
    reach_action_executable._add_condition_monitors(
        reach_action_executable.root_node.observation_variable
    )

    chart = reach_action_executable.motion_state_chart
    [pre_monitor, _] = chart.get_nodes_by_type(ThreadedPredicateMonitor)
    root_goal = reach_action_executable.root_node

    assert pre_monitor.name == "pre_condition"
    assert root_goal.start_condition.free_variables() == [
        pre_monitor.observation_variable
    ]


# %% collision avoidance


def test_prepare_for_execution_avoids_the_robot_colliding_with_itself(
    reach_action_executable,
):
    """
    A run asking for collision avoidance must get both kinds: without the self-collision
    goal nothing stops the arm from moving through the robot's own body, since the
    robot's ``AvoidSelfCollisions`` rule only shapes the collision matrix and never
    becomes a constraint on its own.
    """
    with ExecutionEnvironment(ExecutionType.SIMULATED, collision_avoidance=True):
        reach_action_executable.prepare_for_execution()

    chart = reach_action_executable.motion_state_chart
    assert len(chart.get_nodes_by_type(ExternalCollisionAvoidance)) == 1
    assert len(chart.get_nodes_by_type(SelfCollisionAvoidance)) == 1


def test_prepare_for_execution_leaves_out_collision_avoidance_when_not_asked_for(
    reach_action_executable,
):
    """
    A run that does not ask for collision avoidance gets neither goal.
    """
    with ExecutionEnvironment(ExecutionType.SIMULATED, collision_avoidance=False):
        reach_action_executable.prepare_for_execution()

    chart = reach_action_executable.motion_state_chart
    assert chart.get_nodes_by_type(ExternalCollisionAvoidance) == []
    assert chart.get_nodes_by_type(SelfCollisionAvoidance) == []


# %% how long a motion may take


def test_the_tick_budget_is_not_class_state(reach_action_executable):
    """
    The budget is a policy of the run, carried by its context, so two runs in one
    process cannot be given different budgets by class state that outlives them.
    """
    assert not hasattr(GiskardExecutable, "ticks_per_motion")
    assert reach_action_executable.context.ticks_per_motion
