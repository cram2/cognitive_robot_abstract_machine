"""
Tests for the REAL/SIMULATED branch of ``GiskardExecutable.motion_state_chart`` (see
``coraplex/src/coraplex/plans/executables.py``).

On the real robot, tasks are wrapped in a single ``Sequence`` + ``EndMotion``; in
simulation, tasks are added individually and get pause/interrupt monitors and pre-/post-
condition monitors wired in.
"""

import pytest

from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
    SelfCollisionAvoidance,
)
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import CancelMotion, EndMotion
from giskardpy.motion_statechart.monitors.payload_monitors import (
    ThreadedPredicateMonitor,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose

from coraplex.datastructures.enums import Arms, ExecutionType
from coraplex.execution_environment import (
    ExecutionEnvironment,
    real_robot,
    simulated_robot,
)
from coraplex.plans.condition_nodes import PlanNodeStatusMonitor
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
            world.get_semantic_annotations_by_type(Milk)[0],
        ),
        context=context,
    )
    plan.notify()
    return plan.parse()


def test_motion_state_chart_simulated_execution_adds_tasks_directly(
    reach_action_executable,
):
    tasks = list(reach_action_executable.motion_mappings.values())

    with simulated_robot:
        chart = reach_action_executable.motion_state_chart

    assert chart.get_nodes_by_type(Sequence) == []
    for task in tasks:
        assert task in chart.nodes


def test_motion_state_chart_real_execution_wraps_tasks_in_sequence(
    reach_action_executable,
):
    tasks = list(reach_action_executable.motion_mappings.values())

    with real_robot:
        chart = reach_action_executable.motion_state_chart

    sequences = chart.get_nodes_by_type(Sequence)
    assert len(sequences) == 1
    assert sequences[0].nodes == tasks
    assert len(chart.get_nodes_by_type(EndMotion)) == 1
    # simulation-only machinery must not be present on the real-robot path
    for task in tasks:
        assert task not in chart.nodes


def test_motion_state_chart_simulated_execution_adds_condition_and_pause_interrupt_monitors(
    reach_action_executable,
):
    task_count = len(reach_action_executable.motion_mappings)
    assert reach_action_executable.pre_condition_node
    assert reach_action_executable.post_condition_node

    with simulated_robot:
        chart = reach_action_executable.motion_state_chart

    # one pause + one interrupt monitor per task
    assert len(chart.get_nodes_by_type(PlanNodeStatusMonitor)) == 2 * task_count


# %% collision avoidance


def test_motion_state_chart_avoids_the_robot_colliding_with_itself(
    reach_action_executable,
):
    """
    A run asking for collision avoidance must get both kinds: without the self-collision
    goal nothing stops the arm from moving through the robot's own body, since the
    robot's ``AvoidSelfCollisions`` rule only shapes the collision matrix and never
    becomes a constraint on its own.
    """
    with ExecutionEnvironment(ExecutionType.SIMULATED, collision_avoidance=True):
        chart = reach_action_executable.motion_state_chart

    assert len(chart.get_nodes_by_type(ExternalCollisionAvoidance)) == 1
    assert len(chart.get_nodes_by_type(SelfCollisionAvoidance)) == 1


def test_motion_state_chart_leaves_out_collision_avoidance_when_not_asked_for(
    reach_action_executable,
):
    """
    A run that does not ask for collision avoidance gets neither goal.
    """
    with ExecutionEnvironment(ExecutionType.SIMULATED, collision_avoidance=False):
        chart = reach_action_executable.motion_state_chart

    assert chart.get_nodes_by_type(ExternalCollisionAvoidance) == []
    assert chart.get_nodes_by_type(SelfCollisionAvoidance) == []
