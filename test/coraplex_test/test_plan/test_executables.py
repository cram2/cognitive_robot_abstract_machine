"""
Tests for the REAL/SIMULATED branch of ``GiskardExecutable.motion_state_chart``
(see ``coraplex/src/coraplex/plans/executables.py``).

On the real robot, tasks are wrapped in a single ``Sequence`` + ``EndMotion``; in
simulation, tasks are added individually and get pause/interrupt monitors and
pre-/post-condition monitors wired in.
"""

import pytest

from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import CancelMotion, EndMotion
from giskardpy.motion_statechart.monitors.payload_monitors import (
    ThreadedPredicateMonitor,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose

from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import real_robot, simulated_robot
from coraplex.plans.condition_nodes import PlanNodeStatusMonitor
from coraplex.plans.factories import execute_single
from coraplex.robot_plans.actions.core.pick_up import ReachAction


@pytest.fixture
def reach_action_executable(immutable_model_world):
    """A real, 2-motion ``GiskardExecutable`` with pre-/post-conditions, built the
    same way ``test_merge_motions`` in ``test_graph_parsing.py`` does."""
    world, view, context = immutable_model_world
    world.get_body_by_name("milk.stl").parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(2, 1.5, 0.7, 0, 0, 0)
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
            world.get_body_by_name("milk.stl"),
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
    # pre- and post-condition monitors
    assert len(chart.get_nodes_by_type(ThreadedPredicateMonitor)) == 2
    # abort paths for pre- and post-condition failing
    assert len(chart.get_nodes_by_type(CancelMotion)) == 2
