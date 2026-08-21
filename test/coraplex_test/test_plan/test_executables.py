"""
Tests for the REAL/SIMULATED branch of ``GiskardExecutable.motion_state_chart`` (see
``coraplex/src/coraplex/plans/executables.py``).

On the real robot, tasks are wrapped in a single ``Sequence`` + ``EndMotion``; in
simulation, tasks are added individually and get pause/interrupt monitors and pre-/post-
condition monitors wired in.
"""

from copy import deepcopy

import pytest


from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
    UpdateTemporaryCollisionRules,
)
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import CancelMotion, EndMotion
from giskardpy.motion_statechart.monitors.payload_monitors import (
    ThreadedPredicateMonitor,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.spatial_types.spatial_types import Pose

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import real_robot, simulated_robot
from coraplex.plans.condition_nodes import PlanNodeStatusMonitor
from coraplex.plans.executables import GiskardExecutable
from coraplex.plans.factories import execute_single
from coraplex.robot_plans.actions.core.pick_up import GraspingAction, ReachAction
from coraplex.robot_plans.motions.gripper import MoveGripperMotion
from coraplex.view_manager import ViewManager


@pytest.fixture
def reach_action_executable(immutable_model_world):
    """
    A real, 2-motion ``GiskardExecutable`` with pre-/post-conditions, built the same way
    ``test_merge_motions`` in ``test_graph_parsing.py`` does.
    """
    world, view, context = immutable_model_world
    milk_connection = world.get_body_by_name("milk.stl").parent_connection
    # Where a body stands is model rather than state, so the fixture around this one puts
    # it back itself; left moved, the milk stands in the way of every test that follows.
    origin_before = milk_connection.origin
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
            world.get_body_by_name("milk.stl"),
        ),
        context=context,
    )
    plan.notify()
    yield plan.parse()
    milk_connection.origin = origin_before


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


def test_only_the_last_motion_may_keep_its_goal(immutable_model_world):
    """
    A motion hands the robot over to the motion after it once its goal is observed.

    Only the last one hands it to nobody -- and the chart keeps ticking until the world
    settles -- so only there may a motion keep its goal in force.
    """
    world, view, context = immutable_model_world
    plan = execute_single(
        GraspingAction(
            world.get_body_by_name("milk.stl"),
            Arms.RIGHT,
            GraspDescription(
                ApproachDirection.FRONT,
                VerticalAlignment.NoAlignment,
                view.right_arm.end_effector,
            ),
        ),
        context=context,
    )
    plan.notify()
    executable = plan.parse()

    with simulated_robot:
        executable.motion_state_chart

    motions = list(executable.motion_mappings.keys())
    tasks = list(executable.motion_mappings.values())
    assert motions[-1].designator.holds_its_goal_until_the_motion_ends
    assert str(tasks[-1].end_condition) != str(tasks[-1].observation_variable)
    for task in tasks[:-1]:
        assert str(task.end_condition) == str(task.observation_variable)


def test_collision_avoidance_cancels_the_motion_it_guards(immutable_model_world):
    """
    A violated collision ends the motion it happened in, so a plan cannot carry on
    through what it was told to keep clear of, and whatever contact a motion does need
    has to be declared rather than tolerated everywhere.
    """
    world, view, context = immutable_model_world
    plan = execute_single(
        MoveGripperMotion(GripperState.OPEN, Arms.RIGHT), context=context
    )
    plan.notify()
    executable = plan.parse()

    with simulated_robot(collision_avoidance=True):
        chart = executable.motion_state_chart

    avoidance = chart.get_nodes_by_type(ExternalCollisionAvoidance)
    assert len(avoidance) == 1
    assert avoidance[0].cancel_if_collision_violated


def test_configuring_an_environment_leaves_the_shared_one_alone():
    """
    The environments are module-level and shared by everyone who imports them, so one
    plan asking for collision avoidance must not switch it on for every plan that runs
    after it.
    """
    with simulated_robot(collision_avoidance=True):
        assert GiskardExecutable.collision_avoidance

    with simulated_robot:
        assert not GiskardExecutable.collision_avoidance


def test_a_chart_is_built_from_the_world_the_motion_will_run_in(immutable_model_world):
    """
    Everything before a motion has run by the time it starts, so its chart has to be
    built then rather than when the plan was parsed.

    Built at parse time, a hand that picks something up along the way is still described
    as empty, and what it carries is left out of the contacts the hand is allowed to
    make.
    """
    world, view, context = immutable_model_world
    test_world = deepcopy(world)
    test_context = Context.from_world(test_world)
    milk = test_world.get_body_by_name("milk.stl")
    tool_frame = ViewManager.get_end_effector_view(
        Arms.RIGHT, test_context.robot
    ).tool_frame
    plan = execute_single(
        MoveGripperMotion(GripperState.OPEN, Arms.RIGHT, allow_gripper_collision=True),
        context=test_context,
    )
    plan.notify()
    executable = plan.parse()

    with test_world.modify_world():
        test_world.remove_connection(milk.parent_connection)
        test_world.add_connection(FixedConnection(parent=tool_frame, child=milk))

    rules_node = next(
        node
        for task in executable.motion_mappings.values()
        for node in task.nodes
        if isinstance(node, UpdateTemporaryCollisionRules)
    )

    assert milk in rules_node.temporary_rules[0].body_group_b
