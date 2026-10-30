"""
Tests for how a plan expands into the nodes of its statechart.
"""

import numpy as np
import pytest
from typing_extensions import List

from coraplex.datastructures.enums import DetectionTechnique, PerceptionSource
from coraplex.perception import PerceptionQuery, PerceptionTask
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.misc import DetectAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction, ReachAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from cramph.composites import (
    CancelledWhenTrue,
    Parallel,
    PausedUntilTrue,
    PausedWhileTrue,
    Sequence,
    TryAll,
    TryInOrder,
)
from cramph.data_types import LifeCycleValues
from cramph.node import CancelStatechart, StatechartNode
from cramph.nodes_for_testing import ConstFalseNode
from cramph.statechart import Statechart
from cramph.world_modification_nodes import MoveBranch
from giskardpy.motion_statechart.goals.gripper import MoveGripper
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianPosition,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Point3, Pose
from semantic_digital_twin.world_description.geometry import VolumetricBoundingBox

from ..conftest import expand, motion_nodes_of, tool_center_point_goal
from cramph.exceptions import PlanCancelled
from ...plan_running import (
    context_of,
    run_plan,
    simulated_executor,
    statechart_of,
)
from cramph.context import ContextExtension

# %% helpers


def _compile(plan: StatechartNode, extensions: list[ContextExtension]) -> Statechart:
    """
    Build the statechart that executing `plan` runs, and compile it without ticking.

    Compiling is what validates the scopes of every transition condition.

    :return: The compiled statechart.
    """
    executor = simulated_executor(extensions)
    statechart = statechart_of(executor, plan)
    executor.prepare(statechart)
    statechart.compiler.compile()
    return statechart


def _nodes_of_type(root: StatechartNode, node_type: type) -> list[StatechartNode]:
    """
    :return: Every node of `node_type` at or below `root`, in depth first order.
    """
    return [node for node in [root, *root.descendants] if isinstance(node, node_type)]


# %% an action expands into its motions


def test_an_action_expands_into_its_motion(pr2_apartment_context) -> None:
    world, view, extensions = pr2_apartment_context

    root = expand(MoveTorsoAction(torso_state=TorsoState.HIGH), extensions)

    assert [type(node) for node in _nodes_of_type(root, JointPositionList)] == [
        JointPositionList
    ]


# %% a sequence holds its steps


def test_a_sequence_holds_each_action_with_its_own_motion(
    pr2_apartment_context,
) -> None:
    world, view, extensions = pr2_apartment_context

    root = expand(
        Sequence(
            nodes=[
                MoveTorsoAction(torso_state=TorsoState.LOW),
                MoveTorsoAction(torso_state=TorsoState.HIGH),
            ]
        ),
        extensions,
    )

    assert [type(step) for step in root.nodes] == [MoveTorsoAction, MoveTorsoAction]
    assert [len(_nodes_of_type(step, JointPositionList)) for step in root.nodes] == [
        1,
        1,
    ]


# %% monitored subtrees


def test_pause_monitor_pauses_the_children_goal(
    pr2_apartment_context, rclpy_node
) -> None:
    """
    The monitor and the children's goal are siblings inside the monitored goal, which is
    what makes the pause condition legal: it may only reference a sibling.
    """
    world, view, extensions = pr2_apartment_context
    monitor = ConstFalseNode(name="never")

    plan = PausedWhileTrue(
        monitor=monitor,
        monitored_node=Sequence(nodes=[MoveTorsoAction(torso_state=TorsoState.HIGH)]),
    )
    _compile(plan, extensions)

    monitored_goal = plan
    assert type(monitored_goal) is PausedWhileTrue
    assert monitored_goal.nodes == [monitor, monitored_goal.monitored_node]
    assert monitored_goal.monitored_node.pause_condition.free_variables() == [
        monitor.observes_true
    ]


def test_pause_until_monitor_pauses_the_children_goal(
    pr2_apartment_context, rclpy_node
) -> None:
    """
    The children's goal is paused on the negated monitor observation, so it is held
    until the monitor turns True rather than while it is True.
    """
    world, view, extensions = pr2_apartment_context
    monitor = ConstFalseNode(name="never")

    plan = PausedUntilTrue(
        monitor=monitor,
        monitored_node=Sequence(nodes=[MoveTorsoAction(torso_state=TorsoState.HIGH)]),
    )
    _compile(plan, extensions)

    monitored_goal = plan
    assert type(monitored_goal) is PausedUntilTrue
    assert monitored_goal.nodes == [monitor, monitored_goal.monitored_node]
    assert monitored_goal.monitored_node.pause_condition.free_variables() == [
        monitor.observes_true
    ]


def test_cancel_monitor_ends_the_children_goal(
    pr2_apartment_context, rclpy_node
) -> None:
    world, view, extensions = pr2_apartment_context
    monitor = ConstFalseNode(name="never")

    plan = CancelledWhenTrue(
        monitor=monitor,
        monitored_node=Sequence(nodes=[MoveTorsoAction(torso_state=TorsoState.HIGH)]),
        exception=PlanCancelled(monitor=monitor),
    )
    _compile(plan, extensions)

    monitored_goal = plan
    assert type(monitored_goal) is CancelledWhenTrue
    assert monitored_goal.nodes[:2] == [monitor, monitored_goal.monitored_node]
    # The children's goal already ends itself once it succeeds, so the monitor firing is
    # a reason to interrupt it on top of that. It is read through its last observation,
    # which outlasts a monitor that ends itself on firing.
    assert monitor.last_observed_true in (
        monitored_goal.monitored_node.interrupt_condition.free_variables()
    )


def test_cancel_monitor_ends_the_motion_when_the_monitor_fires(
    pr2_apartment_context, rclpy_node
) -> None:
    """
    The monitored goal holds a node that ends the motion, so giving up on the subtree
    gives up on the plan rather than leaving the rest of it waiting.
    """
    world, view, extensions = pr2_apartment_context
    monitor = ConstFalseNode(name="never")

    plan = CancelledWhenTrue(
        monitor=monitor,
        monitored_node=Sequence(nodes=[MoveTorsoAction(torso_state=TorsoState.HIGH)]),
        exception=PlanCancelled(monitor=monitor),
    )
    _compile(plan, extensions)

    monitored_goal = plan
    [cancelled] = [
        node for node in monitored_goal.nodes if isinstance(node, CancelStatechart)
    ]
    assert cancelled.exception == monitored_goal.exception
    assert cancelled.start_condition.free_variables() == [monitor.last_observed_true]


def test_monitored_subtree_nested_in_a_sequence_compiles(
    pr2_apartment_context, rclpy_node
) -> None:
    """
    A monitored subtree is a node like any other in the surrounding sequence.

    Compiling is the real assertion: it runs the condition scope validation that this
    structure exists to satisfy.
    """
    world, view, extensions = pr2_apartment_context

    never = ConstFalseNode(name="never")

    plan = Sequence(
        nodes=[
            MoveTorsoAction(torso_state=TorsoState.LOW),
            CancelledWhenTrue(
                monitor=never,
                monitored_node=Sequence(
                    nodes=[MoveTorsoAction(torso_state=TorsoState.HIGH)]
                ),
                exception=PlanCancelled(monitor=never),
            ),
        ]
    )
    statechart = _compile(plan, extensions)

    assert len(statechart.get_nodes_by_type(CancelledWhenTrue)) == 1


# %% running actions


def test_a_reach_runs_to_its_target(pr2_apartment_context, rclpy_node) -> None:
    world, view, extensions = pr2_apartment_context

    milk_connection = world.get_body_by_name("milk.stl").parent_connection
    milk_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        2, 1.5, 0.7, 0, 0, 0, reference_frame=milk_connection.parent
    )
    reach = ReachAction(
        grasp=GraspCandidate.from_body_origin(
            world.get_semantic_annotations_by_type(Milk)[0]
        ),
        arm=view.right_arm,
    )

    run_plan(reach, extensions)

    assert reach.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_a_pick_up_moves_the_object_to_the_gripper_between_closing_and_lifting(
    pr2_apartment_context,
) -> None:
    """
    The object only follows the gripper once it belongs to it, and has to before the
    lift, so the branch moves between the two inside the pick-up's own statechart.
    """
    world, view, extensions = pr2_apartment_context

    root = expand(
        PickUpAction(
            grasp=world.get_semantic_annotations_by_type(Milk)[0].grasp_candidates()[0],
            arm=view.right_arm,
        ),
        extensions,
    )

    steps = [
        type(node)
        for node in _nodes_of_type(
            root, (MoveGripper, MoveBranch, CartesianPose, CartesianPosition)
        )
    ]
    # The gripper closes while a pose goal, which brings its own position goal, keeps
    # it at the grasp; the lift is such a pose goal as well.
    assert steps[-6:] == [
        MoveGripper,
        CartesianPose,
        CartesianPosition,
        MoveBranch,
        CartesianPose,
        CartesianPosition,
    ]


def test_a_transport_runs_with_its_underspecified_steps(
    pr2_apartment_context, rclpy_node
) -> None:
    world, view, extensions = pr2_apartment_context

    plan = Sequence(
        nodes=[
            MoveTorsoAction(torso_state=TorsoState.HIGH),
            ParkArmsAction(arms=view.all_arms),
            TransportAction.from_graspable_by_closest_grasps(
                world.get_semantic_annotations_by_type(Milk)[0],
                Pose.from_xyz_rpy(2.37, 2.5, 1.05, reference_frame=world.root),
                view.right_arm,
                context_of(extensions),
            ),
        ]
    )

    run_plan(plan, extensions)

    assert plan.life_cycle_state == LifeCycleValues.SUCCEEDED


# %% perception


def test_perceiving_runs_between_the_motions_around_it(pr2_apartment_context) -> None:
    """
    Perception is a step like any other, so it runs in the same statechart as the
    motions around it, in the order the plan gives.
    """
    world, view, extensions = pr2_apartment_context
    query = PerceptionQuery(
        Milk,
        VolumetricBoundingBox(
            origin=HomogeneousTransformationMatrix(reference_frame=world.root),
            min_x=-10,
            min_y=-10,
            min_z=-10,
            max_x=10,
            max_y=10,
            max_z=10,
        ),
        view,
        world,
    )

    root = expand(
        Sequence(
            nodes=[
                tool_center_point_goal(view, view.left_arm),
                PerceptionTask(query=query, answered_by=PerceptionSource.WORLD_MODEL),
                tool_center_point_goal(view, view.right_arm),
            ]
        ),
        extensions,
    )

    assert [
        type(node) for node in _nodes_of_type(root, (CartesianPose, PerceptionTask))
    ] == [
        CartesianPose,
        PerceptionTask,
        CartesianPose,
    ]


def test_a_detect_action_expands_into_a_perception_task(pr2_apartment_context) -> None:
    world, view, extensions = pr2_apartment_context

    root = expand(
        DetectAction(technique=DetectionTechnique.TYPES, object_sem_annotation=Milk),
        extensions,
    )

    assert [type(node) for node in _nodes_of_type(root, PerceptionTask)] == [
        PerceptionTask
    ]


# %% perceiving before the grasp


def detect_actions_of(
    plan: StatechartNode, extensions: list[ContextExtension]
) -> list[DetectAction]:
    """
    :param plan: The plan to search.
    :param extensions: The extensions the plan is expanded in.
    :return: The detections the plan performs, in no particular order.
    """
    return [
        node
        for node in motion_nodes_of(plan, extensions)
        if isinstance(node, DetectAction)
    ]


def reach_action(milk: Milk, view) -> ReachAction:
    """
    :param milk: The object the reach is aimed at.
    :param view: The robot reaching for it.
    :return: A reach at the object's own frame.
    """
    return ReachAction(grasp=GraspCandidate.from_body_origin(milk), arm=view.right_arm)


def test_a_reach_does_not_perceive_by_default(pr2_apartment_context) -> None:
    """
    A reach acts on the pose the world already holds, so it must not spend a detection
    the caller did not ask for.
    """
    world, view, extensions = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]

    plan = reach_action(milk, view)

    assert detect_actions_of(plan, extensions) == []


# %% expansion-time pose capture


def test_pick_up_motions_follow_the_object_moved_after_expansion(
    pr2_apartment_context,
) -> None:
    """
    A pick-up expands when its plan starts, before the first motion runs, so one that
    captured the object's pose in world coordinates could never act on a pose corrected
    in between (for example by a detection).

    Keeping the motion targets in the object's own frame is what lets them follow it.
    """
    world, view, extensions = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    milk_body = milk.root

    plan = PickUpAction(grasp=milk.grasp_candidates()[0], arm=view.right_arm)
    targets = [
        node.goal_pose
        for node in motion_nodes_of(plan, extensions)
        if isinstance(node, CartesianPose)
    ]
    positions_before = [
        world.transform(target, world.root).position.to_np().flatten()[:3]
        for target in targets
    ]

    displacement = np.array([0.25, -0.4, 0.1])
    milk_body.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        *(milk_body.global_pose.position.to_np().flatten()[:3] + displacement),
        reference_frame=world.root,
    )

    assert targets
    assert all(target.reference_frame is milk_body for target in targets)
    for target, position_before in zip(targets, positions_before):
        np.testing.assert_allclose(
            world.transform(target, world.root).position.to_np().flatten()[:3],
            position_before + displacement,
            atol=1e-9,
        )
