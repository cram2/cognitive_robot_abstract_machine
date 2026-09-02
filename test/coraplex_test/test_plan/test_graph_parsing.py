from dataclasses import dataclass

import numpy as np
import pytest
from typing_extensions import List

from coraplex.datastructures.enums import (
    Arms,
    ApproachDirection,
    DetectionTechnique,
    VerticalAlignment,
)
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import simulated_robot
from coraplex.plans.attachment_nodes import ReAttachNode
from coraplex.perception import PerceptionQuery
from coraplex.plans.executables import (
    Executable,
    GiskardExecutable,
    MoveBranchExecutable,
)
from coraplex.plans.factories import (
    cancel_when,
    execute_single,
    pause_until,
    pause_while,
    repeat,
    sequential,
)
from coraplex.exceptions import PerceptionTargetMissing
from coraplex.plans.plan_node import (
    ActionNode,
    ExecutionBoundaryNode,
    MotionNode,
    PlanNode,
)
from coraplex.robot_plans import MoveToolCenterPointMotion
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.misc import DetectAction
from coraplex.robot_plans.actions.core.pick_up import ReachAction, PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from coraplex.robot_plans.motions.misc import DetectingMotion, PerceptionTask
from coraplex.language import (
    ParallelNode,
    SequentialNode,
    TryAllNode,
    TryInOrderNode,
)
from giskardpy.motion_statechart.monitors.templates import (
    PausedUntilTrue,
    PausedWhileTrue,
)
from coraplex.utils import split_list_by_type
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.templates import (
    Parallel,
    RepeatOnStall,
    Sequence, TryAll, TryInOrder, CancelledWhenTrue,
)
from giskardpy.motion_statechart.graph_node import CancelMotion
from giskardpy.motion_statechart.monitors.payload_monitors import CountNodeResets
from giskardpy.motion_statechart.nodes_for_testing.nodes_for_testing import (
    ConstFalseNode,
)
from giskardpy.ros_executor import Ros2Executor
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose, Point3
from semantic_digital_twin.world_description.geometry import VolumetricBoundingBox


def test_parse_simple_action(immutable_model_world):
    world, view, context = immutable_model_world

    plan = execute_single(MoveTorsoAction(TorsoState.HIGH), context=context)

    plan.notify()

    executable = plan.parse()

    assert type(executable) == GiskardExecutable
    assert executable.pre_condition_node
    assert executable.post_condition_node
    assert len(executable.motion_mappings) == 1
    assert type(list(executable.motion_mappings.values())[0]) == JointPositionList


# %% the chart mirrors the plan tree


def test_language_nodes_create_a_goal_of_their_template():
    """
    Each language node contributes a goal of the template it declares, which is what
    gives the motion state chart the plan's sequential/parallel/try semantics.
    """
    assert type(SequentialNode().create_goal()) is Sequence
    assert type(ParallelNode().create_goal()) is Parallel
    assert type(TryAllNode().create_goal()) is TryAll
    assert type(TryInOrderNode().create_goal()) is TryInOrder


def test_sequential_plan_nests_a_goal_per_plan_node(immutable_model_world):
    """
    Parsing a sequential plan builds a goal per language and action node, with the
    motions as tasks at the leaves, rather than one flat list of tasks.
    """
    world, view, context = immutable_model_world

    plan = sequential(
        [MoveTorsoAction(TorsoState.LOW), MoveTorsoAction(TorsoState.HIGH)],
        context=context,
    )
    plan.notify()
    executable = plan.parse()

    root_goal = executable.root_node
    assert type(root_goal) is Sequence
    assert root_goal.name == "SequentialNode"

    action_goals = root_goal.nodes
    assert len(action_goals) == 2
    assert [type(goal) for goal in action_goals] == [Sequence, Sequence]
    assert [goal.name for goal in action_goals] == ["ActionNode", "ActionNode"]

    tasks = list(executable.motion_mappings.values())
    assert [goal.nodes for goal in action_goals] == [[tasks[0]], [tasks[1]]]


# %% monitored subtrees


def _monitored_goal_of(executable):
    """
    :return: The single monitored goal below the executable's root goal.
    """
    [monitored_goal] = executable.root_node.nodes
    return monitored_goal


def _parse_and_compile(plan, world, context):
    """
    Parse `plan` and compile its motion state chart.

    Compiling is what expands the goals, so it is required before any condition wired by
    a template can be observed.
    """
    plan.notify()
    executable = plan.parse()
    with simulated_robot:
        executable.prepare_for_execution()
    executor = Ros2Executor(
        context=MotionStatechartContext(world=world), ros_node=context.ros_node
    )
    executor.compile(executable.motion_state_chart)
    return executable


def test_pause_monitor_pauses_the_children_goal(immutable_model_world, rclpy_node):
    """
    The monitor and the children's goal are siblings inside the monitored goal, which is
    what makes the pause condition legal: it may only reference a sibling.
    """
    world, view, context = immutable_model_world
    monitor = ConstFalseNode(name="never")

    plan = pause_while(
        [MoveTorsoAction(TorsoState.HIGH)], monitor=monitor, context=context
    )
    executable = _parse_and_compile(plan, world, context)

    monitored_goal = _monitored_goal_of(executable)
    assert type(monitored_goal) is PausedWhileTrue
    assert monitored_goal.nodes == [monitor, monitored_goal.monitored_node]
    assert monitored_goal.monitored_node.pause_condition.free_variables() == [
        monitor.observation_variable
    ]


def test_pause_until_monitor_pauses_the_children_goal(
    immutable_model_world, rclpy_node
):
    """
    The children's goal is paused on the negated monitor observation, so it is held
    until the monitor turns True rather than while it is True.
    """
    world, view, context = immutable_model_world
    monitor = ConstFalseNode(name="never")

    plan = pause_until(
        [MoveTorsoAction(TorsoState.HIGH)], monitor=monitor, context=context
    )
    executable = _parse_and_compile(plan, world, context)

    monitored_goal = _monitored_goal_of(executable)
    assert type(monitored_goal) is PausedUntilTrue
    assert monitored_goal.nodes == [monitor, monitored_goal.monitored_node]
    assert monitored_goal.monitored_node.pause_condition.free_variables() == [
        monitor.observation_variable
    ]


def test_cancel_monitor_ends_the_children_goal(immutable_model_world, rclpy_node):
    world, view, context = immutable_model_world
    monitor = ConstFalseNode(name="never")

    plan = cancel_when(
        [MoveTorsoAction(TorsoState.HIGH)], monitor=monitor, context=context
    )
    executable = _parse_and_compile(plan, world, context)

    monitored_goal = _monitored_goal_of(executable)
    assert type(monitored_goal) is CancelledWhenTrue
    assert monitored_goal.nodes[:2] == [monitor, monitored_goal.monitored_node]
    assert monitored_goal.monitored_node.end_condition.free_variables() == [
        monitor.observation_variable
    ]


def test_cancel_monitor_ends_the_motion_when_the_monitor_fires(
    immutable_model_world, rclpy_node
):
    """
    The monitored goal holds a node that ends the motion, so giving up on the subtree
    gives up on the plan rather than leaving the rest of it waiting.
    """
    world, view, context = immutable_model_world
    monitor = ConstFalseNode(name="never")

    plan = cancel_when(
        [MoveTorsoAction(TorsoState.HIGH)], monitor=monitor, context=context
    )
    executable = _parse_and_compile(plan, world, context)

    monitored_goal = _monitored_goal_of(executable)
    [cancelled] = [
        node for node in monitored_goal.nodes if isinstance(node, CancelMotion)
    ]
    assert cancelled.exception == monitored_goal.exception
    assert cancelled.start_condition.free_variables() == [monitor.observation_variable]


def test_monitored_subtree_nested_in_a_sequence_compiles(
    immutable_model_world, rclpy_node
):
    """
    A monitored subtree is a node like any other in the surrounding sequence.

    Compiling is the real assertion: it runs the condition scope validation that this
    structure exists to satisfy.
    """
    world, view, context = immutable_model_world

    plan = sequential(
        [
            MoveTorsoAction(TorsoState.LOW),
            cancel_when(
                [MoveTorsoAction(TorsoState.HIGH)], monitor=ConstFalseNode(name="never")
            ),
        ],
        context=context,
    )
    executable = _parse_and_compile(plan, world, context)

    assert len(executable.motion_state_chart.get_nodes_by_type(CancelledWhenTrue)) == 1


# %% repeating a subtree


def test_repeat_node_wraps_its_children_in_a_repeating_goal(
    immutable_model_world, rclpy_node
):
    """
    A repeat contributes a goal that holds the children, the attempt counter and the
    node that reports running out of attempts, all as siblings so the wiring between
    them is legal.
    """
    world, view, context = immutable_model_world

    plan = repeat(
        [MoveTorsoAction(TorsoState.HIGH)], maximum_repetitions=3, context=context
    )
    executable = _parse_and_compile(plan, world, context)

    [loop] = executable.root_node.nodes
    assert type(loop) is RepeatOnStall
    assert loop.task in loop.nodes
    [counter] = [node for node in loop.nodes if isinstance(node, CountNodeResets)]
    assert counter.target == 3
    assert counter is loop.stop_retry_monitor
    [exhausted] = [node for node in loop.nodes if isinstance(node, CancelMotion)]
    assert exhausted.start_condition.free_variables() == [counter.observation_variable]


def test_merge_motions(immutable_model_world, rclpy_node):
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

    executable = plan.parse()

    assert type(executable) == GiskardExecutable
    assert len(executable.motion_mappings) == 2
    assert executable.pre_condition_node
    assert executable.post_condition_node

    with simulated_robot:
        executable.execute()


def test_parse_pick_up(immutable_model_world):
    world, view, context = immutable_model_world

    plan = execute_single(
        PickUpAction(
            world.get_semantic_annotations_by_type(Milk)[0],
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

    # plan.plan.visualize()

    executable = plan.parse()

    assert len(executable.execution_list) == 3
    assert type(executable.execution_list[0]) == GiskardExecutable
    assert type(executable.execution_list[1]) == MoveBranchExecutable
    assert type(executable.execution_list[2]) == GiskardExecutable


def test_parse_pick_up_merges_motions_around_model_change(immutable_model_world):
    """
    The motions on each side of the model change (the attach) must be merged into a
    single giskard executable per side, so the model change splits the plan into exactly
    [merged motions, model change, merged motions].
    """
    world, view, context = immutable_model_world

    plan = execute_single(
        PickUpAction(
            world.get_semantic_annotations_by_type(Milk)[0],
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

    # The four motions before the attach (open gripper, reach pre-pose, reach pose,
    # close gripper) merge into one executable; the lift after it into another.
    assert len(executable.execution_list[0].motion_mappings) == 4
    assert len(executable.execution_list[2].motion_mappings) == 1


def test_parse_complex_plan(immutable_model_world):
    world, view, context = immutable_model_world

    plan = sequential(
        [
            ParkArmsAction(Arms.BOTH),
            ReachAction(
                target_pose=Pose(
                    Point3.from_iterable([1, -2, 0.8]), reference_frame=world.root
                ),
                object_designator=world.get_semantic_annotations_by_type(Milk)[0],
                arm=Arms.LEFT,
                grasp_description=GraspDescription(
                    ApproachDirection.FRONT,
                    VerticalAlignment.NoAlignment,
                    view.right_arm.end_effector,
                ),
            ),
        ],
        context=context,
    )

    plan.notify()
    exec = plan.parse()
    assert type(exec) == GiskardExecutable
    assert len(exec.motion_mappings) == 3


def test_parsing_two_actions_into_one_exec(immutable_model_world):
    world, view, context = immutable_model_world

    plan = sequential(
        [
            ParkArmsAction(Arms.BOTH),
            ReachAction(
                target_pose=Pose(
                    Point3.from_iterable([1, -2, 0.8]), reference_frame=world.root
                ),
                object_designator=world.get_semantic_annotations_by_type(Milk)[0],
                arm=Arms.LEFT,
                grasp_description=GraspDescription(
                    ApproachDirection.FRONT,
                    VerticalAlignment.NoAlignment,
                    view.right_arm.end_effector,
                ),
            ),
        ],
        context=context,
    )

    plan.notify()
    exec = plan.parse()

    assert type(exec) == GiskardExecutable
    assert len(exec.motion_mappings) == 3


def test_parse_pick_place(immutable_model_world):
    world, view, context = immutable_model_world

    plan = sequential(
        [
            PickUpAction(
                world.get_semantic_annotations_by_type(Milk)[0],
                Arms.RIGHT,
                GraspDescription(
                    ApproachDirection.FRONT,
                    VerticalAlignment.NoAlignment,
                    view.right_arm.end_effector,
                ),
            ),
            PlaceAction(
                world.get_body_by_name("milk.stl"),
                Pose(reference_frame=world.root),
                Arms.RIGHT,
            ),
        ],
        context=context,
    )

    plan.notify()

    # plan.plan.visualize()

    executable = plan.parse()

    assert len(executable.execution_list) == 2
    assert len(executable.execution_list[0].execution_list) == 3
    assert len(executable.execution_list[1].execution_list) == 3


def test_parse_transport_plan(mutable_model_world, rclpy_node):
    world, view, context = mutable_model_world

    plan = sequential(
        [
            MoveTorsoAction(TorsoState.HIGH),
            ParkArmsAction(Arms.BOTH),
            TransportAction(
                world.get_semantic_annotations_by_type(Milk)[0],
                Pose.from_xyz_rpy(2.37, 2.5, 1.05, reference_frame=world.root),
                Arms.RIGHT,
            ),
        ],
        context=context,
    )

    plan.notify()
    exec = plan.parse()

    with simulated_robot:
        exec.execute()


# %% execution boundaries


@dataclass(eq=False, repr=False)
class BoundaryNode(ExecutionBoundaryNode):
    """
    Node that declares itself an execution boundary and parses to a non-giskard
    executable.

    Stands in for any node that interrupts the merging of motions.
    """

    def notify(self) -> None:
        pass

    def parse(self) -> Executable:
        return Executable(context=self.plan.context)


def test_execution_boundary_splits_the_merged_motion_chart(immutable_model_world):
    """
    A node declaring itself an execution boundary separates the motions around it into
    one merged chart per side, instead of all of them collapsing into a single chart.
    """
    world, view, context = immutable_model_world

    plan = sequential(
        [
            MoveToolCenterPointMotion(Pose(reference_frame=world.root), Arms.LEFT),
            MoveToolCenterPointMotion(Pose(reference_frame=world.root), Arms.RIGHT),
            BoundaryNode(),
            MoveToolCenterPointMotion(Pose(reference_frame=world.root), Arms.LEFT),
        ],
        context=context,
    )

    plan.notify()
    executable = plan.parse()

    assert [type(child) for child in executable.execution_list] == [
        GiskardExecutable,
        Executable,
        GiskardExecutable,
    ]
    assert len(executable.execution_list[0].motion_mappings) == 2
    assert len(executable.execution_list[2].motion_mappings) == 1


# %% perception inside the merged chart


def test_detecting_motion_merges_with_the_motions_around_it(immutable_model_world):
    """
    Perception is a motion like any other, so it does not interrupt the merging of the
    motions around it: one chart holds the detection and both moves.
    """
    world, view, context = immutable_model_world
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

    plan = sequential(
        [
            MoveToolCenterPointMotion(Pose(reference_frame=world.root), Arms.LEFT),
            DetectingMotion(query=query),
            MoveToolCenterPointMotion(Pose(reference_frame=world.root), Arms.RIGHT),
        ],
        context=context,
    )
    plan.notify()
    executable = plan.parse()

    assert type(executable) == GiskardExecutable
    assert len(executable.motion_mappings) == 3
    assert [type(task) for task in executable.motion_mappings.values()] == [
        CartesianPose,
        PerceptionTask,
        CartesianPose,
    ]


def test_detect_action_parses_to_a_single_motion_chart(immutable_model_world):
    """
    An action that only perceives still compiles to a motion chart, so its conditions
    are carried by that chart rather than needing to run around it.
    """
    world, view, context = immutable_model_world

    plan = execute_single(
        DetectAction(DetectionTechnique.TYPES, object_sem_annotation=Milk),
        context=context,
    )
    plan.notify()
    executable = plan.parse()

    assert type(executable) == GiskardExecutable
    assert [type(task) for task in executable.motion_mappings.values()] == [
        PerceptionTask
    ]
    assert executable.pre_condition_node
    assert executable.post_condition_node


# %% perceiving before the grasp


def detect_actions_of(plan: PlanNode) -> List[DetectAction]:
    """
    :param plan: The expanded plan to search.
    :return: The detections the plan performs, in no particular order.
    """
    return [
        node.designator
        for node in plan.descendants
        if isinstance(node, ActionNode) and isinstance(node.designator, DetectAction)
    ]


def reach_action(milk: Milk, view, **kwargs) -> ReachAction:
    """
    :param milk: The object the reach is aimed at.
    :param view: The robot reaching for it.
    :param kwargs: The fields under test.
    :return: A reach at the object's own frame.
    """
    return ReachAction(
        target_pose=Pose(reference_frame=milk.root),
        arm=Arms.RIGHT,
        grasp_description=GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            view.right_arm.end_effector,
        ),
        object_designator=milk,
        **kwargs,
    )


def test_a_reach_does_not_perceive_by_default(immutable_model_world):
    """
    A reach acts on the pose the world already holds, so it must not spend a detection
    the caller did not ask for.
    """
    world, view, context = immutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]

    plan = execute_single(reach_action(milk, view), context=context)
    plan.notify()

    assert detect_actions_of(plan) == []


def test_perceiving_before_the_grasp_detects_the_object_being_reached_for(
    immutable_model_world,
):
    """
    The detection has to ask for the object the reach was given, so that a plan grasping
    something else does not query for the wrong thing.
    """
    world, view, context = immutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]

    plan = execute_single(
        reach_action(milk, view, perceive_before_grasp=True), context=context
    )
    plan.notify()

    [detection] = detect_actions_of(plan)
    assert detection.object_sem_annotation is type(milk)


def test_a_pick_up_passes_perceiving_on_to_its_reach(immutable_model_world):
    """
    The flag is set on the pick-up, but the detection belongs to the reach inside it, so
    it has to survive that hand-over.
    """
    world, view, context = immutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]

    plan = execute_single(
        PickUpAction(
            milk,
            Arms.RIGHT,
            GraspDescription(
                ApproachDirection.FRONT,
                VerticalAlignment.NoAlignment,
                view.right_arm.end_effector,
            ),
            perceive_before_grasp=True,
        ),
        context=context,
    )
    plan.notify()

    [detection] = detect_actions_of(plan)
    assert detection.object_sem_annotation is type(milk)


def test_perceiving_without_an_object_to_detect_is_rejected(immutable_model_world):
    """
    A reach may be given a pose without an object, but then there is nothing to build
    the detection query from, so the contradiction is reported instead of guessed away.
    """
    world, view, context = immutable_model_world

    reach = ReachAction(
        target_pose=Pose(reference_frame=world.root),
        arm=Arms.RIGHT,
        grasp_description=GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            view.right_arm.end_effector,
        ),
        perceive_before_grasp=True,
    )

    with pytest.raises(PerceptionTargetMissing):
        execute_single(reach, context=context).notify()


# %% expansion-time pose capture


def test_pick_up_motions_follow_the_object_moved_after_expansion(immutable_model_world):
    """
    The whole plan is expanded before the first motion runs, so a pick-up that captured
    the object's pose in world coordinates could never act on a pose corrected in
    between (for example by a detection).

    Keeping the motion targets in the object's own frame is what lets them follow it.
    """
    world, view, context = immutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    milk_body = milk.root

    plan = execute_single(
        PickUpAction(
            milk,
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
    targets = [
        node.designator.target
        for node in plan.descendants
        if isinstance(node, MotionNode)
        and isinstance(node.designator, MoveToolCenterPointMotion)
    ]
    positions_before = [
        world.transform(target, world.root).to_position().to_np().flatten()[:3]
        for target in targets
    ]

    displacement = np.array([0.25, -0.4, 0.1])
    milk_body.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        *(milk_body.global_pose.to_position().to_np().flatten()[:3] + displacement),
        reference_frame=world.root,
    )

    assert targets
    assert all(target.reference_frame is milk_body for target in targets)
    for target, position_before in zip(targets, positions_before):
        np.testing.assert_allclose(
            world.transform(target, world.root).to_position().to_np().flatten()[:3],
            position_before + displacement,
            atol=1e-9,
        )


# %% splitting helper


def test_split_by_type(immutable_model_world):
    world, view, context = immutable_model_world

    split_list = [
        MoveToolCenterPointMotion(Pose(), Arms.LEFT),
        ReAttachNode(body=world.get_body_by_name("milk.stl"), new_parent=world.root),
        MoveToolCenterPointMotion(Pose(), Arms.RIGHT),
    ]

    splitted_list = split_list_by_type(split_list, ReAttachNode)

    assert len(splitted_list) == 3
    assert len(splitted_list[0]) == 1
    assert len(splitted_list[1]) == 1
    assert len(splitted_list[2]) == 1


def test_split_by_type_empty_list():
    assert split_list_by_type([], ReAttachNode) == []


def test_split_by_type_without_match_stays_one_group():
    no_model_change = [
        MoveToolCenterPointMotion(Pose(), Arms.LEFT),
        MoveToolCenterPointMotion(Pose(), Arms.RIGHT),
    ]

    splitted_list = split_list_by_type(no_model_change, ReAttachNode)

    assert len(splitted_list) == 1
    assert splitted_list[0] == no_model_change


def test_split_by_type_groups_consecutive_elements(immutable_model_world):
    world, view, context = immutable_model_world
    model_change = ReAttachNode(
        body=world.get_body_by_name("milk.stl"), new_parent=world.root
    )

    split_list = [
        MoveToolCenterPointMotion(Pose(), Arms.LEFT),
        MoveToolCenterPointMotion(Pose(), Arms.RIGHT),
        model_change,
        MoveToolCenterPointMotion(Pose(), Arms.LEFT),
    ]

    splitted_list = split_list_by_type(split_list, ReAttachNode)

    assert [len(group) for group in splitted_list] == [2, 1, 1]
    assert splitted_list[1] == [model_change]
    assert all(not isinstance(element, ReAttachNode) for element in splitted_list[0])


def test_split_by_type_leading_and_trailing_match(immutable_model_world):
    world, view, context = immutable_model_world
    first_model_change = ReAttachNode(
        body=world.get_body_by_name("milk.stl"), new_parent=world.root
    )
    last_model_change = ReAttachNode(
        body=world.get_body_by_name("milk.stl"), new_parent=world.root
    )

    split_list = [
        first_model_change,
        MoveToolCenterPointMotion(Pose(), Arms.LEFT),
        last_model_change,
    ]

    splitted_list = split_list_by_type(split_list, ReAttachNode)

    assert [len(group) for group in splitted_list] == [1, 1, 1]
    assert splitted_list[0] == [first_model_change]
    assert splitted_list[2] == [last_model_change]
