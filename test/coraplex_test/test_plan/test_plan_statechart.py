"""
Tests for running a plan as one statechart, see
:class:`~coraplex.plans.executors.PlanExecutor`.
"""

import pytest
import json
from dataclasses import dataclass, field

from typing_extensions import Iterator

from coraplex.exceptions import MotionDidNotFinish
from coraplex.locations.base import Location
from coraplex.plans.failures import EmptyUnderspecified
from coraplex.robot_plans.actions.composite.facing import FaceAndLookAtAction
from coraplex.robot_plans.actions.core.navigation import (
    FaceAtAction,
    LookAtAction,
    NavigateAction,
)
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    SetGripperAction,
)
from coraplex.plans import executors
from cramph.composites import (
    StatementNode,
    ChildChooser,
    ChildChooserAccess,
    CompositeNodeChoosingItsChild,
    PausedUntilTrue,
    Sequence,
)
from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValues, ObservationStateValues
from cramph.executor import StatechartExecutor
from cramph.nodes_for_testing import ConstTrueNode, NodeFailingOnObservingFalse
from cramph.statechart import Statechart
from cramph.world_modification_nodes import MoveBranch
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from krrood.adapters.json_serializer import from_json, to_json
from krrood.entity_query_language.factories import a, variable
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Milk,
    Spoon,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from ...plan_running import robot_executor, run_plan, simulated_executor, statechart_of
from ...statechart_json_round_trip import StatechartSentThroughJSON

# %% helpers


def _pose_in_front_of_the_robot(world, robot) -> Pose:
    """
    :return: A pose in the world frame, a few centimetres in front of where the robot
        stands now.
    """
    return world.transform(
        Pose.from_xyz_rpy(x=0.05, reference_frame=robot.root), world.root
    )


@dataclass
class RecordingLocationInFrontOfTheRobot(Location):
    """
    A location in front of the robot that records, each time it is sampled, whether the
    torso stood high at that moment.
    """

    world: World
    """
    The world the robot stands in.
    """

    robot: AbstractRobot
    """
    The robot the location is in front of.
    """

    torso_high_when_sampled: list[bool] = field(default_factory=list)
    """
    Whether the torso stood high, once per sampling.
    """

    def candidates(self) -> Iterator[Pose]:
        torso_high = self.robot.get_torso().get_joint_state_by_type(TorsoState.HIGH)
        self.torso_high_when_sampled.append(torso_high.is_achieved())
        yield _pose_in_front_of_the_robot(self.world, self.robot)


# %% one statechart per plan


def test_a_plan_grounding_an_action_mid_sequence_runs_as_one_statechart(
    pr2_apartment_context,
) -> None:
    world, robot, extensions = pr2_apartment_context
    torso = MoveTorsoAction(torso_state=TorsoState.HIGH)
    plan = Sequence(
        nodes=[
            torso,
            StatementNode(
                statement=a(NavigateAction)(
                    target_location=variable(
                        Pose, domain=[_pose_in_front_of_the_robot(world, robot)]
                    )
                )
            ),
        ]
    )

    run_plan(plan, extensions)

    statechart: Statechart = plan.statechart
    navigate = plan.nodes[1].grounded_nodes[-1]
    assert isinstance(navigate, NavigateAction)
    assert navigate.statechart is statechart
    assert torso.statechart is statechart
    assert len(statechart.get_nodes_by_type(EndMotion)) == 1
    assert plan.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_an_underspecified_action_is_grounded_against_the_world_the_steps_before_left(
    pr2_apartment_context,
) -> None:
    world, robot, extensions = pr2_apartment_context
    torso_high = robot.get_torso().get_joint_state_by_type(TorsoState.HIGH)
    assert not torso_high.is_achieved()
    location = RecordingLocationInFrontOfTheRobot(world=world, robot=robot)

    plan = Sequence(
        nodes=[
            MoveTorsoAction(torso_state=TorsoState.HIGH),
            StatementNode(
                statement=a(NavigateAction)(
                    target_location=variable(Pose, domain=location)
                )
            ),
        ]
    )

    run_plan(plan, extensions)

    assert location.torso_high_when_sampled == [True]


def test_a_branch_moved_mid_plan_follows_its_new_parent(pr2_apartment_context) -> None:
    world, robot, extensions = pr2_apartment_context
    milk = world.get_body_by_name("milk.stl")
    tool_frame = robot.left_arm.end_effector.tool_frame
    height_before = milk.global_pose.z

    plan = Sequence(
        nodes=[
            MoveTorsoAction(torso_state=TorsoState.LOW),
            MoveBranch(branch_root=milk, new_parent=tool_frame),
            MoveTorsoAction(torso_state=TorsoState.HIGH),
        ]
    )
    run_plan(plan, extensions)

    assert milk.parent_connection.parent is tool_frame
    assert milk.global_pose.z > height_before


# %% actions as statechart nodes


def test_actions_with_the_same_parameters_are_different_nodes() -> None:
    first = MoveTorsoAction(torso_state=TorsoState.HIGH)
    second = MoveTorsoAction(torso_state=TorsoState.HIGH)

    assert first != second


def test_a_place_finds_the_grasp_of_the_pick_up_before_it(
    pr2_apartment_context,
) -> None:
    world, robot, extensions = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    pick_up = PickUpAction(grasp=milk.grasp_candidates()[0], arm=robot.left_arm)
    place = PlaceAction(
        object_designator=milk,
        target_location=Pose.from_xyz_rpy(0.8, -1.9, 0.7, reference_frame=world.root),
    )
    statechart = Statechart(context=simulated_executor(extensions).context)

    statechart.add_node(Sequence(nodes=[pick_up, place]))

    assert statechart.get_preceding_node_by_type(place, PickUpAction) is pick_up


def test_a_place_below_a_pausing_composite_finds_the_pick_up_before_that(
    pr2_apartment_context,
) -> None:
    world, robot, extensions = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    pick_up = PickUpAction(grasp=milk.grasp_candidates()[0], arm=robot.left_arm)
    place = PlaceAction(
        object_designator=milk,
        target_location=Pose.from_xyz_rpy(0.8, -1.9, 0.7, reference_frame=world.root),
    )
    paused_place = PausedUntilTrue(monitor=ConstTrueNode(), monitored_node=place)
    statechart = Statechart(context=simulated_executor(extensions).context)

    statechart.add_node(Sequence(nodes=[pick_up, paused_place]))

    assert statechart.get_preceding_node_by_type(place, PickUpAction) is pick_up


def test_a_place_after_several_pick_ups_uses_the_arm_that_picked_its_object(
    pr2_apartment_context,
) -> None:
    world, robot, extensions = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    spoon = world.get_semantic_annotations_by_type(Spoon)[0]
    place_milk = PlaceAction(
        object_designator=milk,
        target_location=Pose.from_xyz_rpy(0.8, -1.9, 0.7, reference_frame=world.root),
    )
    statechart = Statechart(context=simulated_executor(extensions).context)

    statechart.add_node(
        Sequence(
            nodes=[
                PickUpAction(grasp=milk.grasp_candidates()[0], arm=robot.left_arm),
                PickUpAction(grasp=spoon.grasp_candidates()[0], arm=robot.right_arm),
                place_milk,
            ]
        )
    )

    moved_tool_frames = {
        node.tip_link
        for node in place_milk.descendants
        if isinstance(node, CartesianPose)
    }
    assert moved_tool_frames == {robot.left_arm.end_effector.tool_frame}


def test_a_place_after_an_earlier_place_uses_the_arm_that_picked_its_object(
    pr2_apartment_context,
) -> None:
    world, robot, extensions = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    spoon = world.get_semantic_annotations_by_type(Spoon)[0]
    place_spoon = PlaceAction(
        object_designator=spoon,
        target_location=Pose.from_xyz_rpy(0.8, -1.9, 0.7, reference_frame=world.root),
    )
    statechart = Statechart(context=simulated_executor(extensions).context)

    statechart.add_node(
        Sequence(
            nodes=[
                PickUpAction(grasp=milk.grasp_candidates()[0], arm=robot.left_arm),
                PlaceAction(
                    object_designator=milk,
                    target_location=Pose.from_xyz_rpy(
                        0.8, -1.5, 0.7, reference_frame=world.root
                    ),
                ),
                PickUpAction(grasp=spoon.grasp_candidates()[0], arm=robot.right_arm),
                place_spoon,
            ]
        )
    )

    moved_tool_frames = {
        node.tip_link
        for node in place_spoon.descendants
        if isinstance(node, CartesianPose)
    }
    assert moved_tool_frames == {robot.right_arm.end_effector.tool_frame}


# %% running on the robot


@dataclass
class GiskardRecordingTheGoal:
    """
    Stands in for the connection to Giskard, recording what it is asked to execute.
    """

    compiled: Statechart | None = None
    """
    The statechart handed over last.
    """

    executed: list[Statechart] = field(default_factory=list)
    """
    Every statechart sent, in order.
    """

    child_choosers: list[ChildChooser] = field(default_factory=list)
    """
    The chooser of the context every sent statechart was built in.
    """

    closed: bool = False
    """
    Whether the connection was closed.
    """

    def compile(self, statechart: Statechart) -> None:
        self.compiled = statechart

    def execute(self) -> None:
        self.executed.append(self.compiled)
        self.child_choosers.extend(
            self.compiled.context.require_extension(ChildChooserAccess).choosers
        )

    def close(self) -> None:
        self.closed = True


def test_a_sent_plan_compiles_without_what_its_actions_read_to_expand(
    pr2_apartment_context,
) -> None:
    """
    Giskard compiles the plan it receives in a context of its own, which holds what
    motions need and nothing of what the actions read while the client expanded them.
    """
    world, robot, extensions = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    client = robot_executor(extensions)
    sent = statechart_of(
        client,
        Sequence(
            nodes=[
                MoveTorsoAction(torso_state=TorsoState.HIGH),
                NavigateAction(
                    target_location=_pose_in_front_of_the_robot(world, robot)
                ),
                PickUpAction(grasp=milk.grasp_candidates()[0], arm=robot.left_arm),
            ]
        ),
    )
    client.prepare(sent)
    receiver = StatechartExecutor(
        context=StatechartContext(world=world), extensions=[MotionControl()]
    )
    received = StatechartSentThroughJSON.received_in(sent, receiver.context).received

    receiver.compile(received)

    assert received.compiler.is_compiled


def test_an_underspecified_node_is_sent_as_a_node_choosing_its_child(
    pr2_apartment_context,
) -> None:
    """
    Giskard receives the children the client chooses, so it needs nothing but the node
    itself, and not the statement it is grounded from.
    """
    world, robot, extensions = pr2_apartment_context
    node = StatementNode(statement=a(NavigateAction)(target_location=...))

    received = from_json(json.loads(json.dumps(to_json(node))))

    assert type(received) is CompositeNodeChoosingItsChild
    assert received.name == node.name


def test_a_plan_on_the_robot_is_sent_once_with_the_chooser_grounding_its_actions(
    pr2_apartment_context, monkeypatch
) -> None:
    world, robot, extensions = pr2_apartment_context
    giskard = GiskardRecordingTheGoal()
    monkeypatch.setattr(
        executors, "GiskardClientExecutor", lambda ros_node, context: giskard
    )
    plan = Sequence(
        nodes=[
            MoveTorsoAction(torso_state=TorsoState.HIGH),
            StatementNode(
                statement=a(NavigateAction)(
                    target_location=variable(
                        Pose, domain=[_pose_in_front_of_the_robot(world, robot)]
                    )
                )
            ),
        ]
    )
    executor = robot_executor(extensions)

    executor.compile(statechart_of(executor, plan))
    executor.execute()

    [statechart] = giskard.executed
    assert plan.statechart is statechart
    [chooser] = giskard.child_choosers
    assert chooser is executor.child_chooser


def test_plans_run_one_after_another_share_one_connection_to_giskard(
    pr2_apartment_context, monkeypatch
) -> None:
    world, robot, extensions = pr2_apartment_context
    connections = []

    def connect(ros_node, context) -> GiskardRecordingTheGoal:
        connections.append(GiskardRecordingTheGoal())
        return connections[-1]

    monkeypatch.setattr(executors, "GiskardClientExecutor", connect)
    executor = robot_executor(extensions)
    plans = [
        MoveTorsoAction(torso_state=TorsoState.HIGH),
        MoveTorsoAction(torso_state=TorsoState.LOW),
    ]

    for plan in plans:
        executor.compile(statechart_of(executor, plan))
        executor.execute()

    [connection] = connections
    assert [statechart.top_level_nodes[0] for statechart in connection.executed] == (
        plans
    )


def test_closing_the_executor_closes_its_connection_to_giskard(
    pr2_apartment_context, monkeypatch
) -> None:
    world, robot, extensions = pr2_apartment_context
    giskard = GiskardRecordingTheGoal()
    monkeypatch.setattr(
        executors, "GiskardClientExecutor", lambda ros_node, context: giskard
    )
    executor = robot_executor(extensions)
    executor.compile(
        statechart_of(executor, MoveTorsoAction(torso_state=TorsoState.HIGH))
    )
    executor.execute()

    executor.close()

    assert giskard.closed


def test_an_executor_that_never_ran_a_plan_connects_to_nothing(
    pr2_apartment_context, monkeypatch
) -> None:
    world, robot, extensions = pr2_apartment_context
    connections = []
    monkeypatch.setattr(
        executors,
        "GiskardClientExecutor",
        lambda ros_node, context: connections.append(GiskardRecordingTheGoal()),
    )
    executor = robot_executor(extensions)

    executor.close()

    assert connections == []


@dataclass
class GiskardTickingTheGoalUntilItEnded:
    """
    Stands in for Giskard, ticking what it is asked to execute until the statechart
    ended, which is the only way Giskard's control loop finishes a goal by itself.
    """

    tick_limit: int = 50
    """
    The number of ticks after which a statechart that did not end is given up on.
    """

    compiled: Statechart | None = None
    """
    The statechart handed over last.
    """

    def compile(self, statechart: Statechart) -> None:
        self.compiled = statechart

    def execute(self) -> None:
        giskard = StatechartExecutor(
            context=self.compiled.context, extensions=[MotionControl()]
        )
        giskard.compile(self.compiled)
        giskard.tick_until_end(timeout=self.tick_limit)


def test_a_plan_that_fails_on_the_robot_raises_motion_did_not_finish(
    pr2_apartment_context, monkeypatch
) -> None:
    world, robot, extensions = pr2_apartment_context
    monkeypatch.setattr(
        executors,
        "GiskardClientExecutor",
        lambda ros_node, context: GiskardTickingTheGoalUntilItEnded(),
    )
    plan = NodeFailingOnObservingFalse(observation=ObservationStateValues.FALSE)
    executor = robot_executor(extensions)
    executor.compile(statechart_of(executor, plan))

    with pytest.raises(MotionDidNotFinish) as failure:
        executor.execute()

    assert failure.value.unfinished_motions == [plan]
    assert plan.life_cycle_state == LifeCycleValues.FAILED


def test_an_underspecified_action_out_of_candidates_on_the_robot_says_it_is_empty(
    pr2_apartment_context, monkeypatch
) -> None:
    world, robot, extensions = pr2_apartment_context
    monkeypatch.setattr(
        executors,
        "GiskardClientExecutor",
        lambda ros_node, context: GiskardTickingTheGoalUntilItEnded(),
    )
    plan = StatementNode(
        statement=a(NavigateAction)(target_location=variable(Pose, domain=[]))
    )
    executor = robot_executor(extensions)
    executor.compile(statechart_of(executor, plan))

    with pytest.raises(EmptyUnderspecified) as failure:
        executor.execute()

    assert failure.value.node is plan


def test_an_expanded_action_is_received_with_the_goal_it_reaches(
    pr2_apartment_context,
) -> None:
    """
    A receiver does not expand the nodes of a statechart again, so an action has to
    arrive knowing the goal it reaches.
    """
    world, robot, extensions = pr2_apartment_context
    sent = Statechart(context=simulated_executor(extensions).context)
    sent.add_node(
        action := SetGripperAction(
            configuration=robot.left_arm.end_effector.default_configuration(
                GripperState.OPEN
            )
        )
    )

    received = Statechart.from_json(
        json.loads(json.dumps(sent.to_json())),
        context=simulated_executor(extensions).context,
        **WorldEntityWithIDKwargsTracker.from_world(world).create_kwargs(),
    )

    received_action = received.get_node_by_index(action.index)
    assert received_action.goal is received.get_node_by_index(action.goal.index)


def test_an_expanded_action_is_received_with_the_steps_it_runs(
    pr2_apartment_context,
) -> None:
    """
    The steps of an action are its children, which arrive linked to it rather than
    created again.
    """
    world, robot, extensions = pr2_apartment_context
    target = Pose.from_xyz_rpy(1.0, 0.0, 1.0, reference_frame=world.root)
    sent = Statechart(context=simulated_executor(extensions).context)
    sent.add_node(
        action := FaceAndLookAtAction(
            face_at=FaceAtAction(target=target), look_at=LookAtAction(target=target)
        )
    )

    received = Statechart.from_json(
        json.loads(json.dumps(sent.to_json())),
        context=simulated_executor(extensions).context,
        **WorldEntityWithIDKwargsTracker.from_world(world).create_kwargs(),
    )

    received_action = received.get_node_by_index(action.index)
    assert type(received_action) is FaceAndLookAtAction
    assert received_action.nodes == [
        received.get_node_by_index(step.index) for step in action.nodes
    ]
    assert [type(step) for step in received_action.nodes] == [
        FaceAtAction,
        LookAtAction,
    ]
