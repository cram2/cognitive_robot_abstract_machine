import json
from dataclasses import dataclass, field
from typing import Any, List, Optional

import pytest

from giskardpy.executor import Executor, NoPacing
from giskardpy.middleware.ros2.action_server import GoalOutcome
from giskardpy.middleware.ros2.command_publishing import CommandPublisher
from giskardpy.middleware.ros2.control_loop import ControlLoop
from giskardpy.middleware.ros2.exceptions import (
    ExecutionCanceledException,
    RequiredWorldUpdateNotReceivedError,
    WorldModelModifiedDuringMotionError,
)
from giskardpy.middleware.ros2.feedback_publisher import ActionFeedbackPublisher
from giskardpy.middleware.ros2.cycle_counter import CycleCounter
from giskardpy.middleware.ros2.input_synchronization import (
    InputSynchronizer,
    WorldStateInputs,
)
from giskardpy.middleware.ros2.motion_goal import MotionGoal
from giskardpy.middleware.ros2.motion_server import MotionServer
from giskardpy.middleware.ros2.post_goal_plotters import PostGoalPlotter
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.payload_monitors import (
    CountSimulationTimeSeconds,
)
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.qp.qp_controller_config import QPControllerConfig
from krrood.adapters.json_serializer import from_json
from semantic_digital_twin.adapters.ros.messages import MetaData, StreamPosition
from semantic_digital_twin.callbacks.callback import StateChangeCallback
from semantic_digital_twin.world import World

# %% mimics of the ros facing collaborators


@dataclass
class GoalQueueMimic:
    """
    Stands in for the action server: hands out one goal and records the outcome.
    """

    goal_json: Optional[str] = None
    """
    The goal that is waiting to be accepted, or ``None`` if there is none.
    """

    action_name: str = "mimic"
    """
    Name reported in cancel exceptions.
    """

    goal_id: int = -1
    """
    Number of goals accepted so far.
    """

    cancel_requested: bool = False
    """
    Whether the current goal should be canceled.
    """

    goal_msg: Optional[Any] = field(init=False, default=None)
    """
    Request of the accepted goal.
    """

    result_message: Optional[Any] = field(init=False, default=None)
    """
    Result built for the accepted goal.
    """

    outcome: Optional[GoalOutcome] = field(init=False, default=None)
    """
    Whether the goal was marked as succeeded, aborted or canceled.
    """

    sent_results: List[Any] = field(init=False, default_factory=list)
    """
    Every result that was handed back to a client.
    """

    feedback_messages: List[Any] = field(init=False, default_factory=list)
    """
    Every feedback message that was published.
    """

    def has_goal(self) -> bool:
        return self.goal_json is not None

    def accept_goal(self) -> None:
        self.goal_msg = GoalMessageMimic(goal=self.goal_json)
        self.goal_json = None
        self.goal_id += 1

    def is_cancel_requested(self) -> bool:
        return self.cancel_requested

    def loginfo(self, message: str) -> None:
        pass

    def send_feedback(self, message: Any) -> None:
        self.feedback_messages.append(message)

    def set_canceled(self) -> None:
        self.outcome = GoalOutcome.CANCELED

    def set_aborted(self) -> None:
        self.outcome = GoalOutcome.ABORTED

    def set_succeeded(self) -> None:
        self.outcome = GoalOutcome.SUCCEEDED

    def send_result(self) -> None:
        self.sent_results.append(self.result_message)


@dataclass
class GoalMessageMimic:
    """
    Stands in for the goal request of the action.
    """

    goal: str
    """
    The motion statechart as json.
    """


@dataclass
class WorldUpdatesMimic:
    """
    Stands in for the incoming world updates, counting how they are drained and
    recording what the server had already done whenever it asked about a position.
    """

    executor: Optional[Executor] = None
    """
    The executor whose compiled motion statechart is observed while a goal waits.
    """

    applied_batches: int = 0
    """
    How often everything that was received was applied.
    """

    applied_state_update_batches: int = 0
    """
    How often the state up to the next model change was applied.
    """

    pending_model_change: bool = False
    """
    Whether a model change is waiting to be applied.
    """

    drains_until_caught_up: Optional[int] = None
    """
    How many drains it takes until an awaited position counts as applied, or ``None`` if
    it never does.
    """

    awaited_positions: List[StreamPosition] = field(default_factory=list)
    """
    Every position this mimic was asked about.
    """

    compiled_while_waiting: List[bool] = field(default_factory=list)
    """
    Whether a motion statechart was already compiled, per position asked about.
    """

    def apply_all(self) -> None:
        self.applied_batches += 1
        self.pending_model_change = False

    def apply_state_updates(self) -> None:
        self.applied_state_update_batches += 1

    def has_applied(self, position: StreamPosition) -> bool:
        self.awaited_positions.append(position)
        self.compiled_while_waiting.append(
            self.executor is not None and self.executor.motion_statechart is not None
        )
        if self.drains_until_caught_up is None:
            return False
        return self.applied_batches >= self.drains_until_caught_up

    @property
    def has_pending_model_change(self) -> bool:
        return self.pending_model_change


@dataclass
class PublicationProgressMimic:
    """
    Stands in for the synchronizer that publishes the changes of the world, reporting
    how far it has published.
    """

    published_sequence_number: int = 0
    """
    Sequence number of the message that was published last.
    """

    origin: MetaData = field(
        default_factory=lambda: MetaData(node_name="mimic", process_id=0)
    )
    """
    The publisher the sequence numbers belong to.
    """

    def publish_one_update(self) -> None:
        """
        Pretend that one more change of the world went out.
        """
        self.published_sequence_number += 1

    @property
    def latest_published_position(self) -> StreamPosition:
        return StreamPosition(
            origin=self.origin, sequence_number=self.published_sequence_number
        )


@dataclass
class RecordingInputSynchronizer(InputSynchronizer):
    """
    Records in which order inputs are read relative to the control cycles.
    """

    executor: Executor = None
    """
    The executor whose control cycles are recorded on every apply.
    """

    applied_at_control_cycles: List[float] = field(default_factory=list)
    """
    The control cycle count at the time of every apply.
    """

    def apply(self) -> bool:
        self.applied_at_control_cycles.append(self.executor.control_cycles)
        return False


@dataclass
class WritingInputSynchronizer(InputSynchronizer):
    """
    Reports that it wrote into the world state, standing in for an input that received a
    message.
    """

    def apply(self) -> bool:
        return True


@dataclass
class RecordingCommandPublisher(CommandPublisher):
    """
    Records how often commands were published and when the robot was stopped.
    """

    published_velocities: List[float] = field(default_factory=list)
    """
    The commanded velocity of every publish.
    """

    stop_count: int = 0
    """
    How often the robot was told to stop.
    """

    world: World = None
    """
    The world the commanded velocities are read from.
    """

    def publish(self) -> None:
        self.published_velocities.append(float(self.world.state.velocities.sum()))

    def stop(self) -> None:
        self.stop_count += 1


@dataclass
class PublishingCommandPublisher(CommandPublisher):
    """
    Advances the publication progress once per control cycle, standing in for a goal
    whose changes to the world are published to the other processes.
    """

    publication_progress: PublicationProgressMimic = None
    """
    The progress that is advanced on every publish.
    """

    def publish(self) -> None:
        self.publication_progress.publish_one_update()

    def stop(self) -> None:
        pass


@dataclass
class FailingInputSynchronizer(InputSynchronizer):
    """
    Fails while reading its input, standing in for a broken robot interface.
    """

    def apply(self) -> bool:
        raise BrokenInputError()


class BrokenInputError(Exception):
    """
    Raised by :class:`FailingInputSynchronizer`.
    """


@dataclass
class CycleWatchingGoalCanceler(InputSynchronizer):
    """
    Watches the completed cycles from inside the control loop and cancels the goal once
    enough of them passed, standing in for a client that cancels a never-ending motion.
    """

    cycle_counter: CycleCounter = None
    """
    The counter that is watched.
    """

    action_server: Optional[GoalQueueMimic] = None
    """
    The action server the cancel request is written to.
    """

    ticks_until_cancel: int = 5
    """
    How many ticks to observe before requesting the cancel.
    """

    observed_ticks: int = 0
    """
    How many ticks were observed while the goal was running.
    """

    def apply(self) -> bool:
        self.observed_ticks = self.cycle_counter.completed_cycles
        if self.observed_ticks >= self.ticks_until_cancel:
            self.action_server.cancel_requested = True
        return False


@dataclass
class FeedbackCountingSynchronizer(InputSynchronizer):
    """
    Records how much feedback was already published when a control cycle read its
    inputs.
    """

    action_server: Optional[GoalQueueMimic] = None
    """
    The action server whose published feedback is counted.
    """

    published_feedback_per_cycle: List[int] = field(default_factory=list)
    """
    Number of feedback messages published before each control cycle.
    """

    def apply(self) -> bool:
        self.published_feedback_per_cycle.append(
            len(self.action_server.feedback_messages)
        )
        return False


@dataclass(eq=False)
class StateChangeRecorder(StateChangeCallback):
    """
    Records every announced state change, standing in for the publishers that observe
    the world through its callbacks.
    """

    announced_changes: int = 0
    """
    How often a state change was announced to the observers of the world.
    """

    def on_state_change(self, **kwargs) -> None:
        self.announced_changes += 1


@dataclass
class PendingModelChangeInjector(InputSynchronizer):
    """
    Announces a pending model change after a few control cycles, standing in for another
    process that changes the world model mid-motion.
    """

    world_updates: Optional[WorldUpdatesMimic] = None
    """
    The incoming updates the model change is announced on.
    """

    cycles_until_model_change: int = 5
    """
    How many control cycles to run before the model change is announced.
    """

    observed_cycles: int = 0
    """
    How many control cycles were observed so far.
    """

    def apply(self) -> bool:
        self.observed_cycles += 1
        if self.observed_cycles >= self.cycles_until_model_change:
            self.world_updates.pending_model_change = True
        return False


@dataclass
class RecordingPlotter(PostGoalPlotter):
    """
    Records for which goals debug plots were requested.
    """

    plotted_goal_ids: List[int] = field(default_factory=list)
    """
    The goal ids that were plotted.
    """

    def plot(self, goal_id: int) -> None:
        self.plotted_goal_ids.append(goal_id)


@dataclass
class FailingPlotter(PostGoalPlotter):
    """
    Fails while drawing its plot, standing in for a broken plotting backend.
    """

    def plot(self, goal_id: int) -> None:
        raise BrokenPlotError()


class BrokenPlotError(Exception):
    """
    Raised by :class:`FailingPlotter`.
    """


# %% fixtures


def create_executor() -> Executor:
    """
    Build an executor that simulates as fast as possible in an empty world.
    """
    return Executor(
        context=MotionStatechartContext(
            world=World(),
            qp_controller_config=QPControllerConfig.create_with_simulation_defaults(),
        ),
        pacer=NoPacing(),
    )


def feedback_data(message: Any) -> dict:
    """
    The payload of a published feedback message.
    """
    return json.loads(message.feedback)


def create_goal_json(
    seconds: float = 0.5, required_position: Optional[StreamPosition] = None
) -> str:
    """
    Build the json of a goal whose motion ends after the given simulated time.
    """
    motion_statechart = MotionStatechart()
    motion_statechart.add_node(counter := CountSimulationTimeSeconds(seconds=seconds))
    motion_statechart.add_node(EndMotion.when_true(counter))
    goal = MotionGoal.for_motion_statechart(
        motion_statechart, required_position=required_position
    )
    return json.dumps(goal.to_json())


@dataclass
class MotionServerFixture:
    """
    A motion server wired to mimics, so its lifecycle can be driven from a test.
    """

    executor: Executor
    action_server: GoalQueueMimic
    world_updates: WorldUpdatesMimic
    publication_progress: PublicationProgressMimic
    motion_server: MotionServer
    control_loop: ControlLoop
    command_publisher: RecordingCommandPublisher
    idle_input: RecordingInputSynchronizer
    control_input: RecordingInputSynchronizer
    plotter: RecordingPlotter
    cycle_counter: CycleCounter


@pytest.fixture()
def motion_server(init_rospy) -> MotionServerFixture:
    executor = create_executor()
    world = executor.context.world
    action_server = GoalQueueMimic()
    world_updates = WorldUpdatesMimic(executor=executor)
    feedback_publisher = ActionFeedbackPublisher(
        executor=executor, action_server=action_server
    )
    command_publisher = RecordingCommandPublisher(world=world)
    control_input = RecordingInputSynchronizer(world=world, executor=executor)
    cycle_counter = CycleCounter()
    control_loop = ControlLoop(
        executor=executor,
        action_server=action_server,
        feedback_publisher=feedback_publisher,
        inputs=WorldStateInputs(world=world, synchronizers=[control_input]),
        cycle_counter=cycle_counter,
        world_updates=world_updates,
        command_publishers=[command_publisher],
    )
    idle_input = RecordingInputSynchronizer(world=world, executor=executor)
    plotter = RecordingPlotter(executor=executor)
    publication_progress = PublicationProgressMimic()
    server = MotionServer(
        executor=executor,
        action_server=action_server,
        control_loop=control_loop,
        world_updates=world_updates,
        world_synchronizer=publication_progress,
        feedback_publisher=feedback_publisher,
        inputs=WorldStateInputs(world=world, synchronizers=[idle_input]),
        cycle_counter=cycle_counter,
        post_goal_plotters=[plotter],
    )
    return MotionServerFixture(
        executor=executor,
        action_server=action_server,
        world_updates=world_updates,
        publication_progress=publication_progress,
        motion_server=server,
        control_loop=control_loop,
        command_publisher=command_publisher,
        idle_input=idle_input,
        control_input=control_input,
        plotter=plotter,
        cycle_counter=cycle_counter,
    )


# %% goal lifecycle


class TestGoalResult:
    """
    A goal is always answered, with an outcome that reflects what happened.
    """

    def test_finished_motion_succeeds(self, motion_server: MotionServerFixture):
        motion_server.action_server.goal_json = create_goal_json()

        motion_server.motion_server.run_idle_cycle()

        assert motion_server.action_server.outcome == GoalOutcome.SUCCEEDED
        assert len(motion_server.action_server.sent_results) == 1

    def test_a_goal_that_changed_the_world_reports_how_far_it_published(
        self, motion_server: MotionServerFixture
    ):
        """
        The client reads the world once the goal is answered, so it has to be told what
        to catch up with first.
        """
        motion_server.action_server.goal_json = create_goal_json()
        motion_server.publication_progress.publish_one_update()
        published_before_goal = (
            motion_server.publication_progress.published_sequence_number
        )
        motion_server.control_loop.command_publishers.append(
            PublishingCommandPublisher(
                publication_progress=motion_server.publication_progress
            )
        )

        motion_server.motion_server.run_idle_cycle()

        result = json.loads(motion_server.action_server.sent_results[0].result)
        position = from_json(result["published_position"])
        assert position.sequence_number > published_before_goal
        assert (
            position.sequence_number
            == motion_server.publication_progress.published_sequence_number
        )

    def test_a_goal_that_changed_nothing_reports_no_position(
        self, motion_server: MotionServerFixture
    ):
        """
        There is nothing to catch up with when the world published nothing, and waiting
        for a position that was already passed before the goal would never return.
        """
        motion_server.action_server.goal_json = create_goal_json()

        motion_server.motion_server.run_idle_cycle()

        result = json.loads(motion_server.action_server.sent_results[0].result)
        assert "published_position" not in result

    def test_canceled_goal_is_reported_as_canceled(
        self, motion_server: MotionServerFixture
    ):
        motion_server.action_server.goal_json = create_goal_json(seconds=100.0)
        motion_server.action_server.cancel_requested = True

        motion_server.motion_server.run_idle_cycle()

        assert motion_server.action_server.outcome == GoalOutcome.CANCELED

    def test_broken_input_aborts_the_goal(self, motion_server: MotionServerFixture):
        motion_server.control_loop.inputs.synchronizers = [
            FailingInputSynchronizer(world=motion_server.executor.context.world)
        ]
        motion_server.action_server.goal_json = create_goal_json()

        motion_server.motion_server.run_idle_cycle()

        assert motion_server.action_server.outcome == GoalOutcome.ABORTED
        assert len(motion_server.action_server.sent_results) == 1

    def test_the_reason_of_a_failure_is_reported(
        self, motion_server: MotionServerFixture
    ):
        """
        A motion terminated by a world model modification can be sent again, so the
        client has to be able to tell it apart from a real failure.
        """
        inject_modification = PendingModelChangeInjector(
            world=motion_server.executor.context.world,
            world_updates=motion_server.world_updates,
            cycles_until_model_change=5,
        )
        motion_server.control_loop.inputs.synchronizers.append(inject_modification)
        motion_server.action_server.goal_json = create_goal_json(seconds=1000.0)

        motion_server.motion_server.run_idle_cycle()

        result = json.loads(motion_server.action_server.sent_results[0].result)
        assert isinstance(
            from_json(result["error"]), WorldModelModifiedDuringMotionError
        )

    def test_the_fields_of_a_failure_survive_the_round_trip(
        self, motion_server: MotionServerFixture
    ):
        """
        An error only helps a client if it arrives with the details it was raised with.
        """
        result = motion_server.motion_server.create_result(
            ExecutionCanceledException(action_server_name="mimic", goal_id=7)
        )

        error = from_json(json.loads(result.result)["error"])

        assert error.action_server_name == "mimic"
        assert error.goal_id == 7

    def test_a_successful_goal_reports_no_failure(
        self, motion_server: MotionServerFixture
    ):
        motion_server.action_server.goal_json = create_goal_json()

        motion_server.motion_server.run_idle_cycle()

        result = json.loads(motion_server.action_server.sent_results[0].result)
        assert "error" not in result

    def test_result_contains_the_final_states(self, motion_server: MotionServerFixture):
        motion_server.action_server.goal_json = create_goal_json()

        motion_server.motion_server.run_idle_cycle()

        result = json.loads(motion_server.action_server.sent_results[0].result)
        assert "life_cycle_state" in result
        assert "observation_state" in result


class TestCleanupAfterGoal:
    """
    Whatever happens to a goal, the robot is stopped and the client is answered.
    """

    def test_robot_is_stopped_after_a_successful_goal(
        self, motion_server: MotionServerFixture
    ):
        motion_server.action_server.goal_json = create_goal_json()

        motion_server.motion_server.run_idle_cycle()

        assert motion_server.command_publisher.stop_count == 1

    def test_robot_is_stopped_after_a_failed_goal(
        self, motion_server: MotionServerFixture
    ):
        motion_server.control_loop.inputs.synchronizers = [
            FailingInputSynchronizer(world=motion_server.executor.context.world)
        ]
        motion_server.action_server.goal_json = create_goal_json()

        motion_server.motion_server.run_idle_cycle()

        assert motion_server.command_publisher.stop_count == 1

    def test_debug_plots_are_written_for_the_goal(
        self, motion_server: MotionServerFixture
    ):
        motion_server.action_server.goal_json = create_goal_json()

        motion_server.motion_server.run_idle_cycle()

        assert motion_server.plotter.plotted_goal_ids == [
            motion_server.action_server.goal_id
        ]

    def test_feedback_is_published_at_the_end_of_the_goal(
        self, motion_server: MotionServerFixture
    ):
        motion_server.action_server.goal_json = create_goal_json()

        motion_server.motion_server.run_idle_cycle()

        assert len(motion_server.action_server.feedback_messages) > 0

    def test_a_failing_plotter_still_answers_the_client(
        self, motion_server: MotionServerFixture
    ):
        motion_server.motion_server.post_goal_plotters = [
            FailingPlotter(executor=motion_server.executor)
        ]
        motion_server.action_server.goal_json = create_goal_json()

        motion_server.motion_server.run_idle_cycle()

        assert len(motion_server.action_server.sent_results) == 1

    def test_a_failing_plotter_keeps_the_server_taking_goals(
        self, motion_server: MotionServerFixture
    ):
        """
        A debug plot is a diagnostic, so it may never escape the idle cycle: an
        exception here would end the loop that serves goals and leave every later client
        waiting forever.
        """
        motion_server.motion_server.post_goal_plotters = [
            FailingPlotter(executor=motion_server.executor)
        ]
        motion_server.action_server.goal_json = create_goal_json()
        motion_server.motion_server.run_idle_cycle()

        motion_server.action_server.goal_json = create_goal_json()
        motion_server.motion_server.run_idle_cycle()

        assert len(motion_server.action_server.sent_results) == 2

    def test_a_failing_plotter_does_not_stop_the_remaining_plotters(
        self, motion_server: MotionServerFixture
    ):
        motion_server.motion_server.post_goal_plotters = [
            FailingPlotter(executor=motion_server.executor),
            motion_server.plotter,
        ]
        motion_server.action_server.goal_json = create_goal_json()

        motion_server.motion_server.run_idle_cycle()

        assert motion_server.plotter.plotted_goal_ids == [
            motion_server.action_server.goal_id
        ]


class TestGoalStructureFeedback:
    """
    The client learns the structure of a motion statechart once per goal.

    Serializing the structure is expensive, so it happens while the goal is compiled and
    not from inside a control cycle, whose duration is budgeted.
    """

    def test_the_structure_is_published_before_the_first_control_cycle(
        self, motion_server: MotionServerFixture
    ):
        watcher = FeedbackCountingSynchronizer(
            world=motion_server.executor.context.world,
            action_server=motion_server.action_server,
        )
        motion_server.control_loop.inputs.synchronizers = [watcher]
        motion_server.action_server.goal_json = create_goal_json()

        motion_server.motion_server.run_idle_cycle()

        assert watcher.published_feedback_per_cycle[0] == 1
        first_feedback = feedback_data(motion_server.action_server.feedback_messages[0])
        assert "motion_statechart" in first_feedback

    def test_the_structure_is_published_once_per_goal(
        self, motion_server: MotionServerFixture
    ):
        motion_server.action_server.goal_json = create_goal_json()

        motion_server.motion_server.run_idle_cycle()

        structures = [
            message
            for message in motion_server.action_server.feedback_messages
            if "motion_statechart" in feedback_data(message)
        ]
        assert len(structures) == 1

    def test_every_goal_publishes_its_own_structure(
        self, motion_server: MotionServerFixture
    ):
        motion_server.action_server.goal_json = create_goal_json()
        motion_server.motion_server.run_idle_cycle()
        motion_server.action_server.goal_json = create_goal_json()
        motion_server.motion_server.run_idle_cycle()

        goal_ids_with_structure = [
            feedback_data(message)["goal_id"]
            for message in motion_server.action_server.feedback_messages
            if "motion_statechart" in feedback_data(message)
        ]
        assert goal_ids_with_structure == [0, 1]


# %% input synchronization


class TestStateChangeAnnouncement:
    """
    Announcing a state change recomputes the forward kinematics and reaches every
    observer of the world, so it is only worth doing when an input wrote something.
    """

    def test_nothing_is_announced_when_no_input_wrote(self, init_rospy):
        world = World()
        inputs = WorldStateInputs(world=world, synchronizers=[])
        recorder = StateChangeRecorder(_world=world)

        inputs.synchronize()

        assert recorder.announced_changes == 0

    def test_the_change_is_announced_when_an_input_wrote(self, init_rospy):
        world = World()
        inputs = WorldStateInputs(
            world=world, synchronizers=[WritingInputSynchronizer(world=world)]
        )
        recorder = StateChangeRecorder(_world=world)

        inputs.synchronize()

        assert recorder.announced_changes == 1

    def test_the_state_is_announced_on_request_even_when_no_input_wrote(
        self, init_rospy
    ):
        """
        The idle loop needs this: nothing else announces while no goal is running, so
        the observers of the world would go stale.
        """
        world = World()
        inputs = WorldStateInputs(world=world, synchronizers=[])
        recorder = StateChangeRecorder(_world=world)

        inputs.synchronize_and_announce()

        assert recorder.announced_changes == 1


# %% control loop


class TestControlCycleOrder:
    """
    Every control cycle reads the robot before it computes the next command.
    """

    def test_inputs_are_read_before_the_controller_ticks(
        self, motion_server: MotionServerFixture
    ):
        motion_server.action_server.goal_json = create_goal_json()

        motion_server.motion_server.run_idle_cycle()

        applied = motion_server.control_input.applied_at_control_cycles
        assert applied == sorted(applied)
        assert applied[0] == 0
        assert applied[-1] == motion_server.executor.control_cycles - 1

    def test_commands_are_published_once_per_cycle(
        self, motion_server: MotionServerFixture
    ):
        motion_server.action_server.goal_json = create_goal_json()

        motion_server.motion_server.run_idle_cycle()

        assert len(motion_server.command_publisher.published_velocities) == len(
            motion_server.control_input.applied_at_control_cycles
        )


class TestStopClearsCommands:
    """
    Stopping the control loop leaves no commanded motion behind.
    """

    def test_derivatives_are_zeroed(self, motion_server: MotionServerFixture):
        world = motion_server.executor.context.world
        world.state.velocities[:] = 1
        world.state.accelerations[:] = 1
        world.state.jerks[:] = 1

        motion_server.control_loop.stop()

        assert not world.state.velocities.any()
        assert not world.state.accelerations.any()
        assert not world.state.jerks.any()

    def test_publishers_are_stopped(self, motion_server: MotionServerFixture):
        motion_server.control_loop.stop()

        assert motion_server.command_publisher.stop_count == 1


# %% idle loop


class TestIdleLoop:
    """
    While waiting for a goal, Giskard keeps its world in sync with the outside.
    """

    def test_world_updates_of_other_processes_are_applied(
        self, motion_server: MotionServerFixture
    ):
        motion_server.motion_server.run_idle_cycle()

        assert motion_server.world_updates.applied_batches == 1

    def test_the_state_change_is_announced_to_the_observers_of_the_world(
        self, motion_server: MotionServerFixture
    ):
        """
        The idle loop hands the published state to the world callbacks instead of
        calling a publisher itself, so every observer of the world sees it.
        """
        recorder = StateChangeRecorder(_world=motion_server.executor.context.world)

        motion_server.motion_server.run_idle_cycle()

        assert recorder.announced_changes == 1

    def test_inputs_are_read(self, motion_server: MotionServerFixture):
        motion_server.motion_server.run_idle_cycle()

        assert len(motion_server.idle_input.applied_at_control_cycles) == 1

    def test_one_cycle_is_counted_per_idle_cycle(
        self, motion_server: MotionServerFixture
    ):
        motion_server.motion_server.run_idle_cycle()
        motion_server.motion_server.run_idle_cycle()

        assert motion_server.cycle_counter.completed_cycles == 2

    def test_nothing_happens_while_the_world_is_being_modified(
        self, motion_server: MotionServerFixture
    ):
        with motion_server.executor.context.world.modify_world():
            motion_server.motion_server.run_idle_cycle()

        assert motion_server.cycle_counter.completed_cycles == 0
        assert motion_server.world_updates.applied_batches == 0


class TestCycleCountingDuringGoals:
    """
    The cycle counter keeps counting while a goal is running, so an observer waiting for
    the server to make progress is never blocked by a motion that only ends on cancel.
    """

    def test_cycles_are_counted_during_a_goal(self, motion_server: MotionServerFixture):
        motion_server.action_server.goal_json = create_goal_json()
        cycles_before_goal = motion_server.cycle_counter.completed_cycles

        motion_server.motion_server.run_idle_cycle()

        control_cycles = len(motion_server.control_input.applied_at_control_cycles)
        assert control_cycles > 1
        assert (
            motion_server.cycle_counter.completed_cycles
            == cycles_before_goal + 1 + control_cycles
        )

    def test_cycles_are_counted_while_a_goal_never_ends_on_its_own(
        self, motion_server: MotionServerFixture
    ):
        """
        A goal without an end motion only stops on cancel; an observer must still see
        progress while it runs.
        """
        cancel_after = CycleWatchingGoalCanceler(
            world=motion_server.executor.context.world,
            cycle_counter=motion_server.cycle_counter,
            action_server=motion_server.action_server,
            ticks_until_cancel=5,
        )
        motion_server.control_loop.inputs.synchronizers.append(cancel_after)
        motion_server.action_server.goal_json = create_goal_json(seconds=1000.0)

        motion_server.motion_server.run_idle_cycle()

        assert cancel_after.observed_ticks >= 5
        assert motion_server.action_server.outcome == GoalOutcome.CANCELED


class TestWorldUpdatesDuringGoals:
    """
    State updates of other processes are applied while a goal runs, but a model
    modification invalidates the compiled motion statechart, so it terminates the motion
    instead of being applied under it.
    """

    def test_state_updates_are_applied_once_per_control_cycle(
        self, motion_server: MotionServerFixture
    ):
        motion_server.action_server.goal_json = create_goal_json()

        motion_server.motion_server.run_idle_cycle()

        control_cycles = len(motion_server.control_input.applied_at_control_cycles)
        assert control_cycles > 1
        assert (
            motion_server.world_updates.applied_state_update_batches == control_cycles
        )

    def test_the_whole_buffer_is_not_applied_while_a_goal_runs(
        self, motion_server: MotionServerFixture
    ):
        motion_server.action_server.goal_json = create_goal_json()

        motion_server.motion_server.run_idle_cycle()

        assert motion_server.world_updates.applied_batches == 1

    def test_a_pending_model_change_terminates_the_goal(
        self, motion_server: MotionServerFixture
    ):
        inject_modification = PendingModelChangeInjector(
            world=motion_server.executor.context.world,
            world_updates=motion_server.world_updates,
            cycles_until_model_change=5,
        )
        motion_server.control_loop.inputs.synchronizers.append(inject_modification)
        motion_server.action_server.goal_json = create_goal_json(seconds=1000.0)

        motion_server.motion_server.run_idle_cycle()

        assert motion_server.action_server.outcome == GoalOutcome.ABORTED
        control_cycles = len(motion_server.control_input.applied_at_control_cycles)
        assert control_cycles < 10, "the goal ran on instead of terminating promptly"

    def test_the_buffered_modification_is_applied_by_the_next_idle_cycle(
        self, motion_server: MotionServerFixture
    ):
        inject_modification = PendingModelChangeInjector(
            world=motion_server.executor.context.world,
            world_updates=motion_server.world_updates,
            cycles_until_model_change=5,
        )
        motion_server.control_loop.inputs.synchronizers.append(inject_modification)
        motion_server.action_server.goal_json = create_goal_json(seconds=1000.0)
        motion_server.motion_server.run_idle_cycle()
        assert motion_server.world_updates.has_pending_model_change

        motion_server.motion_server.run_idle_cycle()

        assert not motion_server.world_updates.has_pending_model_change


# %% waiting for the world the goal was built on


class TestWaitingForTheWorldOfTheClient:
    """
    A goal refers to a world its client already changed, so Giskard executes it only
    once that change reached the world it controls.
    """

    def test_a_goal_that_requires_nothing_is_not_delayed(
        self, motion_server: MotionServerFixture
    ):
        motion_server.action_server.goal_json = create_goal_json()

        motion_server.motion_server.run_idle_cycle()

        assert motion_server.action_server.outcome == GoalOutcome.SUCCEEDED
        assert motion_server.world_updates.awaited_positions == []

    def test_a_goal_waits_until_the_change_it_requires_arrived(
        self, motion_server: MotionServerFixture
    ):
        position = StreamPosition(
            origin=MetaData(node_name="client", process_id=1), sequence_number=4
        )
        motion_server.world_updates.drains_until_caught_up = 3
        motion_server.action_server.goal_json = create_goal_json(
            required_position=position
        )

        motion_server.motion_server.run_idle_cycle()

        assert motion_server.action_server.outcome == GoalOutcome.SUCCEEDED
        assert motion_server.world_updates.awaited_positions[0] == position
        assert motion_server.world_updates.applied_batches == 3

    def test_the_goal_is_not_compiled_before_the_change_arrived(
        self, motion_server: MotionServerFixture
    ):
        """
        Compiling resolves the entities the goal refers to against the world, so a goal
        compiled too early would refer to entities the change was supposed to bring.
        """
        motion_server.world_updates.drains_until_caught_up = 3
        motion_server.action_server.goal_json = create_goal_json(
            required_position=StreamPosition(
                origin=MetaData(node_name="client", process_id=1), sequence_number=4
            )
        )

        motion_server.motion_server.run_idle_cycle()

        waiting_rounds = len(motion_server.world_updates.awaited_positions)
        assert waiting_rounds > 0, "the goal did not wait at all"
        assert (
            motion_server.world_updates.compiled_while_waiting
            == [False] * waiting_rounds
        )

    def test_a_change_that_never_arrives_aborts_the_goal(
        self, motion_server: MotionServerFixture
    ):
        motion_server.motion_server.world_update_timeout = 0.2
        motion_server.action_server.goal_json = create_goal_json(
            required_position=StreamPosition(
                origin=MetaData(node_name="client", process_id=1), sequence_number=4
            )
        )

        motion_server.motion_server.run_idle_cycle()

        assert motion_server.action_server.outcome == GoalOutcome.ABORTED
        result = json.loads(motion_server.action_server.sent_results[0].result)
        error = from_json(result["error"])
        assert isinstance(error, RequiredWorldUpdateNotReceivedError)
        assert error.publisher_name == "client"
        assert error.awaited_sequence_number == 4
