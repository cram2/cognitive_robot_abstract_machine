from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from math import ceil

import numpy as np
import pytest

from cramph.node import EndedByOwner
from cramph.data_types import ObservationStateValues
from giskardpy.motion_statechart.error_signals import (
    time_derivative_from_joint_motion,
)
from cramph.exceptions import CyclicNodeDependencyError, PrerequisiteNotExpandedError
from giskardpy.motion_statechart.exceptions import NoProgressError
from giskardpy.motion_statechart.goals.base_pose_goals import DifferentialDriveBaseGoal
from cramph.composites import Parallel, Sequence
from giskardpy.motion_statechart.graph_node import EndMotion, MotionStatechartNode
from cramph.node import NodeArtifacts
from cramph.monitors import CountSimulationTimeSeconds
from giskardpy.motion_statechart.monitors.progress_monitors import (
    AnyMonitoredTaskShortOfItsGoal,
    NoMonitoredTaskApproachingItsGoal,
    StillProgressing,
)
from cramph.statechart import Statechart
from cramph.nodes_for_testing import ConstFalseNode
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPosition,
    CartesianPositionTrajectory,
    CartesianPositionVelocityLimit,
)
from krrood.symbolic_math.symbolic_math import Scalar
from semantic_digital_twin.robots.robot_parts import MobileBase
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from giskardpy.motion_control import MotionControl
from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor
from semantic_digital_twin.world_description.connections import OmniDrive
from giskardpy.motion_statechart.context import MotionControlContext
from ..motion_control_context import add_to_statechart_with_motion_control

# %% helpers

# Simulated time without progress before a watched node counts as stalled. Short so
# that a test waiting it out stays fast.
STALL_TIMEOUT = timedelta(seconds=0.2)

JITTER_POSITIONS = (0.0, 0.001)
"""
The x positions, in meters, a jittering bot is put back to on alternate control cycles.
"""


def unreachable_arm_goal(world: World) -> CartesianPosition:
    """
    A goal several meters away from a PR2 gripper, so the arm extends to its joint
    limits and then stops converging.
    """
    return CartesianPosition(
        root_link=world.get_kinematic_structure_entity_by_name("base_footprint"),
        tip_link=world.get_kinematic_structure_entity_by_name("r_gripper_tool_frame"),
        goal_point=Point3(
            5,
            0,
            0,
            reference_frame=world.get_kinematic_structure_entity_by_name(
                "base_footprint"
            ),
        ),
    )


def tick_until_end_recording(
    executor: StatechartExecutor,
    motion_statechart: Statechart,
    nodes: list[MotionStatechartNode],
    maximum_cycles: int = 2000,
) -> dict[MotionStatechartNode, list[float]]:
    """
    Tick until the motion ends, recording the observation state of `nodes` each cycle.

    :param executor: The compiled executor to tick.
    :param motion_statechart: The statechart being ticked.
    :param nodes: The nodes whose observation states are recorded.
    :param maximum_cycles: Safety bound so a non-terminating motion fails the test.
    :return: The recorded observation states per node.
    """
    recorded = {node: [] for node in nodes}
    for _ in range(maximum_cycles):
        executor.tick()
        for node in nodes:
            recorded[node].append(motion_statechart.observation_state[node])
        if motion_statechart.is_ended():
            return recorded
    raise TimeoutError("motion never ended")


@dataclass(eq=False, repr=False, kw_only=True)
class NodeWithDeclaredDependencies(EndedByOwner, MotionStatechartNode):
    """
    Node that declares whichever build dependencies a test needs, so dependency ordering
    and cycle detection can be exercised without a real task.
    """

    dependencies: list[MotionStatechartNode] = field(default_factory=list)
    """
    The nodes this node claims to depend on.
    """

    built_after: list[str] = field(default_factory=list)
    """
    Shared list every instance appends its name to when it is built.
    """

    @property
    def prerequisite_nodes(self) -> list[MotionStatechartNode]:
        return self.dependencies

    def build_artifacts(self) -> NodeArtifacts:
        self.built_after.append(self.name)
        return NodeArtifacts(observation=Scalar.const_true())


# %% detecting a stall


class TestStallDetection:
    def test_stall_cancels_the_motion(self, pr2_world_state_reset: World):
        """
        An arm that has extended as far as it can stops closing on its goal, so the
        motion is cancelled instead of running forever.
        """
        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_state_reset),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        goal = unreachable_arm_goal(pr2_world_state_reset)
        motion_statechart.add_node(goal)
        motion_statechart.add_node(EndMotion.when_true(goal))
        progressing = StillProgressing(
            monitored_node=goal, timeout=timedelta(seconds=1)
        )
        motion_statechart.add_node(progressing)
        motion_statechart.add_node(progressing.cancel_motion())

        executor.compile(statechart=motion_statechart)

        with pytest.raises(NoProgressError) as exception_info:
            executor.tick_until_end(2000)

        assert goal.unique_name in str(exception_info.value)

    def test_a_task_jittering_in_place_stalls(self, cylinder_bot_world: World):
        """
        A task pushed back every cycle, as an arm held off by a collision is, keeps
        moving without getting any closer to its goal, so the motion is cancelled.
        """
        goal = CartesianPosition(
            root_link=cylinder_bot_world.root,
            tip_link=cylinder_bot_world.get_kinematic_structure_entity_by_name("bot"),
            goal_point=Point3(1, 0, 0, reference_frame=cylinder_bot_world.root),
        )
        progressing = StillProgressing(monitored_node=goal, timeout=STALL_TIMEOUT)
        motion_control = MotionControl()
        executor = StatechartExecutor(
            context=StatechartContext(world=cylinder_bot_world),
            extensions=[motion_control],
        )
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_nodes([goal, progressing])
        motion_statechart.add_node(EndMotion.when_true(goal))
        motion_statechart.add_node(progressing.cancel_motion())

        executor.compile(statechart=motion_statechart)
        drive = cylinder_bot_world.get_connections_by_type(OmniDrive)[0]
        cycles = 4 * ceil(
            STALL_TIMEOUT.total_seconds()
            / motion_control.qp_controller_config.control_time_step.total_seconds()
        )

        with pytest.raises(NoProgressError):
            for cycle in range(cycles):
                executor.tick()
                cylinder_bot_world.state[drive.x.id].position = JITTER_POSITIONS[
                    cycle % len(JITTER_POSITIONS)
                ]
                cylinder_bot_world.notify_state_change()

    def test_reachable_goal_is_never_reported_as_stalled(
        self, cylinder_bot_world: World
    ):
        """
        A goal the robot converges on reads as progressing for the whole motion.

        Every cycle must be true rather than merely not false: a node ended while it
        is still getting somewhere has to be judged, and an undecided monitor could only
        interrupt it.
        """
        executor = StatechartExecutor(
            context=StatechartContext(world=cylinder_bot_world),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        goal = CartesianPosition(
            root_link=cylinder_bot_world.root,
            tip_link=cylinder_bot_world.get_kinematic_structure_entity_by_name("bot"),
            goal_point=Point3(1, 0, 0, reference_frame=cylinder_bot_world.root),
        )
        motion_statechart.add_node(goal)
        motion_statechart.add_node(EndMotion.when_true(goal))
        progressing = StillProgressing(
            monitored_node=goal, timeout=timedelta(seconds=1)
        )
        motion_statechart.add_node(progressing)

        executor.compile(statechart=motion_statechart)
        recorded = tick_until_end_recording(executor, motion_statechart, [progressing])

        assert set(recorded[progressing]) == {ObservationStateValues.TRUE}

    def test_momentary_stall_shorter_than_the_timeout_is_tolerated(
        self, cylinder_bot_diff_world: World
    ):
        """
        A differential drive turns in place before it translates, so its position error
        stops changing for a while.

        The timeout is what stops that from being mistaken for a stall, so the monitor
        must have fired and still not cancelled the motion.
        """
        executor = StatechartExecutor(
            context=StatechartContext(world=cylinder_bot_diff_world),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        goal = DifferentialDriveBaseGoal(
            mobile_base=cylinder_bot_diff_world.get_semantic_annotations_by_type(
                MobileBase
            )[0],
            goal_pose=Pose.from_xyz_rpy(
                x=1, y=1, reference_frame=cylinder_bot_diff_world.root
            ),
        )
        motion_statechart.add_node(goal)
        motion_statechart.add_node(EndMotion.when_true(goal))
        progressing = StillProgressing(
            monitored_node=goal, timeout=timedelta(seconds=100)
        )
        motion_statechart.add_node(progressing)
        motion_statechart.add_node(progressing.cancel_motion())

        executor.compile(statechart=motion_statechart)
        not_approaching = [
            node
            for node in progressing.nodes
            if isinstance(node, NoMonitoredTaskApproachingItsGoal)
        ]
        recorded = tick_until_end_recording(
            executor, motion_statechart, not_approaching + [progressing]
        )

        assert ObservationStateValues.FALSE not in recorded[progressing]
        assert any(
            ObservationStateValues.TRUE in recorded[monitor]
            for monitor in not_approaching
        ), "no task ever paused, so the timeout was never what prevented the cancel"

    def test_stall_inside_a_sequence_names_the_stuck_step(
        self, pr2_world_state_reset: World
    ):
        """
        Watching each converging task separately means the stuck step of a sequence is
        reported, not just the sequence as a whole.
        """
        base_footprint = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "r_gripper_tool_frame"
        )
        reachable = CartesianPosition(
            name="reachable",
            root_link=base_footprint,
            tip_link=tip,
            goal_point=Point3(0, 0, 0.1, reference_frame=tip),
        )
        unreachable = CartesianPosition(
            name="unreachable",
            root_link=base_footprint,
            tip_link=tip,
            goal_point=Point3(5, 0, 0, reference_frame=base_footprint),
        )
        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_state_reset),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        sequence = Sequence(nodes=[reachable, unreachable])
        motion_statechart.add_node(sequence)
        motion_statechart.add_node(EndMotion.when_true(sequence))
        progressing = StillProgressing(
            monitored_node=sequence, timeout=timedelta(seconds=1)
        )
        motion_statechart.add_node(progressing)
        motion_statechart.add_node(progressing.cancel_motion())

        executor.compile(statechart=motion_statechart)

        assert progressing.monitored_tasks == [reachable, unreachable]

        with pytest.raises(NoProgressError) as exception_info:
            executor.tick_until_end(2000)

        assert progressing.stalled_tasks == [unreachable]
        assert unreachable.unique_name in str(exception_info.value)
        assert reachable.unique_name not in str(exception_info.value)

    def test_a_stall_names_only_the_stuck_task_not_one_holding_its_goal(
        self, pr2_world_state_reset: World
    ):
        """
        A task that holds its goal while another one is stuck has not stopped
        approaching anything, so the stall must not blame it.
        """
        base_footprint = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "r_gripper_tool_frame"
        )
        other_tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "l_gripper_tool_frame"
        )
        holding = CartesianPosition(
            name="holding",
            root_link=base_footprint,
            tip_link=other_tip,
            goal_point=Point3(0, 0, 0, reference_frame=other_tip),
        )
        unreachable = CartesianPosition(
            name="unreachable",
            root_link=base_footprint,
            tip_link=tip,
            goal_point=Point3(5, 0, 0, reference_frame=base_footprint),
        )
        executor = StatechartExecutor(
            context=StatechartContext(world=pr2_world_state_reset),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        parallel = Parallel(nodes=[holding, unreachable])
        motion_statechart.add_node(parallel)
        motion_statechart.add_node(EndMotion.when_true(parallel))
        progressing = StillProgressing(
            monitored_node=parallel, timeout=timedelta(seconds=1)
        )
        motion_statechart.add_node(progressing)
        motion_statechart.add_node(progressing.cancel_motion())

        executor.compile(statechart=motion_statechart)

        with pytest.raises(NoProgressError) as exception_info:
            executor.tick_until_end(2000)

        assert progressing.stalled_tasks == [unreachable]
        assert holding.unique_name not in str(exception_info.value)

    def test_a_motion_whose_running_tasks_all_reached_their_goals_is_not_stalled(
        self, cylinder_bot_world: World
    ):
        """
        A task at its goal has nothing left to approach, so it must not keep alive the
        guard that stops finished work from reading as a stall.

        Nothing ends a task for reaching its goal, so a task that arrived keeps running,
        and a motion also holds invariants that sit at zero error for its whole length.
        Counting either as work still in progress makes every wait after the last goal
        was reached a stall.
        """
        bot = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
        arrived = CartesianPosition(
            root_link=cylinder_bot_world.root,
            tip_link=bot,
            goal_point=Point3(0, 0, 0, reference_frame=bot),
        )
        progressing = StillProgressing(monitored_node=arrived, timeout=STALL_TIMEOUT)
        motion_control = MotionControl()
        executor = StatechartExecutor(
            context=StatechartContext(world=cylinder_bot_world),
            extensions=[motion_control],
        )
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_nodes([arrived, progressing])
        motion_statechart.add_node(progressing.cancel_motion())

        executor.compile(statechart=motion_statechart)
        for _ in range(
            2
            * ceil(
                STALL_TIMEOUT.total_seconds()
                / motion_control.qp_controller_config.control_time_step.total_seconds()
            )
        ):
            executor.tick()

        assert arrived.observation_state == ObservationStateValues.TRUE
        assert progressing.observation_state == ObservationStateValues.TRUE

    def test_stall_time_does_not_accumulate_before_the_goal_starts(
        self, cylinder_bot_world: World
    ):
        """
        Nothing is converging before the watched task starts, which must not be mistaken
        for a stall.
        """
        executor = StatechartExecutor(
            context=StatechartContext(world=cylinder_bot_world),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        goal = CartesianPosition(
            root_link=cylinder_bot_world.root,
            tip_link=cylinder_bot_world.get_kinematic_structure_entity_by_name("bot"),
            goal_point=Point3(1, 0, 0, reference_frame=cylinder_bot_world.root),
        )
        blocker = ConstFalseNode()
        motion_statechart.add_nodes([goal, blocker])
        # The goal only starts once the blocker is true, which never happens.
        goal.start_condition = blocker.observes_true
        progressing = StillProgressing(
            monitored_node=goal, timeout=timedelta(seconds=0.5)
        )
        motion_statechart.add_node(progressing)

        executor.compile(statechart=motion_statechart)
        for _ in range(200):
            executor.tick()

        assert (
            motion_statechart.observation_state[progressing]
            == ObservationStateValues.TRUE
        )

    def test_the_stall_timer_counts_the_whole_timeout(self, cylinder_bot_world: World):
        """
        The timeout reaches the timer as simulated seconds, so neither the days of a
        long window nor the fraction of a sub-second one is lost on the way.
        """
        timeout = timedelta(days=1, milliseconds=500)
        goal = CartesianPosition(
            root_link=cylinder_bot_world.root,
            tip_link=cylinder_bot_world.get_kinematic_structure_entity_by_name("bot"),
            goal_point=Point3(1, 0, 0, reference_frame=cylinder_bot_world.root),
        )
        progressing = StillProgressing(monitored_node=goal, timeout=timeout)
        executor = StatechartExecutor(
            context=StatechartContext(world=cylinder_bot_world),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_nodes([goal, progressing])

        executor.compile(statechart=motion_statechart)

        timer = [
            node
            for node in progressing.nodes
            if isinstance(node, CountSimulationTimeSeconds)
        ]
        assert [node.seconds for node in timer] == [timeout.total_seconds()]

    def test_one_monitor_watches_every_task_beneath_the_watched_node(
        self, cylinder_bot_world: World
    ) -> None:
        """
        A plan holds dozens of converging tasks, and a monitor per task made the
        statechart several times larger than the plan itself.
        """
        bot = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
        tasks = [
            CartesianPosition(
                root_link=cylinder_bot_world.root,
                tip_link=bot,
                goal_point=Point3(x, 0, 0, reference_frame=cylinder_bot_world.root),
            )
            for x in (1, 2, 3)
        ]
        sequence = Sequence(nodes=tasks)
        progressing = StillProgressing(monitored_node=sequence, timeout=STALL_TIMEOUT)
        executor = StatechartExecutor(
            context=StatechartContext(world=cylinder_bot_world),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_nodes([sequence, progressing])
        motion_statechart.add_node(EndMotion.when_true(sequence))

        executor.compile(statechart=motion_statechart)

        assert [type(node) for node in progressing.nodes] == [
            CountSimulationTimeSeconds,
            NoMonitoredTaskApproachingItsGoal,
            AnyMonitoredTaskShortOfItsGoal,
        ]
        assert progressing.nodes[1].monitored_tasks == tasks
        assert progressing.monitored_tasks == tasks

    def test_a_task_at_its_goal_is_not_approaching_one(self, cylinder_bot_world: World):
        """
        A task that is exactly at its goal has nothing left to approach, so it must not
        read as the progress that keeps a stalled motion from being given up on.
        """
        bot = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
        arrived = CartesianPosition(
            name="arrived",
            root_link=cylinder_bot_world.root,
            tip_link=bot,
            goal_point=Point3(0, 0, 0, reference_frame=bot),
        )
        monitor = NoMonitoredTaskApproachingItsGoal(monitored_tasks=[arrived])
        executor = StatechartExecutor(
            context=StatechartContext(world=cylinder_bot_world),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_node(arrived)
        motion_statechart.add_node(monitor)
        motion_statechart.add_node(EndMotion())

        executor.compile(statechart=motion_statechart)
        executor.tick()
        executor.tick()

        assert arrived.error_signal.evaluate()[0] == pytest.approx(0)
        assert monitor.observation_state == ObservationStateValues.TRUE


# %% the rule that decides whether a task approaches its goal


CONVERGENCE_RATE = 0.5
"""
A minimum convergence rate other than the default, so it is seen to be what the rule
follows.
"""

TASK_THRESHOLD = 0.1
"""
The error below which the watched task would count as having reached its goal.
"""

GOAL_X = 10.0
"""
The x position of the watched task's goal, far enough away that the task never reaches
it.
"""


@dataclass
class ScriptedErrorRun:
    """
    A :class:`NoMonitoredTaskApproachingItsGoal` watching tasks whose errors a test sets
    on every control cycle, by putting the bot where it should be.
    """

    world: World
    """
    The world of the bot whose distances to the goals are the tasks' errors.
    """

    goal_xs: list[float] = field(default_factory=lambda: [GOAL_X])
    """
    The x position of each watched task's goal.
    """

    tasks: list[CartesianPosition] = field(init=False)
    """
    The watched tasks, one per goal in :attr:`goal_xs`.
    """

    monitor: NoMonitoredTaskApproachingItsGoal = field(init=False)
    """
    The monitor under test.
    """

    context: StatechartContext = field(init=False)
    """
    The context the motion runs in.
    """

    executor: StatechartExecutor = field(init=False)
    """
    The executor ticking the motion.
    """

    def __post_init__(self):
        self.tasks = [
            CartesianPosition(
                name=f"goal at {goal_x}",
                root_link=self.world.root,
                tip_link=self.world.get_kinematic_structure_entity_by_name("bot"),
                goal_point=Point3(goal_x, 0, 0, reference_frame=self.world.root),
                threshold=TASK_THRESHOLD,
            )
            for goal_x in self.goal_xs
        ]
        self.monitor = NoMonitoredTaskApproachingItsGoal(
            monitored_tasks=self.tasks, minimum_convergence_rate=CONVERGENCE_RATE
        )
        self.executor = StatechartExecutor(
            context=StatechartContext(world=self.world),
            extensions=[MotionControl()],
        )
        self.context = self.executor.context
        motion_statechart = Statechart(context=self.context)
        motion_statechart.add_nodes([*self.tasks, self.monitor])
        motion_statechart.add_node(EndMotion.when_all_true(self.tasks))
        self.executor.compile(statechart=motion_statechart)

    @property
    def required_fall_per_cycle(self) -> float:
        """
        :return: How far a task's error has to fall on every control cycle.
        """
        return (
            CONVERGENCE_RATE
            * TASK_THRESHOLD
            * self.context.require_extension(
                MotionControlContext
            ).qp_controller_config.control_time_step.total_seconds()
        )

    def observations_while_the_error_is(
        self, errors: list[float]
    ) -> list[ObservationStateValues]:
        """
        :param errors: The first task's error on each control cycle, the bot being
            short of its goal.
        :return: What the monitor observes on each of those control cycles.
        """
        return self.observations_while_the_bot_is_at(
            [self.goal_xs[0] - error for error in errors]
        )

    def observations_while_the_bot_is_at(
        self, positions: list[float]
    ) -> list[ObservationStateValues]:
        """
        :param positions: The bot's x position on each control cycle.
        :return: What the monitor observes on each of those control cycles.
        """
        drive = self.world.get_connections_by_type(OmniDrive)[0]
        observations = []
        for position in positions:
            self.world.state[drive.x.id].position = position
            self.world.notify_state_change()
            self.executor.tick()
            observations.append(self.monitor.observation_state)
        return observations


class TestConvergenceRule:
    @pytest.mark.parametrize(
        "fall_factor, observed",
        [(0.8, ObservationStateValues.TRUE), (1.2, ObservationStateValues.FALSE)],
        ids=["too-slow", "fast-enough"],
    )
    def test_an_error_has_to_fall_by_the_minimum_rate(
        self, cylinder_bot_world: World, fall_factor, observed
    ):
        run = ScriptedErrorRun(cylinder_bot_world)
        start = 1.0

        observations = run.observations_while_the_error_is(
            [start, start - fall_factor * run.required_fall_per_cycle]
        )

        assert observations == [ObservationStateValues.FALSE, observed]

    def test_falling_back_after_a_detour_is_not_progress_until_it_beats_the_start(
        self, cylinder_bot_world: World
    ):
        """
        An error that rises and then falls back to where it was has made no progress,
        and only falling below that by the rate for all the time since counts.
        """
        run = ScriptedErrorRun(cylinder_bot_world)
        start = 1.0
        beyond = start - 4 * run.required_fall_per_cycle

        observations = run.observations_while_the_error_is(
            [start, start + 0.5, start, beyond]
        )

        assert observations == [
            ObservationStateValues.FALSE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.FALSE,
        ]

    def test_a_restarted_task_is_judged_afresh(self, cylinder_bot_world: World):
        """
        A task that starts again has nothing to be compared with from its last run.
        """
        run = ScriptedErrorRun(cylinder_bot_world)
        assert run.observations_while_the_error_is([1.0, 1.0]) == [
            ObservationStateValues.FALSE,
            ObservationStateValues.TRUE,
        ]

        run.monitor.on_start()

        assert run.observations_while_the_error_is([5.0]) == [
            ObservationStateValues.FALSE
        ]

    def test_a_task_cycling_through_its_goal_makes_no_progress(
        self, cylinder_bot_world: World
    ) -> None:
        """
        A task that falls into its goal, is pushed back out and falls in again, over and
        over, gets no closer than it already was, so it cannot hide that another task is
        stuck.
        """
        run = ScriptedErrorRun(cylinder_bot_world, goal_xs=[GOAL_X, 0.0])
        threshold_of_the_cycling_task = run.tasks[1].threshold
        cycle = [
            3 * threshold_of_the_cycling_task,
            2 * threshold_of_the_cycling_task,
            threshold_of_the_cycling_task / 2,
        ]

        run.observations_while_the_bot_is_at(cycle)
        later_cycles = run.observations_while_the_bot_is_at(cycle + cycle)

        assert later_cycles == [ObservationStateValues.TRUE] * len(later_cycles)

    def test_several_tasks_are_stalled_only_while_none_of_them_approaches(
        self, cylinder_bot_world: World
    ) -> None:
        """
        Driving towards one goal and away from another is progress, and standing still
        is progress for neither, which the one monitor watching both has to tell apart.
        """
        run = ScriptedErrorRun(cylinder_bot_world, goal_xs=[GOAL_X, -GOAL_X])
        step = 4 * run.required_fall_per_cycle

        moving = run.observations_while_the_bot_is_at([0.0, step, 2 * step])
        standing = run.observations_while_the_bot_is_at([2 * step])

        assert moving == [
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
        ]
        assert standing == [ObservationStateValues.TRUE]

    def test_the_stalled_tasks_are_those_not_approaching_their_goal(
        self, cylinder_bot_world: World
    ) -> None:
        """
        The monitor names the tasks it judges stuck, so a stall can blame them and not
        the tasks that still get closer.
        """
        run = ScriptedErrorRun(cylinder_bot_world, goal_xs=[GOAL_X, -GOAL_X])
        approached, receding = run.tasks
        step = 4 * run.required_fall_per_cycle

        run.observations_while_the_bot_is_at([0.0, step, 2 * step])

        assert run.monitor.stalled_tasks == [receding]


# %% differentiating with respect to time


class TestTimeDerivative:
    def test_expression_without_joints_has_no_rate(self):
        """
        An error that does not depend on any joint cannot change through robot motion.
        """
        assert time_derivative_from_joint_motion(Scalar(3.0)).evaluate()[0] == 0.0

    def test_an_expression_of_a_joint_changes_with_its_velocity(
        self, cylinder_bot_world: World
    ):
        """
        An error that depends on a joint changes as fast as the joint moves, scaled by
        how strongly it depends on it.
        """
        drive = cylinder_bot_world.get_connections_by_type(OmniDrive)[0]
        scale = 2.0
        velocity = 0.5
        rate = time_derivative_from_joint_motion(drive.x.variables.position * scale)

        cylinder_bot_world.state[drive.x.id].velocity = velocity
        cylinder_bot_world.notify_state_change()

        assert rate.evaluate()[0] == pytest.approx(scale * velocity)


# %% error drives the observation


class TestErrorDrivesObservation:
    def test_error_is_the_distance_to_the_goal(self, cylinder_bot_world: World):
        """
        CartesianPosition reports the distance between tip and goal as its error.
        """
        bot = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
        goal_point = Point3(1, 0, 0, reference_frame=cylinder_bot_world.root)
        goal = CartesianPosition(
            root_link=cylinder_bot_world.root, tip_link=bot, goal_point=goal_point
        )

        add_to_statechart_with_motion_control(goal, cylinder_bot_world)
        artifacts = goal.build()

        tip_position = cylinder_bot_world.compute_forward_kinematics_np(
            cylinder_bot_world.root, bot
        )[:3, 3]
        expected = np.linalg.norm(goal_point.to_np()[:3] - tip_position)
        assert artifacts.error.evaluate()[0] == pytest.approx(expected)

    def test_observation_follows_from_the_error(self, cylinder_bot_world: World):
        """
        The success condition is derived from the error and the threshold rather than
        written out a second time.
        """
        bot = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
        goal = CartesianPosition(
            root_link=cylinder_bot_world.root,
            tip_link=bot,
            goal_point=Point3(1, 0, 0, reference_frame=cylinder_bot_world.root),
        )

        add_to_statechart_with_motion_control(goal, cylinder_bot_world)
        artifacts = goal.build()

        expected = (artifacts.error <= goal.threshold).evaluate()[0]
        assert artifacts.observation.evaluate()[0] == expected

    def test_normalized_error_is_one_at_the_threshold(self, cylinder_bot_world: World):
        """
        Dividing by the threshold makes errors of different tasks comparable, with 1
        meaning "exactly at the threshold".
        """
        bot = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
        goal = CartesianPosition(
            root_link=cylinder_bot_world.root,
            tip_link=bot,
            goal_point=Point3(1, 0, 0, reference_frame=cylinder_bot_world.root),
        )
        executor = StatechartExecutor(
            context=StatechartContext(world=cylinder_bot_world),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_node(goal)
        motion_statechart.add_node(EndMotion.when_true(goal))
        executor.compile(statechart=motion_statechart)

        raw_error = goal.error_signal.evaluate()[0]
        assert goal.normalized_error.evaluate()[0] == pytest.approx(
            raw_error / goal.threshold
        )


# %% a task that has reached its goal

GOAL_DISTANCE = 1.0
"""
How far from where the cylinder bot starts the task's goal lies, in meters.
"""

THRESHOLD_WIDER_THAN_THE_GOAL_DISTANCE = 2.0
"""
A threshold the bot starts inside of, so the task counts as having reached its goal from
the first cycle, yet small enough that the bot moving towards the goal changes the error
by far more than the minimum convergence rate.
"""

MOTION_DURATION = timedelta(seconds=1)
"""
How long the bot is driven, so its error keeps changing throughout.
"""


def position_task_at_its_goal(world: World) -> CartesianPosition:
    """
    A position task already within its threshold, whose error keeps changing because it
    still drives the bot.
    """
    return CartesianPosition(
        root_link=world.root,
        tip_link=world.get_kinematic_structure_entity_by_name("bot"),
        goal_point=Point3(GOAL_DISTANCE, 0, 0, reference_frame=world.root),
        threshold=THRESHOLD_WIDER_THAN_THE_GOAL_DISTANCE,
    )


def trajectory_task_at_its_goal(world: World) -> CartesianPositionTrajectory:
    """
    A trajectory task already within its threshold, whose error is sampled rather than
    differentiated and keeps changing as the bot follows the trajectory.
    """
    return CartesianPositionTrajectory(
        root_link=world.root,
        tip_link=world.get_kinematic_structure_entity_by_name("bot"),
        goal_points=[
            Point3(GOAL_DISTANCE * x / 100, 0, 0, reference_frame=world.root)
            for x in range(100)
        ],
        threshold=THRESHOLD_WIDER_THAN_THE_GOAL_DISTANCE,
    )


class TestTaskAtItsGoal:
    @pytest.mark.parametrize(
        "task_at_its_goal", [position_task_at_its_goal, trajectory_task_at_its_goal]
    )
    def test_a_task_at_its_goal_is_not_approaching_it(
        self, cylinder_bot_world: World, task_at_its_goal
    ):
        """
        A task holding its goal, like a grip held while something else moves, can see
        its error move within the threshold for as long as the motion runs.

        That movement is not progress, so it must not keep a stuck motion from being
        given up on.
        """
        task = task_at_its_goal(cylinder_bot_world)
        executor = StatechartExecutor(
            context=StatechartContext(world=cylinder_bot_world),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_node(task)
        not_approaching = NoMonitoredTaskApproachingItsGoal(monitored_tasks=[task])
        motion_statechart.add_node(not_approaching)
        timer = CountSimulationTimeSeconds(seconds=MOTION_DURATION.total_seconds())
        motion_statechart.add_node(timer)
        motion_statechart.add_node(EndMotion.when_true(timer))

        executor.compile(statechart=motion_statechart)
        recorded = tick_until_end_recording(
            executor, motion_statechart, [not_approaching]
        )

        assert ObservationStateValues.FALSE not in recorded[not_approaching]


# %% nodes with nothing converging beneath them


class TestNothingToConverge:
    def test_a_node_that_never_converges_stalls_after_the_timeout(
        self, cylinder_bot_world: World
    ):
        """
        A velocity limit enforces an invariant rather than closing on a goal, so nothing
        beneath it can approach one and the timeout alone decides when to give up on it.
        """
        bot = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
        limit = CartesianPositionVelocityLimit(
            root_link=cylinder_bot_world.root, tip_link=bot
        )
        progressing = StillProgressing(monitored_node=limit, timeout=STALL_TIMEOUT)
        motion_control = MotionControl()
        executor = StatechartExecutor(
            context=StatechartContext(world=cylinder_bot_world),
            extensions=[motion_control],
        )
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_node(limit)
        motion_statechart.add_node(progressing)
        motion_statechart.add_node(EndMotion())

        executor.compile(statechart=motion_statechart)

        # The first tick starts the stall timer, so the observation only becomes
        # measurable on the tick after it.
        executor.tick()
        executor.tick()
        assert progressing.observation_state == ObservationStateValues.TRUE

        for _ in range(
            ceil(
                STALL_TIMEOUT.total_seconds()
                / motion_control.qp_controller_config.control_time_step.total_seconds()
            )
        ):
            executor.tick()
        assert progressing.observation_state == ObservationStateValues.FALSE


# %% dependency ordering


class TestNodeDependencies:
    def test_a_dependency_is_built_first_even_when_added_later(
        self, cylinder_bot_world: World
    ):
        """
        Build order follows declared dependencies, not the order nodes were added.
        """
        built_after: list[str] = []
        dependency = NodeWithDeclaredDependencies(
            name="dependency", built_after=built_after
        )
        dependent = NodeWithDeclaredDependencies(
            name="dependent", dependencies=[dependency], built_after=built_after
        )
        executor = StatechartExecutor(
            context=StatechartContext(world=cylinder_bot_world),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_node(dependent)
        motion_statechart.add_node(dependency)
        motion_statechart.add_node(EndMotion())

        executor.compile(statechart=motion_statechart)

        assert built_after == ["dependency", "dependent"]

    def test_watching_a_goal_that_did_not_join_yet_is_rejected(
        self, cylinder_bot_world: World
    ) -> None:
        """
        The monitor finds the tasks of the goal it watches when it expands, which is
        when it joins, so the goal has to join first.
        """
        bot = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
        goal = CartesianPosition(
            root_link=cylinder_bot_world.root,
            tip_link=bot,
            goal_point=Point3(1, 0, 0, reference_frame=cylinder_bot_world.root),
        )
        sequence = Sequence(nodes=[goal])
        progressing = StillProgressing(
            monitored_node=sequence, timeout=timedelta(seconds=1)
        )

        executor = StatechartExecutor(
            context=StatechartContext(world=cylinder_bot_world),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)

        with pytest.raises(PrerequisiteNotExpandedError):
            motion_statechart.add_node(progressing)

    def test_a_dependency_cycle_is_reported(self, cylinder_bot_world: World):
        """
        Nodes that depend on each other cannot be ordered, which must be said plainly
        instead of recursing forever.
        """
        first = NodeWithDeclaredDependencies(name="first")
        second = NodeWithDeclaredDependencies(name="second", dependencies=[first])
        first.dependencies.append(second)
        executor = StatechartExecutor(
            context=StatechartContext(world=cylinder_bot_world),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_nodes([first, second])
        motion_statechart.add_node(EndMotion())

        with pytest.raises(CyclicNodeDependencyError):
            executor.compile(statechart=motion_statechart)


# %% errors kept by the task itself


class TestTrajectoryProgress:
    def test_a_trajectory_being_followed_is_progressing(
        self, cylinder_bot_world: World
    ):
        """
        A trajectory task knows how far it has come only from its own bookkeeping, and
        following it reads as progress for the whole motion.
        """
        bot = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
        trajectory = CartesianPositionTrajectory(
            root_link=cylinder_bot_world.root,
            tip_link=bot,
            goal_points=[
                Point3(x / 100, 0, 0, reference_frame=cylinder_bot_world.root)
                for x in range(100)
            ],
        )
        executor = StatechartExecutor(
            context=StatechartContext(world=cylinder_bot_world),
            extensions=[MotionControl()],
        )
        motion_statechart = Statechart(context=executor.context)
        motion_statechart.add_node(trajectory)
        motion_statechart.add_node(EndMotion.when_true(trajectory))
        progressing = StillProgressing(
            monitored_node=trajectory, timeout=timedelta(seconds=1)
        )
        motion_statechart.add_node(progressing)

        executor.compile(statechart=motion_statechart)

        recorded = tick_until_end_recording(executor, motion_statechart, [progressing])
        assert ObservationStateValues.FALSE not in recorded[progressing]
