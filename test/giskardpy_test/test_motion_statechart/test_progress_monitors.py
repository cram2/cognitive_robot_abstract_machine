from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
from typing_extensions import List

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.error_signals import (
    SampledErrorSignal,
    SymbolicErrorSignal,
    time_derivative_from_joint_motion,
)
from giskardpy.motion_statechart.exceptions import (
    CyclicNodeDependencyError,
    NoConvergingTaskError,
    NoProgressError,
)
from giskardpy.motion_statechart.goals.cartesian_goals import DifferentialDriveBaseGoal
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import (
    EndMotion,
    MotionStatechartNode,
    NodeArtifacts,
)
from giskardpy.motion_statechart.monitors.progress_monitors import (
    NotApproachingGoal,
    ProgressStalled,
)
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.nodes_for_testing.nodes_for_testing import (
    ConstFalseNode,
)
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPosition,
    CartesianPositionTrajectory,
    CartesianPositionVelocityLimit,
)
from krrood.symbolic_math.symbolic_math import Scalar
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World

# %% helpers


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
    executor: Executor,
    motion_statechart: MotionStatechart,
    nodes: List[MotionStatechartNode],
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
        if motion_statechart.is_end_motion():
            return recorded
    raise TimeoutError("motion never ended")


@dataclass(eq=False, repr=False)
class NodeWithDeclaredDependencies(MotionStatechartNode):
    """
    Node that declares whichever build dependencies a test needs, so dependency ordering
    and cycle detection can be exercised without a real task.
    """

    dependencies: List[MotionStatechartNode] = field(default_factory=list, kw_only=True)
    """
    The nodes this node claims to depend on.
    """

    built_after: List[str] = field(default_factory=list, kw_only=True)
    """
    Shared list every instance appends its name to when it is built.
    """

    @property
    def prerequisite_nodes(self) -> List[MotionStatechartNode]:
        return self.dependencies

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        self.built_after.append(self.name)
        return NodeArtifacts(observation=Scalar.const_true())


# %% detecting a stall


class TestStallDetection:

    def test_stall_cancels_the_motion(self, pr2_world_state_reset: World):
        """
        An arm that has extended as far as it can stops closing on its goal, so the
        motion is cancelled instead of running forever.
        """
        motion_statechart = MotionStatechart()
        goal = unreachable_arm_goal(pr2_world_state_reset)
        motion_statechart.add_node(goal)
        motion_statechart.add_node(EndMotion.when_true(goal))
        stalled = ProgressStalled(monitored_node=goal, timeout=1.0)
        motion_statechart.add_node(stalled)
        motion_statechart.add_node(stalled.cancel_motion())

        executor = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        executor.compile(motion_statechart=motion_statechart)

        with pytest.raises(NoProgressError) as exception_info:
            executor.tick_until_end(2000)

        assert goal.unique_name in str(exception_info.value)

    def test_reachable_goal_is_never_reported_as_stalled(
        self, cylinder_bot_world: World
    ):
        """
        A goal the robot converges on must complete without the monitor ever firing.
        """
        motion_statechart = MotionStatechart()
        goal = CartesianPosition(
            root_link=cylinder_bot_world.root,
            tip_link=cylinder_bot_world.get_kinematic_structure_entity_by_name("bot"),
            goal_point=Point3(1, 0, 0, reference_frame=cylinder_bot_world.root),
        )
        motion_statechart.add_node(goal)
        motion_statechart.add_node(EndMotion.when_true(goal))
        stalled = ProgressStalled(monitored_node=goal, timeout=1.0)
        motion_statechart.add_node(stalled)

        executor = Executor(MotionStatechartContext(world=cylinder_bot_world))
        executor.compile(motion_statechart=motion_statechart)
        recorded = tick_until_end_recording(executor, motion_statechart, [stalled])

        assert ObservationStateValues.TRUE not in recorded[stalled]

    def test_momentary_stall_shorter_than_the_timeout_is_tolerated(
        self, cylinder_bot_diff_world: World
    ):
        """
        A differential drive turns in place before it translates, so its position error
        stops changing for a while.

        The timeout is what stops that from being mistaken for a stall, so the monitor
        must have fired and still not cancelled the motion.
        """
        motion_statechart = MotionStatechart()
        goal = DifferentialDriveBaseGoal(
            goal_pose=Pose.from_xyz_rpy(
                x=1, y=1, reference_frame=cylinder_bot_diff_world.root
            )
        )
        motion_statechart.add_node(goal)
        motion_statechart.add_node(EndMotion.when_true(goal))
        stalled = ProgressStalled(monitored_node=goal, timeout=100.0)
        motion_statechart.add_node(stalled)
        motion_statechart.add_node(stalled.cancel_motion())

        executor = Executor(MotionStatechartContext(world=cylinder_bot_diff_world))
        executor.compile(motion_statechart=motion_statechart)
        not_approaching = [
            node for node in stalled.nodes if isinstance(node, NotApproachingGoal)
        ]
        recorded = tick_until_end_recording(
            executor, motion_statechart, not_approaching + [stalled]
        )

        assert ObservationStateValues.TRUE not in recorded[stalled]
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
        motion_statechart = MotionStatechart()
        sequence = Sequence(nodes=[reachable, unreachable])
        motion_statechart.add_node(sequence)
        motion_statechart.add_node(EndMotion.when_true(sequence))
        stalled = ProgressStalled(monitored_node=sequence, timeout=1.0)
        motion_statechart.add_node(stalled)
        motion_statechart.add_node(stalled.cancel_motion())

        executor = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        executor.compile(motion_statechart=motion_statechart)

        assert stalled.monitored_tasks == [reachable, unreachable]

        with pytest.raises(NoProgressError) as exception_info:
            executor.tick_until_end(2000)

        assert stalled.stalled_tasks == [unreachable]
        assert unreachable.unique_name in str(exception_info.value)
        assert reachable.unique_name not in str(exception_info.value)

    def test_stall_time_does_not_accumulate_before_the_goal_starts(
        self, cylinder_bot_world: World
    ):
        """
        Nothing is converging before the watched task starts, which must not be mistaken
        for a stall.
        """
        motion_statechart = MotionStatechart()
        goal = CartesianPosition(
            root_link=cylinder_bot_world.root,
            tip_link=cylinder_bot_world.get_kinematic_structure_entity_by_name("bot"),
            goal_point=Point3(1, 0, 0, reference_frame=cylinder_bot_world.root),
        )
        blocker = ConstFalseNode()
        motion_statechart.add_nodes([goal, blocker])
        # The goal only starts once the blocker is true, which never happens.
        goal.start_condition = blocker.observation_variable
        stalled = ProgressStalled(monitored_node=goal, timeout=0.5)
        motion_statechart.add_node(stalled)

        executor = Executor(MotionStatechartContext(world=cylinder_bot_world))
        executor.compile(motion_statechart=motion_statechart)
        for _ in range(200):
            executor.tick()

        assert (
            motion_statechart.observation_state[stalled] != ObservationStateValues.TRUE
        )


# %% measuring the convergence rate


class TestConvergenceRate:

    def test_rate_is_zero_when_the_robot_stands_still(self, cylinder_bot_world: World):
        """
        The convergence rate is the error's derivative times the joint velocities, so it
        vanishes when nothing moves.
        """
        goal = CartesianPosition(
            root_link=cylinder_bot_world.root,
            tip_link=cylinder_bot_world.get_kinematic_structure_entity_by_name("bot"),
            goal_point=Point3(1, 0, 0, reference_frame=cylinder_bot_world.root),
        )
        artifacts = goal.build(MotionStatechartContext(world=cylinder_bot_world))
        rate = artifacts.error.create_rate_expression()

        for degree_of_freedom in cylinder_bot_world.active_degrees_of_freedom:
            cylinder_bot_world.state[degree_of_freedom.id].velocity = 0.0
        cylinder_bot_world.notify_state_change()

        assert rate.evaluate()[0] == pytest.approx(0.0)

    def test_rate_is_negative_while_closing_on_the_goal(
        self, cylinder_bot_world: World
    ):
        """
        Moving towards the goal shrinks the error, so its rate of change is negative.
        """
        bot = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
        goal = CartesianPosition(
            root_link=cylinder_bot_world.root,
            tip_link=bot,
            goal_point=Point3(1, 0, 0, reference_frame=cylinder_bot_world.root),
        )
        artifacts = goal.build(MotionStatechartContext(world=cylinder_bot_world))
        rate = artifacts.error.create_rate_expression()

        # Drive every degree of freedom that shortens the distance to the goal.
        error_gradient = artifacts.error.expression.jacobian(
            [
                degree_of_freedom.variables.position
                for degree_of_freedom in cylinder_bot_world.active_degrees_of_freedom
            ]
        ).evaluate()[0]
        for index, degree_of_freedom in enumerate(
            cylinder_bot_world.active_degrees_of_freedom
        ):
            cylinder_bot_world.state[degree_of_freedom.id].velocity = -float(
                np.sign(error_gradient[index])
            )
        cylinder_bot_world.notify_state_change()

        assert rate.evaluate()[0] < 0

    def test_expression_without_joints_has_no_rate(self):
        """
        An error that does not depend on any joint cannot change through robot motion.
        """
        assert time_derivative_from_joint_motion(Scalar(3.0)).evaluate()[0] == 0.0

    def test_sampled_error_has_no_symbolic_rate(self):
        """
        A sampled error is differenced across control cycles instead of differentiated.
        """
        assert SampledErrorSignal(Scalar(1.0)).create_rate_expression() is None
        assert SymbolicErrorSignal(Scalar(1.0)).create_rate_expression() is not None


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

        artifacts = goal.build(MotionStatechartContext(world=cylinder_bot_world))

        tip_position = cylinder_bot_world.compute_forward_kinematics_np(
            cylinder_bot_world.root, bot
        )[:3, 3]
        expected = np.linalg.norm(goal_point.to_np()[:3] - tip_position)
        assert artifacts.error.expression.evaluate()[0] == pytest.approx(expected)

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

        artifacts = goal.build(MotionStatechartContext(world=cylinder_bot_world))

        expected = (artifacts.error.expression <= goal.threshold).evaluate()[0]
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
        motion_statechart = MotionStatechart()
        motion_statechart.add_node(goal)
        motion_statechart.add_node(EndMotion.when_true(goal))
        executor = Executor(MotionStatechartContext(world=cylinder_bot_world))
        executor.compile(motion_statechart=motion_statechart)

        raw_error = goal.error_signal.expression.evaluate()[0]
        assert goal.normalized_error.evaluate()[0] == pytest.approx(
            raw_error / goal.threshold
        )


# %% misuse is reported


class TestMisuse:

    def test_watching_a_task_that_never_converges_is_rejected(
        self, cylinder_bot_world: World
    ):
        """
        A velocity limit enforces an invariant rather than closing on a goal, so it has
        no progress to watch.
        """
        bot = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
        limit = CartesianPositionVelocityLimit(
            root_link=cylinder_bot_world.root, tip_link=bot
        )
        motion_statechart = MotionStatechart()
        motion_statechart.add_node(limit)
        motion_statechart.add_node(ProgressStalled(monitored_node=limit))
        motion_statechart.add_node(EndMotion())

        executor = Executor(MotionStatechartContext(world=cylinder_bot_world))
        with pytest.raises(NoConvergingTaskError):
            executor.compile(motion_statechart=motion_statechart)


# %% dependency ordering


class TestNodeDependencies:

    def test_a_dependency_is_built_first_even_when_added_later(
        self, cylinder_bot_world: World
    ):
        """
        Build order follows declared dependencies, not the order nodes were added.
        """
        built_after: List[str] = []
        dependency = NodeWithDeclaredDependencies(
            name="dependency", built_after=built_after
        )
        dependent = NodeWithDeclaredDependencies(
            name="dependent", dependencies=[dependency], built_after=built_after
        )
        motion_statechart = MotionStatechart()
        motion_statechart.add_node(dependent)
        motion_statechart.add_node(dependency)
        motion_statechart.add_node(EndMotion())

        executor = Executor(MotionStatechartContext(world=cylinder_bot_world))
        executor.compile(motion_statechart=motion_statechart)

        assert built_after == ["dependency", "dependent"]

    def test_watching_a_goal_added_before_it_works(self, cylinder_bot_world: World):
        """
        The monitor is expanded after the goal it watches, so it can find that goal's
        tasks however the nodes were ordered.
        """
        bot = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
        goal = CartesianPosition(
            root_link=cylinder_bot_world.root,
            tip_link=bot,
            goal_point=Point3(1, 0, 0, reference_frame=cylinder_bot_world.root),
        )
        sequence = Sequence(nodes=[goal])
        stalled = ProgressStalled(monitored_node=sequence, timeout=1.0)

        motion_statechart = MotionStatechart()
        motion_statechart.add_node(stalled)
        motion_statechart.add_node(sequence)
        motion_statechart.add_node(EndMotion.when_true(sequence))

        executor = Executor(MotionStatechartContext(world=cylinder_bot_world))
        executor.compile(motion_statechart=motion_statechart)

        assert stalled.monitored_tasks == [goal]

    def test_a_dependency_cycle_is_reported(self, cylinder_bot_world: World):
        """
        Nodes that depend on each other cannot be ordered, which must be said plainly
        instead of recursing forever.
        """
        first = NodeWithDeclaredDependencies(name="first")
        second = NodeWithDeclaredDependencies(name="second", dependencies=[first])
        first.dependencies.append(second)
        motion_statechart = MotionStatechart()
        motion_statechart.add_nodes([first, second])
        motion_statechart.add_node(EndMotion())

        executor = Executor(MotionStatechartContext(world=cylinder_bot_world))
        with pytest.raises(CyclicNodeDependencyError):
            executor.compile(motion_statechart=motion_statechart)


# %% errors that cannot be differentiated


class TestSampledError:

    def test_trajectory_progress_is_measured_by_sampling(
        self, cylinder_bot_world: World
    ):
        """
        A trajectory task knows how far it has come only from its own bookkeeping, so
        its progress is differenced across control cycles rather than differentiated.
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
        motion_statechart = MotionStatechart()
        motion_statechart.add_node(trajectory)
        motion_statechart.add_node(EndMotion.when_true(trajectory))
        stalled = ProgressStalled(monitored_node=trajectory, timeout=1.0)
        motion_statechart.add_node(stalled)

        executor = Executor(MotionStatechartContext(world=cylinder_bot_world))
        executor.compile(motion_statechart=motion_statechart)

        assert isinstance(trajectory.error_signal, SampledErrorSignal)
        recorded = tick_until_end_recording(executor, motion_statechart, [stalled])
        assert ObservationStateValues.TRUE not in recorded[stalled]
