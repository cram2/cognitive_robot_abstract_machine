"""
Visualization is opt-in and shows the world of the executor it is added to while that
executor runs a statechart.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field

import pytest
import rclpy

import coraplex.testing as testing_module
import cramph.visualization as visualization_module
from coraplex.datastructures.enums import ActionTrialVisualization
from coraplex.plans.executors import (
    RealRobotPlanExecutor,
    KinematicallySimulatedPlanExecutor,
)
from cramph.context import StatechartContext
from cramph.exceptions import RvizUnavailableError
from cramph.executor import StatechartExecutor
from cramph.node import EndStatechart
from cramph.nodes_for_testing import ConstTrueNode
from cramph.statechart import Statechart
from cramph.visualization import (
    RerunVisualization,
    RvizVisualization,
    WorldVisualization,
)
from semantic_digital_twin.adapters.rerun import RerunMode
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.world import World

from ..plan_running import robot_extensions
from .test_demonstrations import RecordingDemonstration


# %% a visualization recording its runs
@dataclass
class VisualizationRecordingItsRuns(WorldVisualization):
    """
    A visualization recording which worlds it was started on and how often it stopped.
    """

    started_worlds: list[World] = field(default_factory=list)
    """
    The world of every start, in order.
    """

    stops: int = 0
    """
    How often a started run was stopped.
    """

    running: bool = False
    """
    Whether a run is started and not yet stopped.
    """

    @property
    def is_rendering(self) -> bool:
        return self.running

    def _start(self, world: World, cleanup: ExitStack) -> None:
        self.started_worlds.append(world)
        self.running = True
        cleanup.callback(self._record_stop)

    def _record_stop(self) -> None:
        """
        Count a stop of the started run.
        """
        self.stops += 1
        self.running = False


def an_ending_statechart(context: StatechartContext) -> Statechart:
    """
    :param context: The context the statechart is built in.
    :return: A statechart that ends once its only step succeeded.
    """
    statechart = Statechart(context=context)
    step = ConstTrueNode(name="step")
    statechart.add_node(step)
    statechart.add_node(EndStatechart.when_true(step))
    return statechart


def an_executor(
    visualization: WorldVisualization, world: World | None = None
) -> StatechartExecutor:
    """
    :param visualization: The visualization added to the executor.
    :param world: The world the executor runs in, a new one unless given.
    :return: An executor showing its world through `visualization`.
    """
    return StatechartExecutor(
        context=StatechartContext(world=World() if world is None else world),
        extensions=[visualization],
    )


# %% one run per executor
def test_a_visualization_shows_the_world_of_its_executor_while_it_runs() -> None:
    visualization = VisualizationRecordingItsRuns()
    world = World()
    executor = an_executor(visualization, world)

    executor.compile(an_ending_statechart(executor.context))
    assert visualization.started_worlds == [world]
    assert visualization.is_rendering

    executor.execute()
    assert visualization.stops == 1
    assert not visualization.is_rendering


def test_a_visualization_is_not_started_before_its_executor_compiles() -> None:
    visualization = VisualizationRecordingItsRuns()

    an_executor(visualization)

    assert visualization.started_worlds == []


def test_a_recompile_does_not_restart_the_visualization() -> None:
    visualization = VisualizationRecordingItsRuns()
    executor = an_executor(visualization)
    statechart = an_ending_statechart(executor.context)
    executor.compile(statechart)

    executor.after_recompile()

    assert len(visualization.started_worlds) == 1
    executor.execute()


def test_one_visualization_shows_each_run_of_sequential_executors() -> None:
    visualization = VisualizationRecordingItsRuns()
    worlds = [World(), World()]

    for world in worlds:
        executor = an_executor(visualization, world)
        executor.compile(an_ending_statechart(executor.context))
        executor.execute()

    assert visualization.started_worlds == worlds
    assert visualization.stops == len(worlds)


def test_a_plan_run_on_the_real_robot_is_shown_once_compiled() -> None:
    visualization = VisualizationRecordingItsRuns()
    world = World()
    executor = RealRobotPlanExecutor(
        StatechartContext(world=world), extensions=[visualization]
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(ConstTrueNode(name="step"))

    executor.compile(statechart)

    assert visualization.started_worlds == [world]
    visualization.stop()


def test_starting_twice_keeps_the_running_visualization() -> None:
    visualization = VisualizationRecordingItsRuns()
    world = World()

    visualization.start(world)
    visualization.start(World())

    assert visualization.started_worlds == [world]
    visualization.stop()
    visualization.stop()
    assert visualization.stops == 1


# %% the visualizations of the testing helper
def test_testing_helper_stays_headless_without_ros(monkeypatch, rclpy_node) -> None:
    """
    Without ROS, tests run headless.
    """
    monkeypatch.setattr(testing_module, "VizMarkerPublisher", None)
    assert testing_module.selected_visualizations(rclpy_node) == []


def test_testing_helper_stays_headless_without_a_ros_node() -> None:
    assert testing_module.selected_visualizations(None) == []


def test_testing_helper_shows_rviz_with_ros(rclpy_node) -> None:
    [selected] = testing_module.selected_visualizations(rclpy_node)
    assert isinstance(selected, RvizVisualization)


# %% native renderers
def test_rviz_publishes_through_its_node_while_its_executor_runs(
    cylinder_bot_world, rclpy_node
) -> None:
    visualization = RvizVisualization(ros_node=rclpy_node, collision_visualization=True)
    executor = an_executor(visualization, cylinder_bot_world)

    executor.compile(an_ending_statechart(executor.context))
    publisher = visualization.publisher
    assert publisher.node is rclpy_node
    assert publisher._collision_publisher is not None

    executor.execute()
    assert not visualization.is_rendering
    assert rclpy_node.context.ok()
    assert (
        publisher
        not in cylinder_bot_world.get_world_model_manager().model_change_callbacks
    )


def test_rviz_shows_a_world_without_an_executor(cylinder_bot_world, rclpy_node) -> None:
    visualization = RvizVisualization(ros_node=rclpy_node)

    visualization.start(cylinder_bot_world)
    publisher = visualization.publisher
    assert publisher.node is rclpy_node

    visualization.stop()
    assert not visualization.is_rendering


def test_unavailable_ros_is_reported_without_starting(monkeypatch, rclpy_node) -> None:
    """
    RViz requires its optional ROS publisher.
    """
    monkeypatch.setattr(visualization_module, "VizMarkerPublisher", None)
    executor = an_executor(RvizVisualization(ros_node=rclpy_node))
    with pytest.raises(RvizUnavailableError):
        executor.compile(an_ending_statechart(executor.context))


def test_rerun_save_uses_existing_native_recording(
    cylinder_bot_world, tmp_path
) -> None:
    """
    Recording is delegated to the unchanged native Rerun adapter.
    """
    recording = tmp_path / "world.rrd"
    selected = RerunVisualization(mode=RerunMode.SAVE, target=str(recording))
    selected.start(cylinder_bot_world)
    selected.stop()
    assert recording.stat().st_size > 0
    assert not selected.is_rendering


# %% demonstrations and action trials
def test_one_visualization_shows_every_repetition_of_a_demonstration(
    monkeypatch, cylinder_bot_world
) -> None:
    shown: list[VisualizationRecordingItsRuns] = []
    create_executor = RecordingDemonstration.create_executor

    def recording_executor(demonstration, world):
        executor = create_executor(demonstration, world)
        shown.extend(
            extension
            for extension in executor.extensions
            if isinstance(extension, VisualizationRecordingItsRuns)
        )
        return executor

    monkeypatch.setattr(RecordingDemonstration, "create_executor", recording_executor)
    demonstration = RecordingDemonstration(
        world=cylinder_bot_world,
        used_robot=MinimalRobot,
        repetitions=2,
        visualization=VisualizationRecordingItsRuns(),
    )

    demonstration.run()

    assert [
        visualization is demonstration.visualization for visualization in shown
    ] == [True] * demonstration.repetitions
    assert (
        demonstration.visualization.started_worlds
        == [cylinder_bot_world] * demonstration.repetitions
    )
    assert demonstration.visualization.stops == demonstration.repetitions
    assert demonstration.ros_session is None
    assert not rclpy.ok()


def test_a_debugging_action_trial_is_shown_apart_from_the_world_it_copies(
    cylinder_bot_world, rclpy_node
) -> None:
    robot = cylinder_bot_world.get_semantic_annotations_by_type(MinimalRobot)[0]
    executor = KinematicallySimulatedPlanExecutor(
        StatechartContext.with_extensions(cylinder_bot_world, robot_extensions(robot)),
        ros_node=rclpy_node,
        debug=True,
    )
    trials = executor.child_chooser.trial

    trial_executor = trials._trial_executor(trials._copy())

    [shown] = [
        extension
        for extension in trial_executor.extensions
        if isinstance(extension, RvizVisualization)
    ]
    assert trial_executor.ros_node is None
    assert shown.ros_node is rclpy_node
    assert shown.frame_prefix == ActionTrialVisualization.FRAME_PREFIX
    assert shown.marker_topic == ActionTrialVisualization.MARKER_TOPIC
    assert shown.marker_alpha == trials.copy_marker_alpha
    trials.discard()
