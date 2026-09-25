"""
Native state history publication and recording boundaries.
"""

from __future__ import annotations

from typing_extensions import TYPE_CHECKING

from coraplex.plans.executables import MotionPlanHistory
from coraplex.plans.plan import Plan
from coraplex.plans.plan_node import MotionNode
from coraplex.robot_plans.motions.base import BaseMotion
from giskardpy.motion_statechart.data_types import LifeCycleValues

from cramera.live.bridge import TaskStatusName
from cramera.live.recording import Recording
from cramera.live.visualization import (
    LiveVisualization,
    WorldStateSync,
)

from .dataset.motion_execution import motion_execution
from .test_live_visualization import world

if TYPE_CHECKING:
    from semantic_digital_twin.world import World

    from .dataset.motion_execution import MotionExecution

# %% history publication


class TestMotionHistoryPublication:
    """
    History subscriptions publish motion changes for the plan's lifetime.
    """

    def test_motion_start_publishes_the_bound_chart(
        self, motion_execution: MotionExecution
    ) -> None:
        """
        The viewer sees the chart before the first controller update.
        """
        motion_execution.callback.on_start(motion_execution.motion)

        assert [node.name for node in motion_execution.bridge.chart_state.nodes] == [
            node.name for node in motion_execution.chart.nodes
        ]

    def test_native_history_changes_publish_the_chart_and_plan(
        self, motion_execution: MotionExecution
    ) -> None:
        """
        A recorded native state change refreshes both execution views.
        """
        motion_execution.callback.on_start(motion_execution.motion)
        motion_execution.plan.root.status = LifeCycleValues.RUNNING

        motion_execution.record(LifeCycleValues.RUNNING)

        assert motion_execution.bridge.chart_state.nodes[0].life_cycle == (
            LifeCycleValues.RUNNING.name
        )
        assert motion_execution.bridge.plan_state.nodes[0].status == (
            TaskStatusName.RUNNING
        )

    def test_merged_motions_subscribe_to_their_shared_history_once(
        self, motion_execution: MotionExecution
    ) -> None:
        """
        A shared chart does not accumulate one subscription per motion.
        """
        sibling = MotionNode(designator=BaseMotion())
        motion_execution.plan.add_edge(motion_execution.plan.root, sibling)
        sibling.motion_statechart = motion_execution.chart

        motion_execution.callback.on_start(motion_execution.motion)
        motion_execution.callback.on_start(sibling)

        assert motion_execution.chart.history.observers == [motion_execution.callback]

    def test_native_reset_clears_motion_and_parent_progress(
        self, motion_execution: MotionExecution
    ) -> None:
        """
        A reset chart restores both plan entries to their unstarted state.
        """
        motion_execution.plan.node_callbacks.append(motion_execution.callback)
        MotionPlanHistory(
            statechart=motion_execution.chart,
            motion_mappings={motion_execution.motion: motion_execution.chart.nodes[0]},
        )
        motion_execution.record(LifeCycleValues.RUNNING)
        motion_execution.record(LifeCycleValues.SUCCEEDED)

        motion_execution.record(LifeCycleValues.NOT_STARTED)

        assert [node.status for node in motion_execution.bridge.plan_state.nodes] == [
            TaskStatusName.CREATED,
            TaskStatusName.CREATED,
        ]

    def test_root_completion_removes_history_subscriptions(
        self, motion_execution: MotionExecution
    ) -> None:
        """
        Completed plans no longer alter the published state.
        """
        motion_execution.callback.on_start(motion_execution.motion)
        motion_execution.record(LifeCycleValues.RUNNING)
        published = motion_execution.bridge.chart_state
        assert motion_execution.chart.history.observers == [motion_execution.callback]

        motion_execution.callback.on_end(motion_execution.plan.root)
        motion_execution.callback.on_end(motion_execution.plan.root)
        motion_execution.record(LifeCycleValues.SUCCEEDED)

        assert motion_execution.chart.history.observers == []
        assert motion_execution.bridge.chart_state == published

    def test_visualization_stop_removes_history_subscriptions(
        self, world: World, motion_execution: MotionExecution
    ) -> None:
        """
        Stopping a viewer also detaches histories of unfinished plans.
        """
        visualization = LiveVisualization(world=world, bridge=motion_execution.bridge)
        callback = visualization.plan_callback(motion_execution.plan)
        callback.on_start(motion_execution.motion)
        assert motion_execution.chart.history.observers == [callback]

        visualization.stop()
        visualization.stop()

        assert motion_execution.chart.history.observers == []


# %% recording alignment


class TestHistoryRecordingAlignment:
    """
    Recorded poses retain the chart state of their own control cycle.
    """

    def test_next_history_change_does_not_overwrite_previous_world_frame(
        self, world: World, motion_execution: MotionExecution
    ) -> None:
        """
        History is published before the corresponding world frame is appended.
        """
        bridge = motion_execution.bridge
        bridge.attach(world)
        bridge.recording = Recording()
        bridge.recording.start()
        world_sync = WorldStateSync(_world=world, bridge=bridge)
        motion_execution.callback.on_start(motion_execution.motion)
        motion_execution.record(LifeCycleValues.RUNNING)
        world_sync.on_state_change()

        motion_execution.record(LifeCycleValues.SUCCEEDED)
        world_sync.on_state_change()

        frames = bridge.recording.stop()
        assert [frame.statechart.nodes[0].life_cycle for frame in frames] == [
            LifeCycleValues.RUNNING.name,
            LifeCycleValues.SUCCEEDED.name,
        ]

    def test_plan_end_flushes_a_chart_change_without_another_world_update(
        self, world: World, motion_execution: MotionExecution
    ) -> None:
        """
        A final chart-only controller update completes the last captured pose.
        """
        bridge = motion_execution.bridge
        bridge.attach(world)
        bridge.recording = Recording()
        bridge.recording.start()
        world_sync = WorldStateSync(_world=world, bridge=bridge)
        motion_execution.callback.on_start(motion_execution.motion)
        motion_execution.record(LifeCycleValues.RUNNING)
        world_sync.on_state_change()

        motion_execution.record(LifeCycleValues.SUCCEEDED)
        motion_execution.callback.on_end(motion_execution.motion)
        motion_execution.callback.on_end(motion_execution.plan.root)

        frames = bridge.recording.stop()
        assert len(frames) == 1
        assert (
            frames[0].statechart.nodes[0].life_cycle == LifeCycleValues.SUCCEEDED.name
        )

    def test_motion_root_flushes_the_native_final_chart_before_unsubscribing(
        self, world: World, motion_execution: MotionExecution
    ) -> None:
        """
        A motion that is the plan root retains its terminal chart-only update.
        """
        plan = Plan()
        plan.add_node(motion_execution.motion)
        motion_execution.callback.plan = plan
        plan.node_callbacks.append(motion_execution.callback)
        bridge = motion_execution.bridge
        bridge.begin_plan(plan)
        bridge.attach(world)
        bridge.recording = Recording()
        bridge.recording.start()
        world_sync = WorldStateSync(_world=world, bridge=bridge)
        MotionPlanHistory(
            statechart=motion_execution.chart,
            motion_mappings={motion_execution.motion: motion_execution.chart.nodes[0]},
        )
        motion_execution.record(LifeCycleValues.RUNNING)
        world_sync.on_state_change()

        motion_execution.record(LifeCycleValues.SUCCEEDED)

        frames = bridge.recording.stop()
        assert len(frames) == 1
        assert (
            frames[0].statechart.nodes[0].life_cycle == LifeCycleValues.SUCCEEDED.name
        )
        assert not any(
            observer is motion_execution.callback
            for observer in motion_execution.chart.history.observers
        )
