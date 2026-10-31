"""
Native plan status publication and plan inspection after recording.
"""

from __future__ import annotations

from typing_extensions import TYPE_CHECKING

from coraplex.language import SequentialNode
from coraplex.plans.plan import Plan
from coraplex.plans.plan_node import ActionNode, MotionNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.motions.base import BaseMotion

from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)

from cramera.live.bridge import Bridge
from cramera.live.chart_structure import ObservationName
from cramera.live.recording_bundle import write_recording_bundle
from cramera.live.recording_storage import trim_recording_bundle
from cramera.live.frame_range import FrameRange
from cramera.live.recording import RecordedFrame, Recording
from cramera.knowledge.recorded_statecharts import RecordedStatecharts, STATECHART_FILE
from cramera.generated_json import GeneratedJson
from cramera import paths

from .dataset.motion_execution import motion_execution
from .test_live_bridge import nodes_by_kind, plan_bridge
from .test_live_bundle import attached_bridge
from .test_recording_bundle import frame_with_milk
from .test_live_recording import statechart, snapshot

if TYPE_CHECKING:
    from .dataset.motion_execution import MotionExecution


# %% native status publication


class TestNativePlanStatus:
    """
    Native lifecycle values retain their meaning in the viewer.
    """

    def test_an_unstarted_parent_keeps_its_state_with_a_running_child(self) -> None:
        """
        A child's execution does not replace its parent's current lifecycle.
        """
        bridge = Bridge()
        child = MotionNode(designator=BaseMotion(), status=LifeCycleValues.RUNNING)
        root = SequentialNode()
        plan = Plan()
        plan.add_edge(root, child)
        bridge.begin_plan(plan)
        assert nodes_by_kind(bridge)["SequentialNode"]["status"] == root.status.name

    def test_a_paused_motion_publishes_a_paused_status(self) -> None:
        """
        A paused native motion is serialized with its native lifecycle name.
        """
        bridge = Bridge()
        motion = MotionNode(designator=BaseMotion(), status=LifeCycleValues.PAUSED)
        plan = Plan()
        plan.add_node(motion)
        bridge.begin_plan(plan)
        assert nodes_by_kind(bridge)["MotionNode"]["status"] == motion.status.name

    def test_unexecuted_conditions_do_not_keep_a_completed_action_running(
        self, plan_bridge
    ) -> None:
        """
        An action retains completion even when its condition was not executed.

        :param plan_bridge: The bridge and its native plan nodes.
        """
        bridge, _, action, condition, motion = plan_bridge
        action.status = motion.status = LifeCycleValues.SUCCEEDED
        bridge.snapshot_plan()
        assert nodes_by_kind(bridge)["ActionNode"]["status"] == action.status.name
        assert nodes_by_kind(bridge)["ConditionNode"]["status"] == condition.status.name


# %% plan persistence


class TestMotionHistoryRecording:
    """
    Plan completion retains a final chart observation on the last recorded pose.
    """

    def test_last_world_frame_keeps_the_final_observation(
        self, motion_execution: MotionExecution
    ) -> None:
        """
        The final chart-only observation preserves the last world pose.
        """
        bridge = motion_execution.bridge
        bridge.recording = Recording()
        bridge.recording.start()
        chart = motion_execution.chart
        chart.observation_state.data[-1] = ObservationStateValues.FALSE
        motion_execution.callback.on_start(motion_execution.motion)
        motion_execution.record(LifeCycleValues.RUNNING)
        bridge.recording.append(
            snapshot(frames={"joint": 0.5}), statechart=bridge.executing_statechart()
        )
        first_world_frame = bridge.recording.frames_in(FrameRange(0, 0))[0]
        chart.observation_state.data[-1] = ObservationStateValues.TRUE

        motion_execution.record(LifeCycleValues.SUCCEEDED)
        motion_execution.callback.on_end(motion_execution.plan.root)

        [recorded] = bridge.recording.stop()
        assert recorded.statechart == bridge.executing_statechart()
        assert recorded.frames == first_world_frame.frames
        assert recorded.statechart.nodes[-1].observation == ObservationName.TRUE
        assert (
            first_world_frame.statechart.nodes[-1].observation == ObservationName.FALSE
        )

    def test_a_finalized_recording_does_not_change_on_later_history_updates(
        self, motion_execution: MotionExecution
    ) -> None:
        """
        History changes and plan completion leave saved captures unchanged.
        """
        bridge = motion_execution.bridge
        bridge.recording = Recording()
        bridge.recording.start()
        motion_execution.callback.on_start(motion_execution.motion)
        motion_execution.record(LifeCycleValues.RUNNING)
        bridge.recording.append(snapshot(), statechart=bridge.executing_statechart())
        original = bridge.recording.stop()
        motion_execution.chart.observation_state.data[-1] = ObservationStateValues.TRUE

        motion_execution.record(LifeCycleValues.SUCCEEDED)
        motion_execution.callback.on_end(motion_execution.plan.root)

        assert bridge.recording.stop() == original


class TestRecordedPlan:
    """
    A replay keeps the completed run's plan available for inspection.
    """

    def test_recording_contains_the_published_plan(self, tmp_path) -> None:
        """
        The saved hierarchy retains native plan labels and completion states.

        :param tmp_path: Temporary directory for the exported recording.
        """
        bridge = attached_bridge()
        child = ActionNode(
            designator=ActionDescription(), status=LifeCycleValues.SUCCEEDED
        )
        root = SequentialNode(status=LifeCycleValues.SUCCEEDED)
        plan = Plan()
        plan.add_edge(root, child)
        bridge.begin_plan(plan)

        scene = write_recording_bundle(
            bridge, [frame_with_milk()], 20.0, tmp_path / "recording", "finished_run"
        )

        [recorded_root] = scene["planTrees"]
        assert recorded_root["label"] == type(root).__name__
        assert recorded_root["status"] == LifeCycleValues.SUCCEEDED.name
        [recorded_child] = recorded_root["children"]
        assert recorded_child["label"] == type(child.designator).__name__
        assert recorded_child["children"] == []

    def test_trim_keeps_the_statecharts_of_the_selected_frames(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path))
        bridge = attached_bridge()
        frames = [
            RecordedFrame(
                frames={}, base=None, objects={}, statechart=statechart(status)
            )
            for status in (
                LifeCycleValues.NOT_STARTED.name,
                LifeCycleValues.RUNNING.name,
                LifeCycleValues.SUCCEEDED.name,
            )
        ]
        bundle = paths.local_scenes_directory() / paths.RECORDING_SCENE_NAME
        write_recording_bundle(bridge, frames, 20.0, bundle, paths.RECORDING_SCENE_NAME)
        original = RecordedStatecharts.of_payload(
            GeneratedJson(bundle / STATECHART_FILE).read()
        )

        trim_recording_bundle(FrameRange(first=1, last=2))

        trimmed = RecordedStatecharts.of_payload(
            GeneratedJson(bundle / STATECHART_FILE).read()
        )
        assert trimmed.moment_of_frame == original.moment_of_frame[1:3]
