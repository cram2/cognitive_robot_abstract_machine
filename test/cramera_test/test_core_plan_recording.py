"""
Native plan status publication and plan inspection after recording.
"""

from __future__ import annotations

from typing_extensions import TYPE_CHECKING

from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)

from cramera.live.bridge import Bridge, TaskStatusName
from cramera.live.chart_structure import ObservationName
from cramera.live.recording_bundle import write_recording_bundle
from cramera.live.recording_storage import trim_recording_bundle
from cramera.live.frame_range import FrameRange
from cramera.live.recording import RecordedFrame, Recording
from cramera.knowledge.recorded_statecharts import RecordedStatecharts, STATECHART_FILE
from cramera.generated_json import GeneratedJson
from cramera import paths

from .dataset.motion_execution import motion_execution
from .test_live_bridge import PlanWithRoot, make_plan_node, nodes_by_kind
from .test_live_bundle import attached_bridge
from .test_recording_bundle import frame_with_milk
from .test_live_recording import statechart, snapshot

if TYPE_CHECKING:
    from .dataset.motion_execution import MotionExecution


# %% native status translation


class TestNativePlanStatus:
    """
    Native lifecycle values retain their meaning in the viewer.
    """

    def test_an_unstarted_parent_inherits_its_running_child(self):
        bridge = Bridge()
        child = make_plan_node("MotionNode")
        child.status = LifeCycleValues.RUNNING
        root = make_plan_node("SequentialNode", children=[child])
        root.status = LifeCycleValues.NOT_STARTED

        bridge.begin_plan(PlanWithRoot(root=root))

        assert (
            nodes_by_kind(bridge)["SequentialNode"]["status"] == TaskStatusName.RUNNING
        )

    def test_a_paused_motion_publishes_a_paused_status(self):
        bridge = Bridge()
        motion = make_plan_node("MotionNode")
        motion.status = LifeCycleValues.PAUSED
        bridge.begin_plan(PlanWithRoot(root=motion))

        bridge.observe_motion_ended(motion)

        assert nodes_by_kind(bridge)["MotionNode"]["status"] == TaskStatusName.PAUSE

    def test_unexecuted_conditions_do_not_keep_a_completed_action_running(self):
        bridge = Bridge()
        condition = make_plan_node("ConditionNode")
        condition.status = LifeCycleValues.NOT_STARTED
        motion = make_plan_node("MotionNode")
        motion.status = LifeCycleValues.SUCCEEDED
        action = make_plan_node("ActionNode", children=[condition, motion])
        action.status = LifeCycleValues.NOT_STARTED

        bridge.begin_plan(PlanWithRoot(root=action))

        assert nodes_by_kind(bridge)["ActionNode"]["status"] == TaskStatusName.SUCCEEDED
        assert (
            nodes_by_kind(bridge)["ConditionNode"]["status"] == TaskStatusName.CREATED
        )


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

    def test_recording_contains_the_published_plan(self, tmp_path):
        bridge = attached_bridge()
        child = make_plan_node("ActionNode", status=TaskStatusName.SUCCEEDED)
        root = make_plan_node(
            "SequentialNode", status=TaskStatusName.SUCCEEDED, children=[child]
        )
        bridge.begin_plan(PlanWithRoot(root=root))

        scene = write_recording_bundle(
            bridge, [frame_with_milk()], 20.0, tmp_path / "recording", "finished_run"
        )

        [recorded_root] = scene["planTrees"]
        assert recorded_root["label"] == type(root).__name__
        assert recorded_root["status"] == TaskStatusName.SUCCEEDED
        [recorded_child] = recorded_root["children"]
        assert recorded_child["label"] == type(child).__name__
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
