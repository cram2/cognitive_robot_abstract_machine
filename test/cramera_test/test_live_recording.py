"""
Tests of the live recording buffer: the lifecycle and per-tick capture of one live run,
independent of the bridge/world it is fed from.
"""

from __future__ import annotations

import pytest

from cramera.live.chart_structure import (
    ChartEdgeEntry,
    ChartNodeEntry,
    ChartSnapshot,
    ObservationName,
)
from cramera.live.bridge import (
    WorldStateSnapshot,
)
from cramera.live.recording import (
    FrameRange,
    InvalidFrameRange,
    NoActiveRecording,
    Recording,
    RecordingState,
)


def snapshot(frames=None, base=None, objects=None) -> WorldStateSnapshot:
    return WorldStateSnapshot(frames=frames or {}, base=base, objects=objects or {})


def statechart(life_cycle: str = "RUNNING") -> ChartSnapshot:
    """
    A one-node motion statechart, as the bridge publishes it.
    """
    return ChartSnapshot(
        signature="c1",
        title="Reach",
        nodes=[
            ChartNodeEntry(
                id="goal",
                name="ReachGoal",
                class_name="CartesianPose",
                parent=None,
                life_cycle=life_cycle,
                observation=ObservationName.UNKNOWN,
            )
        ],
        edges=[ChartEdgeEntry(source="goal", target="goal", kind="END")],
    )


class TestLifecycle:
    def test_a_fresh_recording_is_idle(self):
        assert Recording().state is RecordingState.IDLE

    def test_start_moves_to_recording(self):
        recording = Recording()

        recording.start()

        assert recording.state is RecordingState.RECORDING

    def test_stop_moves_to_finalized(self):
        recording = Recording()
        recording.start()

        recording.stop()

        assert recording.state is RecordingState.FINALIZED

    def test_stop_without_a_start_raises(self):
        with pytest.raises(NoActiveRecording):
            Recording().stop()

    def test_stop_is_idempotent(self):
        recording = Recording()
        recording.start()
        recording.append(snapshot(frames={"joint": 1.0}))

        first = recording.stop()
        second = recording.stop()

        assert first == second
        assert recording.state is RecordingState.FINALIZED

    def test_discard_returns_to_idle_and_clears_frames(self):
        recording = Recording()
        recording.start()
        recording.append(snapshot(frames={"joint": 1.0}))
        recording.stop()

        recording.discard()

        assert recording.state is RecordingState.IDLE
        assert recording.frame_count() == 0

    def test_starting_again_clears_a_previous_recordings_frames(self):
        recording = Recording()
        recording.start()
        recording.append(snapshot(frames={"joint": 1.0}))
        recording.stop()

        recording.start()

        assert recording.frame_count() == 0


class TestAppend:
    def test_a_tick_while_recording_is_buffered(self):
        recording = Recording()
        recording.start()

        recording.append(snapshot(frames={"joint": 1.0}, base=[0, 0, 0, 0, 0, 0, 1]))

        assert recording.frame_count() == 1

    def test_the_action_being_performed_is_buffered_with_the_tick(self):
        """
        What each tick was doing is what names its stretch of the replay timeline (see
        cramera.live.recording_segments).
        """
        recording = Recording()
        recording.start()

        recording.append(snapshot(frames={"joint": 1.0}), "TransportAction")

        assert recording.stop()[0].step == "TransportAction"

    def test_a_tick_with_no_action_running_is_buffered_without_one(self):
        recording = Recording()
        recording.start()

        recording.append(snapshot(frames={"joint": 1.0}))

        assert recording.stop()[0].step is None

    def test_a_tick_while_idle_is_dropped(self):
        recording = Recording()

        recording.append(snapshot(frames={"joint": 1.0}))

        assert recording.frame_count() == 0

    def test_a_tick_after_stop_is_dropped(self):
        recording = Recording()
        recording.start()
        recording.stop()

        recording.append(snapshot(frames={"joint": 1.0}))

        assert recording.frame_count() == 0

    def test_frames_preserve_recorded_order(self):
        recording = Recording()
        recording.start()
        recording.append(snapshot(frames={"joint": 1.0}))
        recording.append(snapshot(frames={"joint": 2.0}))

        frames = recording.stop()

        assert [frame.frames["joint"] for frame in frames] == [1.0, 2.0]

    def test_a_later_mutation_of_the_source_snapshot_does_not_affect_the_recording(
        self,
    ):
        """
        Bridge.snapshot() reuses one WorldStateSnapshot's dicts are not mutated in
        place, but a recording must not assume that: it defensively copies each tick.
        """
        recording = Recording()
        recording.start()
        frames = {"joint": 1.0}
        objects = {"milk": [0, 0, 0, 0, 0, 0, 1]}
        recording.append(snapshot(frames=frames, objects=objects))

        frames["joint"] = 99.0
        objects["milk"][0] = 99.0

        [recorded] = recording.stop()
        assert recorded.frames["joint"] == 1.0
        assert recorded.objects["milk"][0] == 0


class TestFramesPerSecond:
    def test_falls_back_with_fewer_than_two_ticks(self):
        recording = Recording()
        recording.start()

        assert recording.frames_per_second(fallback=42.0) == 42.0

    def test_computed_from_the_span_between_the_first_and_last_tick(self, monkeypatch):
        import cramera.live.recording as recording_module

        ticks = iter([100.0, 100.5, 101.0])
        monkeypatch.setattr(recording_module.time, "time", lambda: next(ticks))
        recording = Recording()
        recording.start()
        recording.append(snapshot())
        recording.append(snapshot())
        recording.append(snapshot())

        # 3 ticks over 1.0s of wall time -> 3 fps
        assert recording.frames_per_second() == 3.0


class TestStatusPayload:
    def test_idle(self):
        assert Recording().status_payload() == {
            "state": "idle",
            "frameCount": 0,
            "durationSeconds": 0.0,
            "framesPerSecond": 30.0,
            "sceneName": None,
        }

    def test_the_frame_rate_is_reported_for_the_trim_to_measure_with(self, monkeypatch):
        import cramera.live.recording as recording_module

        ticks = iter([100.0, 100.5, 101.0])
        monkeypatch.setattr(recording_module.time, "time", lambda: next(ticks))
        recording = Recording()
        recording.start()
        recording.append(snapshot())
        recording.append(snapshot())
        recording.append(snapshot())

        payload = recording.status_payload()

        assert payload["framesPerSecond"] == recording.frames_per_second()

    def test_recording_reports_frame_count(self):
        recording = Recording()
        recording.start()
        recording.append(snapshot())
        recording.append(snapshot())

        payload = recording.status_payload()

        assert payload["state"] == "recording"
        assert payload["frameCount"] == 2

    def test_finalized_reports_the_scene_name_once_set(self):
        recording = Recording()
        recording.start()
        recording.stop()
        recording.scene_name = "__recording__"

        assert recording.status_payload()["sceneName"] == "__recording__"


# %% choosing a stretch of a recording to keep


class TestFrameRange:
    def recording_of(self, length: int) -> Recording:
        recording = Recording()
        recording.start()
        for index in range(length):
            recording.append(snapshot(frames={"joint": float(index)}))
        return recording

    def test_the_whole_recording_covers_every_frame(self):
        assert FrameRange.whole(3) == FrameRange(first=0, last=2)

    def test_a_range_selects_its_frames_inclusively(self):
        recording = self.recording_of(4)

        selected = recording.frames_in(FrameRange(first=1, last=2))

        assert [frame.frames["joint"] for frame in selected] == [1.0, 2.0]

    def test_a_single_frame_range_selects_that_frame(self):
        recording = self.recording_of(4)

        selected = recording.frames_in(FrameRange(first=3, last=3))

        assert [frame.frames["joint"] for frame in selected] == [3.0]

    def test_a_reversed_range_is_rejected(self):
        with pytest.raises(InvalidFrameRange):
            FrameRange(first=2, last=1)

    def test_a_negative_first_frame_is_rejected(self):
        with pytest.raises(InvalidFrameRange):
            FrameRange(first=-1, last=2)

    def test_a_range_reaching_past_the_last_frame_is_rejected(self):
        recording = self.recording_of(2)

        with pytest.raises(InvalidFrameRange):
            recording.frames_in(FrameRange(first=0, last=5))

    def test_the_whole_range_of_an_empty_recording_is_rejected(self):
        with pytest.raises(InvalidFrameRange):
            FrameRange.whole(0)


# %% motion statecharts
class TestStatechartCapture:
    def test_a_tick_keeps_the_statechart_that_was_executing(self):
        recording = Recording()
        recording.start()
        chart = statechart()

        recording.append(snapshot(), statechart=chart)

        assert recording.stop()[0].statechart == chart

    def test_a_tick_without_a_statechart_keeps_none(self):
        recording = Recording()
        recording.start()

        recording.append(snapshot())

        assert recording.stop()[0].statechart is None

    def test_consecutive_unchanged_ticks_share_one_snapshot(self):
        """
        A statechart ticks far more often than it changes; holding the published
        snapshot rather than a copy of it keeps a long run's buffer small.
        """
        recording = Recording()
        recording.start()

        recording.append(snapshot(), statechart=statechart())
        recording.append(snapshot(), statechart=statechart())

        frames = recording.stop()
        assert frames[0].statechart is frames[1].statechart

    def test_a_trimmed_range_keeps_the_statechart_of_its_frames(self):
        recording = Recording()
        recording.start()
        recording.append(snapshot())
        recording.append(snapshot(), statechart=statechart())

        kept = recording.frames_in(FrameRange(first=1, last=1))

        assert [frame.statechart for frame in kept] == [statechart()]
