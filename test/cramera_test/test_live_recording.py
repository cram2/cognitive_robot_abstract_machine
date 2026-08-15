"""
Tests of the live recording buffer: the lifecycle and per-tick capture of one live run,
independent of the bridge/world it is fed from.
"""

from __future__ import annotations

import pytest

from cramera.live.bridge import WorldStateSnapshot
from cramera.live.recording import NoActiveRecording, Recording, RecordingState


def snapshot(frames=None, base=None, objects=None) -> WorldStateSnapshot:
    return WorldStateSnapshot(frames=frames or {}, base=base, objects=objects or {})


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
            "sceneName": None,
        }

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
