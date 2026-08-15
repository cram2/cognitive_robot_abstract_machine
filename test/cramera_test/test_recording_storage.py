"""
Tests of managing an already-finalized live-recording bundle on disk: saving and
discarding, purely as filesystem operations independent of any live bridge.
"""

from __future__ import annotations

import json

import pytest

from cramera import paths
from cramera.live.recording import Recording
from cramera.live.recording_bundle import finalize_recording
from cramera.live.recording_storage import (
    NoSavedRecording,
    SceneNameTaken,
    discard_recording_bundle,
    has_saveable_recording,
    save_recording_bundle,
)
from cramera.onboard.scene_index import InvalidSceneName

from .test_live_bundle import attached_bridge


def finalized_on_disk(tmp_path, monkeypatch) -> None:
    """
    Write a finalized ``__recording__`` bundle under a scratch ``CRAMERA_DATA``, as if a
    demo process had already produced and finalized one — with no live bridge involved
    afterward, matching the pure-filesystem contract of save/discard.
    """
    monkeypatch.setenv("CRAMERA_DATA", str(tmp_path))
    bridge = attached_bridge()
    recording = Recording()
    recording.start()
    recording.append(bridge.state)
    finalize_recording(bridge, recording)


class TestHasSaveableRecording:
    def test_false_before_anything_is_finalized(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path))
        assert has_saveable_recording() is False

    def test_true_once_a_recording_is_finalized(self, tmp_path, monkeypatch):
        finalized_on_disk(tmp_path, monkeypatch)
        assert has_saveable_recording() is True

    def test_false_again_after_it_is_saved_or_discarded(self, tmp_path, monkeypatch):
        finalized_on_disk(tmp_path, monkeypatch)
        discard_recording_bundle()
        assert has_saveable_recording() is False


class TestDiscardRecordingBundle:
    def test_removes_the_bundle(self, tmp_path, monkeypatch):
        finalized_on_disk(tmp_path, monkeypatch)

        discard_recording_bundle()

        assert not (tmp_path / "scenes" / paths.RECORDING_SCENE_NAME).exists()

    def test_is_harmless_with_nothing_to_discard(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path))
        discard_recording_bundle()  # must not raise


class TestSaveRecordingBundle:
    def test_moves_the_bundle_and_renames_it(self, tmp_path, monkeypatch):
        finalized_on_disk(tmp_path, monkeypatch)

        name = save_recording_bundle("my_run")

        assert name == "my_run"
        saved = tmp_path / "scenes" / "my_run"
        assert json.loads((saved / "scene.json").read_text())["name"] == "my_run"
        assert not (tmp_path / "scenes" / paths.RECORDING_SCENE_NAME).exists()

    def test_registers_the_saved_scene_in_the_local_index(self, tmp_path, monkeypatch):
        finalized_on_disk(tmp_path, monkeypatch)

        save_recording_bundle("my_run")

        index = json.loads((tmp_path / "scenes" / "index.json").read_text())
        assert any(entry["name"] == "my_run" for entry in index["scenes"])

    def test_works_without_a_live_bridge_at_all(self, tmp_path, monkeypatch):
        """
        The whole point: saving must succeed purely from what is on disk, exactly the
        scenario a demo process that already exited leaves behind.
        """
        finalized_on_disk(tmp_path, monkeypatch)

        name = save_recording_bundle("after_process_exit")

        assert name == "after_process_exit"

    def test_rejects_an_unsafe_name(self, tmp_path, monkeypatch):
        finalized_on_disk(tmp_path, monkeypatch)

        with pytest.raises(InvalidSceneName):
            save_recording_bundle("../escape")

    def test_rejects_a_name_collision(self, tmp_path, monkeypatch):
        finalized_on_disk(tmp_path, monkeypatch)
        (tmp_path / "scenes" / "kitchen").mkdir()

        with pytest.raises(SceneNameTaken):
            save_recording_bundle("kitchen")

    def test_nothing_to_save_is_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path))

        with pytest.raises(NoSavedRecording):
            save_recording_bundle("my_run")
