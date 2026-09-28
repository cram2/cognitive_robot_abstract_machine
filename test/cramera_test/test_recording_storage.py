"""
Tests of managing an already-finalized live-recording bundle on disk: saving and
discarding, purely as filesystem operations independent of any live bridge.
"""

from __future__ import annotations

import json

import pytest

from cramera import paths
from cramera.live.bridge import WorldStateSnapshot
from cramera.live.recording import FrameRange, InvalidFrameRange, Recording
from cramera.live.recording_bundle import finalize_recording
from cramera.knowledge.detected_events import SceneField
from cramera.paths import RECORDING_SCENE_NAME
from typing_extensions import Any, Dict
from cramera.live.recording_storage import (
    NoSavedRecording,
    SceneDestination,
    SceneNameTaken,
    SharedScenesUnavailable,
    discard_recording_bundle,
    has_saveable_recording,
    save_recording_bundle,
    trim_recording_bundle,
)
from cramera.live.recording_segments import clip_segment_payloads
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


# %% cutting a finalized bundle down before it is saved


class TestTrimRecordingBundle:
    def finalized_run(self, tmp_path, monkeypatch, poses):
        """
        A finalized bundle on disk whose milk object visits each given pose in turn.
        """
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path))
        bridge = attached_bridge()
        recording = Recording()
        recording.start()
        for pose in poses:
            recording.append(
                WorldStateSnapshot(frames={}, base=None, objects={"milk.stl": pose})
            )
        finalize_recording(bridge, recording)

    def carried_poses(self, count):
        return [[0.1 * index, 0.2, 0.3, 0, 0, 0, 1] for index in range(count)]

    def bundle_file(self, tmp_path, name):
        return json.loads(
            (tmp_path / "scenes" / paths.RECORDING_SCENE_NAME / name).read_text()
        )

    def test_only_the_kept_frames_remain(self, tmp_path, monkeypatch):
        poses = self.carried_poses(6)
        self.finalized_run(tmp_path, monkeypatch, poses)

        trim_recording_bundle(FrameRange(first=1, last=3))

        trajectory = self.bundle_file(tmp_path, "trajectory.json")
        assert trajectory["objects"] == [{"milk.stl": pose} for pose in poses[1:4]]
        assert len(trajectory["frames"]) == 3
        assert len(trajectory["base"]) == 3

    def test_objects_spawn_where_the_kept_stretch_starts(self, tmp_path, monkeypatch):
        poses = self.carried_poses(6)
        self.finalized_run(tmp_path, monkeypatch, poses)

        trim_recording_bundle(FrameRange(first=4, last=5))

        [entry] = self.bundle_file(tmp_path, "scene.json")["objects"]
        assert entry["spawn"] == poses[4]

    def test_the_timeline_is_rebased_on_the_kept_stretch(self, tmp_path, monkeypatch):
        poses = self.carried_poses(6)
        self.finalized_run(tmp_path, monkeypatch, poses)
        before = self.bundle_file(tmp_path, "scene.json")["segments"]

        trim_recording_bundle(FrameRange(first=2, last=5))

        assert self.bundle_file(tmp_path, "scene.json")["segments"] == (
            clip_segment_payloads(before, FrameRange(first=2, last=5))
        )

    def test_trimming_twice_cuts_the_already_trimmed_run(self, tmp_path, monkeypatch):
        poses = self.carried_poses(6)
        self.finalized_run(tmp_path, monkeypatch, poses)

        trim_recording_bundle(FrameRange(first=1, last=4))
        trim_recording_bundle(FrameRange(first=1, last=2))

        trajectory = self.bundle_file(tmp_path, "trajectory.json")
        assert trajectory["objects"] == [{"milk.stl": pose} for pose in poses[2:4]]

    def test_a_range_past_the_recording_is_rejected(self, tmp_path, monkeypatch):
        self.finalized_run(tmp_path, monkeypatch, self.carried_poses(3))

        with pytest.raises(InvalidFrameRange):
            trim_recording_bundle(FrameRange(first=0, last=9))

    def test_trimming_without_a_bundle_is_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path))

        with pytest.raises(NoSavedRecording):
            trim_recording_bundle(FrameRange(first=0, last=1))


# %% promoting a saved episode into the shared scenes root


class TestSceneDestination:
    def separate_roots(self, tmp_path, monkeypatch):
        """
        A data directory and a distinct shared scenes root, as an initialized ``cram-
        scenes`` submodule alongside the local recordings makes them.
        """
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path / "data"))
        monkeypatch.setenv("CRAMERA_SCENES", str(tmp_path / "shared"))
        (tmp_path / "shared").mkdir(parents=True)

    def test_local_is_the_data_directorys_scenes(self, tmp_path, monkeypatch):
        self.separate_roots(tmp_path, monkeypatch)

        assert SceneDestination.LOCAL.directory() == paths.local_scenes_directory()

    def test_shared_is_the_configured_scenes_root(self, tmp_path, monkeypatch):
        self.separate_roots(tmp_path, monkeypatch)

        assert SceneDestination.SHARED.directory() == paths.scenes_directory()


class TestSharingARecording:
    def separate_roots(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path / "data"))
        monkeypatch.setenv("CRAMERA_SCENES", str(tmp_path / "shared"))
        (tmp_path / "shared").mkdir(parents=True)
        bridge = attached_bridge()
        recording = Recording()
        recording.start()
        recording.append(bridge.state)
        finalize_recording(bridge, recording)

    def test_sharing_writes_into_the_shared_root(self, tmp_path, monkeypatch):
        self.separate_roots(tmp_path, monkeypatch)

        save_recording_bundle("my_run", SceneDestination.SHARED)

        assert (tmp_path / "shared" / "my_run" / "scene.json").is_file()
        assert not (tmp_path / "data" / "scenes" / "my_run").exists()

    def test_sharing_indexes_it_in_the_shared_root(self, tmp_path, monkeypatch):
        self.separate_roots(tmp_path, monkeypatch)

        save_recording_bundle("my_run", SceneDestination.SHARED)

        index = json.loads((tmp_path / "shared" / "index.json").read_text())
        assert any(entry["name"] == "my_run" for entry in index["scenes"])

    def test_saving_locally_still_stays_out_of_the_shared_root(
        self, tmp_path, monkeypatch
    ):
        self.separate_roots(tmp_path, monkeypatch)

        save_recording_bundle("my_run", SceneDestination.LOCAL)

        assert (tmp_path / "data" / "scenes" / "my_run" / "scene.json").is_file()
        assert not (tmp_path / "shared" / "my_run").exists()

    def test_a_name_taken_in_either_root_is_refused(self, tmp_path, monkeypatch):
        self.separate_roots(tmp_path, monkeypatch)
        (tmp_path / "shared" / "my_run").mkdir()

        with pytest.raises(SceneNameTaken):
            save_recording_bundle("my_run", SceneDestination.SHARED)

    def test_sharing_without_a_shared_root_is_refused(self, tmp_path, monkeypatch):
        """
        Without an initialized submodule or CRAMERA_SCENES the shared root *is* the
        local one, so sharing would quietly be an ordinary save.
        """
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path / "data"))
        monkeypatch.delenv("CRAMERA_SCENES", raising=False)
        bridge = attached_bridge()
        recording = Recording()
        recording.start()
        recording.append(bridge.state)
        finalize_recording(bridge, recording)

        with pytest.raises(SharedScenesUnavailable):
            save_recording_bundle("my_run", SceneDestination.SHARED)


class TestNamingWhatWasRecorded:
    """
    A person saving a recording knows more about it than the run does: that the world
    built in code is a warehouse, and what the robot was doing in it.

    Whatever they say is written into the saved bundle, and what they leave out stays as
    derived.
    """

    def saved(self, tmp_path, monkeypatch, **given: str) -> Dict[str, Any]:
        """
        A finalized recording, saved under a name, read back.

        :param tmp_path: The scenes root to save into.
        :param monkeypatch: The active monkeypatch fixture.
        :param given: What the person saving it called it.
        """
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path))
        recording = tmp_path / "scenes" / RECORDING_SCENE_NAME
        recording.mkdir(parents=True)
        (recording / "scene.json").write_text(
            json.dumps({"robot": {"name": "unitreeg1"}, "models": []})
        )

        save_recording_bundle("run", **given)

        return json.loads(
            (tmp_path / "scenes" / "run" / "scene.json").read_text(encoding="utf-8")
        )

    def test_the_task_is_written_into_the_bundle(self, tmp_path, monkeypatch):
        scene = self.saved(tmp_path, monkeypatch, task="fetch a wrench")

        assert scene[SceneField.TASK] == "fetch a wrench"

    def test_the_names_are_written_into_the_bundle(self, tmp_path, monkeypatch):
        scene = self.saved(
            tmp_path, monkeypatch, robot="Unitree G1", environment="warehouse"
        )

        assert scene[SceneField.ROBOT_NAME] == "Unitree G1"
        assert scene[SceneField.ENVIRONMENT_NAME] == "warehouse"

    def test_what_nobody_named_is_left_to_the_recording(self, tmp_path, monkeypatch):
        scene = self.saved(tmp_path, monkeypatch)

        assert SceneField.TASK not in scene
        assert SceneField.ROBOT_NAME not in scene
        assert SceneField.ENVIRONMENT_NAME not in scene
