"""
Tests for the scenes index (``index.json``): what gets listed, and how the shared and
local-only recording roots are merged into one index the viewer reads.
"""

from __future__ import annotations

import json

import pytest

from cramera import paths
from cramera.onboard.scene_index import (
    InvalidSceneName,
    SceneIndexEntry,
    merged_scene_index,
    validate_scene_name,
    write_scene_index,
)


def _write_bundle(directory, name, robot="pr2", models=None):
    bundle = directory / name
    bundle.mkdir(parents=True)
    (bundle / "scene.json").write_text(
        json.dumps(
            {
                "robot": {"name": robot},
                "models": (
                    models if models is not None else [{"name": robot, "robot": True}]
                ),
            }
        )
    )
    return bundle


class TestSceneIndexEntry:
    def test_a_bundle_is_indexed_with_its_robot_and_environment(self, tmp_path):
        """
        The viewer's pickers resolve a (robot, environment) pair back to a bundle, so
        the index has to carry both per scene.
        """
        _write_bundle(
            tmp_path,
            "lab_scene",
            robot="pr2",
            models=[
                {"name": "pr2", "robot": True},
                {"name": "kitchen", "robot": False},
                {"name": "table", "robot": False},
            ],
        )

        [entry] = SceneIndexEntry.of_directory(tmp_path)

        assert entry.to_payload() == {
            "name": "lab_scene",
            "robot": "pr2",
            "environment": "kitchen+table",
            "task": None,
        }

    def test_a_bench_only_bundle_has_no_environment(self, tmp_path):
        _write_bundle(tmp_path, "bench", robot="tracy")

        [entry] = SceneIndexEntry.of_directory(tmp_path)

        assert entry.environment is None

    def test_a_directory_without_a_scene_file_is_skipped(self, tmp_path):
        (tmp_path / "not_a_bundle").mkdir()

        assert SceneIndexEntry.of_directory(tmp_path) == []

    def test_the_reserved_live_scene_name_is_skipped(self, tmp_path):
        """
        A live-attach snapshot (:mod:`cramera.live.live_bundle`) is a throwaway bundle
        rebuilt on every attach, never something a user onboarded — it must never show
        up as a robot/environment choice in the real picker.
        """
        _write_bundle(tmp_path, paths.LIVE_SCENE_NAME)

        assert SceneIndexEntry.of_directory(tmp_path) == []

    def test_the_reserved_recording_scene_name_is_skipped(self, tmp_path):
        """
        An unsaved live recording (:mod:`cramera.live.recording_bundle`) is likewise a
        throwaway bundle, never something a user onboarded.
        """
        _write_bundle(tmp_path, paths.RECORDING_SCENE_NAME)

        assert SceneIndexEntry.of_directory(tmp_path) == []

    def test_a_scene_without_a_bound_robot_is_indexed_with_an_empty_robot(
        self, tmp_path
    ):
        """
        A live recording captured without a robot annotation (a bench/environment-only
        run) has no robot to name — unlike an onboarded scene, which always has one.
        """
        bundle = tmp_path / "bench_only"
        bundle.mkdir()
        (bundle / "scene.json").write_text(
            json.dumps({"robot": None, "models": [{"name": "table", "robot": False}]})
        )

        [entry] = SceneIndexEntry.of_directory(tmp_path)

        assert entry.robot == ""


class TestWriteSceneIndex:
    def test_a_scene_is_registered_with_its_robot_and_environment(self, tmp_path):
        _write_bundle(tmp_path, "lab_scene")

        write_scene_index(tmp_path / "index.json", "lab_scene")

        index = json.loads((tmp_path / "index.json").read_text())
        assert index["scenes"] == [
            {"name": "lab_scene", "robot": "pr2", "environment": None, "task": None}
        ]

    def test_the_first_scene_becomes_the_default(self, tmp_path):
        _write_bundle(tmp_path, "lab_scene")

        write_scene_index(tmp_path / "index.json", "lab_scene")

        assert (
            json.loads((tmp_path / "index.json").read_text())["default"] == "lab_scene"
        )

    def test_a_later_default_is_left_alone(self, tmp_path):
        _write_bundle(tmp_path, "first")
        _write_bundle(tmp_path, "second")
        write_scene_index(tmp_path / "index.json", "first")

        write_scene_index(tmp_path / "index.json", "second")

        assert json.loads((tmp_path / "index.json").read_text())["default"] == "first"

    def test_a_removed_bundle_leaves_no_stale_entry(self, tmp_path):
        _write_bundle(tmp_path, "first")
        write_scene_index(tmp_path / "index.json", "first")
        _write_bundle(tmp_path, "second")
        write_scene_index(tmp_path / "index.json", "second")

        import shutil

        shutil.rmtree(tmp_path / "first")
        write_scene_index(tmp_path / "index.json", "second")

        names = [
            e["name"]
            for e in json.loads((tmp_path / "index.json").read_text())["scenes"]
        ]
        assert names == ["second"]


class TestMergedSceneIndex:
    def test_shared_only(self, monkeypatch, tmp_path):
        shared = tmp_path / "shared"
        _write_bundle(shared, "kitchen")
        (shared / "index.json").write_text(
            json.dumps({"default": "kitchen", "scenes": []})
        )
        monkeypatch.setenv("CRAMERA_SCENES", str(shared))
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path / "data"))

        index = merged_scene_index()

        assert index == {
            "default": "kitchen",
            "scenes": [
                {"name": "kitchen", "robot": "pr2", "environment": None, "task": None}
            ],
        }

    def test_local_only(self, monkeypatch, tmp_path):
        monkeypatch.delenv("CRAMERA_SCENES", raising=False)
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path))
        _write_bundle(tmp_path / "scenes", "my_run")

        index = merged_scene_index()

        assert index["scenes"] == [
            {"name": "my_run", "robot": "pr2", "environment": None, "task": None}
        ]
        assert index["default"] is None

    def test_a_local_recording_shadows_a_shared_scene_of_the_same_name(
        self, monkeypatch, tmp_path
    ):
        shared = tmp_path / "shared"
        _write_bundle(shared, "lab", robot="pr2")
        (shared / "index.json").write_text(json.dumps({"default": "lab", "scenes": []}))
        monkeypatch.setenv("CRAMERA_SCENES", str(shared))
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path / "data"))
        _write_bundle(tmp_path / "data" / "scenes", "lab", robot="tracy")

        index = merged_scene_index()

        [entry] = index["scenes"]
        assert entry["robot"] == "tracy"

    def test_both_roots_absent(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CRAMERA_SCENES", str(tmp_path / "nowhere"))
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path / "data"))

        assert merged_scene_index() == {"default": None, "scenes": []}


class TestValidateSceneName:
    def test_a_plain_name_is_accepted(self):
        assert validate_scene_name("my_run-2") == "my_run-2"

    @pytest.mark.parametrize(
        "name",
        ["", "a" * 65, "../escape", "has space", "slash/in/name", "dots.here"],
    )
    def test_an_unsafe_name_is_rejected(self, name):
        with pytest.raises(InvalidSceneName):
            validate_scene_name(name)

    @pytest.mark.parametrize(
        "name", [paths.LIVE_SCENE_NAME, paths.RECORDING_SCENE_NAME]
    )
    def test_a_reserved_name_is_rejected(self, name):
        with pytest.raises(InvalidSceneName):
            validate_scene_name(name)


# %% what a bundle says it is


class TestTheTaskARecordingShows:
    """
    A bundle is one robot, in one environment, doing one task.

    The first two fall out of what the recording loaded; the task is the one thing only
    the run itself knows, so a recording that names none takes what its plan did.
    """

    def bundle(self, directory: Path, name: str, task: object) -> Path:
        """
        A scene bundle on disk, with or without a task named.

        :param directory: The scenes directory to write into.
        :param name: Name of the bundle.
        :param task: The task to write, or None to leave it out.
        """
        bundle = directory / name
        bundle.mkdir(parents=True)
        scene = {
            "robot": {"name": "pr2"},
            "models": [
                {"name": "apartment", "robot": False},
                {"name": "pr2", "robot": True},
            ],
            "segments": [{"step": "transport_milk"}],
        }
        if task is not None:
            scene["task"] = task
        (bundle / "scene.json").write_text(json.dumps(scene))
        return bundle

    def test_the_entry_carries_the_task_the_bundle_names(self, tmp_path):
        self.bundle(tmp_path, "breakfast", "make breakfast")

        [entry] = SceneIndexEntry.of_directory(tmp_path)

        assert entry.task == "make breakfast"

    def test_a_bundle_naming_no_task_has_none(self, tmp_path):
        self.bundle(tmp_path, "breakfast", None)

        [entry] = SceneIndexEntry.of_directory(tmp_path)

        assert entry.task is None

    def test_the_task_travels_in_the_index(self, tmp_path):
        self.bundle(tmp_path, "breakfast", "make breakfast")

        [payload] = [
            entry.to_payload() for entry in SceneIndexEntry.of_directory(tmp_path)
        ]

        assert payload["task"] == "make breakfast"

    def test_an_entry_states_what_the_recording_is(self, tmp_path):
        self.bundle(tmp_path, "breakfast", "make breakfast")

        [entry] = SceneIndexEntry.of_directory(tmp_path)

        assert entry.describes() == "pr2 · apartment · make breakfast"

    def test_what_a_recording_without_a_task_states(self, tmp_path):
        self.bundle(tmp_path, "breakfast", None)

        [entry] = SceneIndexEntry.of_directory(tmp_path)

        assert entry.describes() == "pr2 · apartment"


class TestWhatAPersonCalledTheRecording:
    """
    What a bundle is gets derived from what the run loaded: the robot's own name, and
    the models that are not the robot.

    For a world built in code that derivation is
    thin -- every such world's environment is called ``environment`` -- so a person
    saving a recording can say what it really is, and what they say wins.
    """

    def bundle(self, directory: Path, **given: str) -> None:
        """
        A scene bundle on disk, with whatever a person named written into it.

        :param directory: The scenes directory to write into.
        :param given: The names a person gave, if any.
        """
        bundle = directory / "run"
        bundle.mkdir(parents=True)
        scene = {
            "robot": {"name": "unitreeg1"},
            "models": [
                {"name": "environment", "robot": False},
                {"name": "unitreeg1", "robot": True},
            ],
        }
        scene.update(given)
        (bundle / "scene.json").write_text(json.dumps(scene))

    def test_without_names_it_states_what_it_derived(self, tmp_path):
        self.bundle(tmp_path)

        [entry] = SceneIndexEntry.of_directory(tmp_path)

        assert entry.describes() == "unitreeg1 · environment"

    def test_a_named_environment_replaces_the_derived_one(self, tmp_path):
        self.bundle(tmp_path, environmentName="warehouse")

        [entry] = SceneIndexEntry.of_directory(tmp_path)

        assert entry.environment == "warehouse"

    def test_a_named_robot_replaces_the_derived_one(self, tmp_path):
        self.bundle(tmp_path, robotName="Unitree G1")

        [entry] = SceneIndexEntry.of_directory(tmp_path)

        assert entry.robot == "Unitree G1"

    def test_all_three_together_are_what_it_states(self, tmp_path):
        self.bundle(
            tmp_path,
            robotName="Unitree G1",
            environmentName="warehouse",
            task="fetch a wrench",
        )

        [entry] = SceneIndexEntry.of_directory(tmp_path)

        assert entry.describes() == "Unitree G1 · warehouse · fetch a wrench"
