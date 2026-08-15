"""
Tests for the scene-bundle search order (env → submodule → data dir).
"""

import json

from cramera import paths


class TestScenesDir:
    def test_env_override_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CRAMERA_SCENES", str(tmp_path / "mine"))
        assert paths.scenes_directory() == tmp_path / "mine"

    def test_initialized_submodule_is_used(self, monkeypatch, tmp_path):
        monkeypatch.delenv("CRAMERA_SCENES", raising=False)
        submodule = tmp_path / "scenes"
        submodule.mkdir()
        (submodule / "index.json").write_text(
            json.dumps({"default": None, "scenes": []})
        )
        monkeypatch.setattr(paths, "SCENES_SUBMODULE", submodule)
        assert paths.scenes_directory() == submodule

    def test_empty_submodule_falls_back_to_data_dir(self, monkeypatch, tmp_path):
        # an un-initialized submodule is an empty directory — must be skipped
        monkeypatch.delenv("CRAMERA_SCENES", raising=False)
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path / "data"))
        empty = tmp_path / "scenes"
        empty.mkdir()
        monkeypatch.setattr(paths, "SCENES_SUBMODULE", empty)
        assert paths.scenes_directory() == tmp_path / "data" / "scenes"

    def test_data_dir_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path / "d"))
        assert paths.data_directory() == tmp_path / "d"


class TestLocalScenesDirectory:
    def test_ignores_the_shared_scenes_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CRAMERA_SCENES", str(tmp_path / "shared"))
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path / "data"))
        assert paths.local_scenes_directory() == tmp_path / "data" / "scenes"

    def test_ignores_an_initialized_submodule(self, monkeypatch, tmp_path):
        monkeypatch.delenv("CRAMERA_SCENES", raising=False)
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path / "data"))
        submodule = tmp_path / "submodule"
        submodule.mkdir()
        (submodule / "index.json").write_text(json.dumps({"default": None, "scenes": []}))
        monkeypatch.setattr(paths, "SCENES_SUBMODULE", submodule)
        assert paths.local_scenes_directory() == tmp_path / "data" / "scenes"


class TestSceneRoots:
    def test_one_root_when_shared_and_local_coincide(self, monkeypatch, tmp_path):
        monkeypatch.delenv("CRAMERA_SCENES", raising=False)
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path))
        assert paths.scene_roots() == [tmp_path / "scenes"]

    def test_local_first_when_a_shared_submodule_is_checked_out(self, monkeypatch, tmp_path):
        monkeypatch.delenv("CRAMERA_SCENES", raising=False)
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path / "data"))
        submodule = tmp_path / "submodule"
        submodule.mkdir()
        (submodule / "index.json").write_text(json.dumps({"default": None, "scenes": []}))
        monkeypatch.setattr(paths, "SCENES_SUBMODULE", submodule)
        assert paths.scene_roots() == [tmp_path / "data" / "scenes", submodule]


class TestResolveSceneDirectory:
    def test_finds_a_scene_in_the_only_root(self, monkeypatch, tmp_path):
        monkeypatch.delenv("CRAMERA_SCENES", raising=False)
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path))
        bundle = tmp_path / "scenes" / "lab"
        bundle.mkdir(parents=True)
        (bundle / "scene.json").write_text("{}")
        assert paths.resolve_scene_directory("lab") == bundle

    def test_local_wins_on_a_name_collision(self, monkeypatch, tmp_path):
        monkeypatch.delenv("CRAMERA_SCENES", raising=False)
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path / "data"))
        submodule = tmp_path / "submodule"
        (submodule / "lab").mkdir(parents=True)
        (submodule / "lab" / "scene.json").write_text("{}")
        (submodule / "index.json").write_text(json.dumps({"default": None, "scenes": []}))
        monkeypatch.setattr(paths, "SCENES_SUBMODULE", submodule)
        local_bundle = tmp_path / "data" / "scenes" / "lab"
        local_bundle.mkdir(parents=True)
        (local_bundle / "scene.json").write_text("{}")

        assert paths.resolve_scene_directory("lab") == local_bundle

    def test_none_when_no_root_has_it(self, monkeypatch, tmp_path):
        monkeypatch.delenv("CRAMERA_SCENES", raising=False)
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path))
        assert paths.resolve_scene_directory("missing") is None
