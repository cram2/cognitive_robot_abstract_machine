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
