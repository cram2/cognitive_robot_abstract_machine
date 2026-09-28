"""
Tests for resolving which scene bundle the viewer and the knowledge base open on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cramera.knowledge.scene_bundle import SceneBundle

MONTESSORI_SCENE = {"name": "Franka_Montessori", "robot": "panda", "environment": None}
"""
The scene declared first by the index the tests write.
"""

WAREHOUSE_SCENE = {"name": "G1_warehouse", "robot": "unitreeg1", "environment": None}
"""
A second declared scene, so a fallback to the first one is a real choice.
"""


@pytest.fixture()
def scenes_directory(tmp_path, monkeypatch) -> Path:
    """
    An empty scenes directory cramera is pointed at, with no scene named by hand.
    """
    directory = tmp_path / "scenes"
    directory.mkdir()
    monkeypatch.setenv("CRAMERA_SCENES", str(directory))
    monkeypatch.delenv("CRAMERA_SCENE", raising=False)
    return directory


def write_index(directory: Path, default: str | None, scenes: list[dict]) -> None:
    """
    Write the scenes index that ``directory`` advertises to the viewer.
    """
    (directory / "index.json").write_text(
        json.dumps({"default": default, "scenes": scenes})
    )


# %% the scene the viewer opens on


class TestActiveScene:
    """
    Which scene :meth:`SceneBundle.active_name` picks out of a scenes index.
    """

    def test_a_declared_default_is_the_active_scene(self, scenes_directory: Path):
        write_index(
            scenes_directory, "G1_warehouse", [MONTESSORI_SCENE, WAREHOUSE_SCENE]
        )

        assert SceneBundle.active_name() == "G1_warehouse"

    def test_a_default_the_index_does_not_declare_falls_back_to_the_first_scene(
        self, scenes_directory: Path
    ):
        write_index(
            scenes_directory, "pr2_kitchen", [MONTESSORI_SCENE, WAREHOUSE_SCENE]
        )

        assert SceneBundle.active_name() == "Franka_Montessori"

    def test_an_index_declaring_no_scene_has_no_active_scene(
        self, scenes_directory: Path
    ):
        write_index(scenes_directory, "pr2_kitchen", [])

        assert SceneBundle.active_name() is None

    def test_a_scene_named_by_hand_wins_over_the_index(
        self, scenes_directory: Path, monkeypatch
    ):
        write_index(scenes_directory, "Franka_Montessori", [MONTESSORI_SCENE])
        monkeypatch.setenv("CRAMERA_SCENE", "G1_warehouse")

        assert SceneBundle.active_name() == "G1_warehouse"

    def test_without_an_index_there_is_no_active_scene(self, scenes_directory: Path):
        assert SceneBundle.active_name() is None
