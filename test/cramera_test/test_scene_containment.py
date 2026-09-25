"""
Scene lookup and storage names stay within their configured directories.
"""

from pathlib import Path

import pytest
from pytest import MonkeyPatch

from cramera import paths
from cramera.onboard.scene_index import InvalidSceneName, validate_scene_name


# %% scene names
@pytest.mark.parametrize("name", ["episode\n", "__live__\n", "__recording__\n"])
def test_scene_name_rejects_a_trailing_newline(name: str) -> None:
    """
    Whitespace cannot escape the name alphabet or reserved-name comparison.

    :param name: Name accepted by a partial regex match.
    """
    with pytest.raises(InvalidSceneName):
        validate_scene_name(name)


@pytest.mark.parametrize("reference", ["parent", "absolute", "symlink"])
def test_scene_lookup_cannot_leave_configured_roots(
    tmp_path: Path, monkeypatch: MonkeyPatch, reference: str
) -> None:
    """
    A neighboring scene cannot be selected through a path or symlink escape.

    :param tmp_path: Isolated data directory and external scene.
    :param monkeypatch: Configures the permitted recording root.
    :param reference: Form of the attempted outside-root lookup.
    """
    monkeypatch.setenv("CRAMERA_DATA", str(tmp_path))
    monkeypatch.setenv("CRAMERA_SCENES", str(tmp_path / "scenes"))
    scenes = tmp_path / "scenes"
    outside = tmp_path / "outside"
    scenes.mkdir()
    outside.mkdir()
    (outside / "scene.json").write_text("{}")
    if reference == "symlink":
        (scenes / "alias").symlink_to(outside, target_is_directory=True)
    name = {"parent": "../outside", "absolute": str(outside), "symlink": "alias"}[
        reference
    ]
    assert paths.resolve_scene_directory(name) is None


@pytest.mark.parametrize(
    "name", [paths.LIVE_SCENE_NAME, paths.RECORDING_SCENE_NAME, "saved_scene-1"]
)
def test_lookup_preserves_reserved_and_saved_scenes(
    tmp_path: Path, monkeypatch: MonkeyPatch, name: str
) -> None:
    """
    Internal live captures and ordinary recordings remain addressable by name.

    :param tmp_path: Isolated recording directory.
    :param monkeypatch: Configures the permitted root.
    :param name: Valid internal or saved scene name.
    """
    monkeypatch.setenv("CRAMERA_DATA", str(tmp_path))
    monkeypatch.setenv("CRAMERA_SCENES", str(tmp_path / "scenes"))
    scene = tmp_path / "scenes" / name
    scene.mkdir(parents=True)
    (scene / "scene.json").write_text("{}")
    assert paths.resolve_scene_directory(name) == scene
