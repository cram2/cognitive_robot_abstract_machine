"""
Damaged metadata remains visible without hiding valid scene recordings.
"""

import json
from pathlib import Path

import pytest
from pytest import LogCaptureFixture

from cramera.onboard.scene_index import SceneIndexEntry, write_scene_index

from .test_scene_index import _write_bundle


# %% damaged generated metadata
@pytest.mark.parametrize("content", [b"invalid JSON", b"[]"])
def test_bad_neighbor_does_not_hide_valid_scenes(
    tmp_path: Path, caplog: LogCaptureFixture, content: bytes
) -> None:
    """
    Unreadable or non-object metadata is reported while valid bundles stay listed.

    :param tmp_path: Directory containing valid and damaged scene neighbors.
    :param caplog: Captures the visible diagnostic for the damaged metadata.
    :param content: Unusable metadata bytes that must remain unchanged.
    """
    _write_bundle(tmp_path, "valid")
    damaged = tmp_path / "damaged" / "scene.json"
    damaged.parent.mkdir()
    damaged.write_bytes(content)
    assert [entry.name for entry in SceneIndexEntry.of_directory(tmp_path)] == ["valid"]
    assert damaged.read_bytes() == content
    assert str(damaged) in caplog.text


@pytest.mark.parametrize("content", [b"invalid JSON", b"[]"])
def test_rebuilt_index_preserves_the_damaged_original(
    tmp_path: Path, caplog: LogCaptureFixture, content: bytes
) -> None:
    """
    Saving a valid scene rebuilds a bad index and keeps the original for inspection.

    :param tmp_path: Directory containing a valid recording and damaged index.
    :param caplog: Captures the diagnostic locating the preserved original.
    :param content: Unusable index bytes that must be recoverable afterwards.
    """
    _write_bundle(tmp_path, "valid")
    index = tmp_path / "index.json"
    index.write_bytes(content)
    write_scene_index(index, "valid")
    rebuilt = json.loads(index.read_text())
    assert rebuilt["default"] == "valid"
    assert [entry["name"] for entry in rebuilt["scenes"]] == ["valid"]
    [preserved] = tmp_path.glob(index.name + ".corrupt-*")
    assert preserved.read_bytes() == content
    assert str(preserved) in caplog.text
