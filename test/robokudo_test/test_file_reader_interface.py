from __future__ import annotations

# %% Imports

from pathlib import Path

import cv2
import numpy as np
import pytest

from robokudo.cas import CASViews
from robokudo.descriptors.camera_configs.config_filereader_playback import (
    FileReaderCameraConfig,
)
from robokudo.io.file_reader_interface import FileReaderInterface
from robokudo.io.exceptions import InvalidFileSource

# %% Playback


@pytest.fixture
def file_reader(tmp_path: Path) -> FileReaderInterface:
    """
    Provide two recorded frames with distinct color images.
    """
    configuration = FileReaderCameraConfig(target_dir=str(tmp_path), loop=False)
    for timestamp in reversed(range(2)):
        image_path = tmp_path / (
            f"{configuration.filename_prefix}{timestamp}_{CASViews.COLOR_IMAGE}.png"
        )
        assert cv2.imwrite(
            str(image_path), np.full((2, 2, 3), timestamp, dtype=np.uint8)
        )
    return FileReaderInterface(configuration)


def test_reads_frames_in_file_order(file_reader: FileReaderInterface) -> None:
    """
    Playback follows sorted filenames and exposes each frame once.
    """
    for expected_frame in file_reader.loaded_data.values():
        assert file_reader.has_new_data() is True
        assert file_reader.has_new_data() is True
        assert file_reader.get_next_data() is expected_frame
    assert file_reader.has_new_data() is False
    assert file_reader.get_next_data() is None
    assert file_reader.get_next_data() is None


def test_restarts_loop_after_last_frame(file_reader: FileReaderInterface) -> None:
    """
    Looping restarts playback after every frame has been consumed.
    """
    file_reader.camera_config.loop = True
    for expected_frame in file_reader.loaded_data.values():
        assert file_reader.get_next_data() is expected_frame
    assert file_reader.has_new_data() is True
    assert file_reader.get_next_data() is next(iter(file_reader.loaded_data.values()))


@pytest.mark.parametrize("loop", [False, True])
def test_empty_directory_has_no_frames(tmp_path: Path, loop: bool) -> None:
    """
    An empty recording stays exhausted even when looping is enabled.
    """
    reader = FileReaderInterface(
        FileReaderCameraConfig(target_dir=str(tmp_path), loop=loop)
    )
    assert reader.has_new_data() is False
    assert reader.get_next_data() is None


# %% Filename validation


def test_ignores_legacy_camera_info_file(tmp_path: Path) -> None:
    """
    Legacy camera-info files do not create playback frames.
    """
    configuration = FileReaderCameraConfig(target_dir=str(tmp_path))
    (tmp_path / f"{configuration.filename_prefix}0_cam_info.json").touch()

    reader = FileReaderInterface(configuration)

    assert reader.loaded_paths == {}
    assert reader.loaded_data == {}
    assert reader.has_new_data() is False


def test_rejects_unknown_view_suffix(tmp_path: Path) -> None:
    """
    A recording with an unknown view suffix is an invalid file source.
    """
    configuration = FileReaderCameraConfig(target_dir=str(tmp_path))
    (tmp_path / f"{configuration.filename_prefix}0_unknown_view.png").touch()

    with pytest.raises(InvalidFileSource):
        FileReaderInterface(configuration)


def test_recording_keys_are_views(file_reader: FileReaderInterface) -> None:
    """
    Loaded paths and data use validated view enum members as keys.
    """
    for recording in (file_reader.loaded_paths, file_reader.loaded_data):
        for frame in recording.values():
            assert list(frame) == [CASViews.COLOR_IMAGE]
            assert all(isinstance(view, CASViews) for view in frame)
