import shutil
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import pooch
import pytest
from typing_extensions import Callable

import robokudo.utils.data_downloader

# %% simulated test data archive

UNPACKING_DELAY_SECONDS = 0.5
"""
How long the first caller takes before it writes any unpacked file, which leaves a second caller
time to find the test data downloaded but not yet unpacked.
"""

ARCHIVED_FILE_NAME = "rk_1669996652.766856_depth_image.png"
"""
The one file inside the simulated test data archive.
"""


@dataclass
class CopyingDownloader:
    """
    Stands in for :class:`pooch.HTTPDownloader` by copying a local archive.
    """

    archive: Path
    """
    The archive every download delivers.
    """

    def __call__(self, url: str, output_file: str, pooch_instance: pooch.Pooch) -> None:
        shutil.copyfile(self.archive, output_file)


@dataclass
class UnpackingRecorder:
    """
    Records each time :class:`pooch.Unzip` unpacks an archive, and delays the first unpacking.
    """

    unpack: Callable[[pooch.Unzip, str, str], None]
    """
    The unpacking it records.
    """

    unpacking_count: int = 0
    """
    How many times an archive has been unpacked.
    """

    first_unpacking_started: threading.Event = field(default_factory=threading.Event)
    """
    Set once the first unpacking has begun.
    """

    def unpack_recorded(
        self, unzip: pooch.Unzip, archive: str, extract_dir: str
    ) -> None:
        """
        Unpack ``archive`` into ``extract_dir`` as ``unzip`` would, and record that it happened.
        """
        self.unpacking_count += 1
        if self.unpacking_count == 1:
            self.first_unpacking_started.set()
            time.sleep(UNPACKING_DELAY_SECONDS)
        self.unpack(unzip, archive, extract_dir)


@pytest.fixture
def unpacking_recorder(tmp_path, monkeypatch) -> UnpackingRecorder:
    """
    Make :func:`robokudo.utils.data_downloader.test_data_path` fetch a local archive into a fresh
    cache, as a CI run starting without the test data does, and record its unpacking.
    """
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr(ARCHIVED_FILE_NAME, b"depth")

    recorder = UnpackingRecorder(unpack=pooch.Unzip._extract_file)
    monkeypatch.setattr(robokudo.utils.data_downloader, "KNOWN_HASH", None)
    monkeypatch.setattr(pooch, "os_cache", lambda package_name: tmp_path / "cache")
    monkeypatch.setattr(pooch, "HTTPDownloader", lambda: CopyingDownloader(archive))
    monkeypatch.setattr(
        pooch.Unzip,
        "_extract_file",
        lambda unzip, archive, extract_dir: recorder.unpack_recorded(
            unzip, archive, extract_dir
        ),
    )
    return recorder


# %% concurrent callers


def test_a_caller_arriving_during_unpacking_does_not_unpack_again(
    unpacking_recorder,
):
    with ThreadPoolExecutor(max_workers=2) as executor:
        first_caller = executor.submit(robokudo.utils.data_downloader.test_data_path)
        assert unpacking_recorder.first_unpacking_started.wait(timeout=60)
        second_caller = executor.submit(robokudo.utils.data_downloader.test_data_path)
        first_caller.result()
        second_caller.result()

    assert unpacking_recorder.unpacking_count == 1
