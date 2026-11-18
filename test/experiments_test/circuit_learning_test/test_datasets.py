"""
Tests of how the benchmark datasets are downloaded, cached and read.
"""

from __future__ import annotations

# %% imports
import io
import pickle
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.circuit_learning.datasets import (
    BinaryDatasetFile,
    BinaryDatasetSource,
    LegacyPandasUnpickler,
    MiniBooNESource,
    PandasModule,
    RemovedPandasIndex,
)

# %% fixtures


@dataclass(kw_only=True)
class RecordedDownloads:
    """
    Serves fixed content for every address and records the addresses asked for.
    """

    content: bytes
    """
    The content served for every address.
    """

    addresses: list[str] = field(default_factory=list)
    """
    The addresses asked for, in order.
    """

    def __call__(self, url: str) -> io.BytesIO:
        self.addresses.append(url)
        return io.BytesIO(self.content)


def archive_with_miniboone(*, rows: np.ndarray) -> bytes:
    """
    :param rows: The rows of MINIBOONE.
    :return: A gzipped tar archive laid out like the archive of the paper.
    """
    data = io.BytesIO()
    np.save(data, rows)
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as file:
        member = tarfile.TarInfo("data/miniboone/data.npy")
        member.size = len(data.getvalue())
        file.addfile(member, io.BytesIO(data.getvalue()))
    return archive.getvalue()


# %% download cache


def test_binary_file_is_downloaded_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    A loaded binary dataset is read from the cache on the next load.
    """
    downloads = RecordedDownloads(content=b"0,1\n1,0\n")
    monkeypatch.setattr(urllib.request, "urlopen", downloads)
    source = BinaryDatasetSource(
        cache_directory=tmp_path, dataset=BinaryDatasetFile.NLTCS
    )
    source.load()
    downloads_of_first_load = len(downloads.addresses)
    source.load()
    assert len(downloads.addresses) == downloads_of_first_load


def test_archive_is_downloaded_and_unpacked_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    The archive is downloaded and unpacked on the first load only.
    """
    rows = np.arange(200, dtype=float).reshape(100, 2)
    downloads = RecordedDownloads(content=archive_with_miniboone(rows=rows))
    monkeypatch.setattr(urllib.request, "urlopen", downloads)
    source = MiniBooNESource(cache_directory=tmp_path)
    source.load()
    source.load()
    assert downloads.addresses == [source.archive_url]
    np.testing.assert_array_equal(
        np.load(source.data_directory / "miniboone" / "data.npy"), rows
    )


# %% pickles of old pandas versions


@pytest.mark.parametrize("index", list(RemovedPandasIndex))
def test_removed_index_class_is_read_as_index(index: RemovedPandasIndex):
    """
    An index class that pandas removed is read as :class:`pandas.Index`.
    """
    reference = f"c{PandasModule.REMOVED_INDEX}\n{index}\n.".encode()
    assert LegacyPandasUnpickler(io.BytesIO(reference)).load() is pd.Index


def test_current_classes_are_read_unchanged():
    """
    Classes outside the removed index module are found where they are.
    """
    assert (
        LegacyPandasUnpickler(io.BytesIO(pickle.dumps(pd.DataFrame))).load()
        is pd.DataFrame
    )
