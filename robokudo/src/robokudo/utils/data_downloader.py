from __future__ import annotations

# %% Imports
from dataclasses import dataclass
from pathlib import Path

import pooch
from filelock import FileLock


# %% Test dataset
@dataclass(frozen=True)
class TestDataset:
    """
    Identify the downloadable RoboKudo test dataset.
    """

    package_name: str = "robokudo_test_data"
    """
    Dataset archive and cache name.
    """

    revision: str = "859e8ef72d820de229bef3ef3e80f61e8be9643e"
    """
    Repository revision packaged in the archive.
    """

    checksum: str | None = (
        "sha256:370ea67e870ae5b25a7de46eab046167ae825836d223b5d2527d5a3194d872d5"
    )
    """
    Expected archive checksum.
    """


def test_data_path() -> Path:
    """
    Retrieve the extracted dataset directory, downloading it when absent.
    """
    dataset = TestDataset()
    cache_directory = Path(pooch.os_cache(dataset.package_name))
    cache_directory.mkdir(parents=True, exist_ok=True)
    with FileLock(cache_directory / f"{dataset.package_name}.zip.lock"):
        extracted_files = pooch.retrieve(
            url=(
                "https://gitlab.informatik.uni-bremen.de/robokudo/robokudo_test_data/-/jobs/artifacts/"
                f"{dataset.revision}/raw/{dataset.package_name}-{dataset.revision}.zip?job=package_zip"
            ),
            known_hash=dataset.checksum,
            path=cache_directory,
            fname=f"{dataset.package_name}.zip",
            downloader=pooch.HTTPDownloader(),
            processor=pooch.Unzip(),
        )
    return Path(extracted_files[0]).parent
