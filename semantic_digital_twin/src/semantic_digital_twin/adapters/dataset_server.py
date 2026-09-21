from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from http import HTTPStatus
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import requests
from typing_extensions import Any, Dict, List, Optional, Self

from semantic_digital_twin.adapters.package_resolver import PathResolver
from semantic_digital_twin.exceptions import DatasetServerError, PathResolutionError
from semantic_digital_twin.utils import create_cache_dir

# %% the listing the server answers a directory with


class ListedItemKind(StrEnum):
    """
    The kinds of item a directory listing distinguishes.
    """

    FILE = "file"
    DIRECTORY = "directory"


@dataclass
class ListedItem:
    """
    One item of a directory listing.

    The listing is the server's, so its field names and the path into it are written
    here once rather than at every use.
    """

    name: str
    """
    The item's name within the directory that was listed.
    """

    kind: ListedItemKind
    """
    Whether the item is a file or a directory.
    """

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> Self:
        """
        :param data: One element of a listing.
        :return: The item it describes.
        """
        return cls(name=data["name"], kind=ListedItemKind(data["type"]))


# %% reaching a dataset that is served over http


class DatasetServerVariable(StrEnum):
    """
    Environment variables describing the dataset server to read from.
    """

    BASE_URL = "SEMANTIC_DIGITAL_TWIN_DATASET_SERVER"
    DATASET_ROOT = "SEMANTIC_DIGITAL_TWIN_DATASET_ROOT"
    CACHE_DIRECTORY = "SEMANTIC_DIGITAL_TWIN_MESH_CACHE"


@dataclass
class DatasetServer(PathResolver):
    """
    A dataset that lives on another machine and is read over http.

    A mesh is a directory rather than a file -- the material and texture files beside it
    are found by name relative to it -- so a reference is answered by copying the whole
    directory containing it into :attr:`cache` and answering with the copy. Every
    consumer that expects a readable path therefore gets one, and the names inside the
    directory are the ones the mesh refers to.

    ..note:: A cached directory is never checked against the server again, which holds
        because a dataset addressed by the content hash of its entries never changes one.
    """

    base_url: str
    """
    The address serving :attr:`dataset_root`.
    """

    dataset_root: PurePosixPath
    """
    The dataset's location on the machine holding it, which is what the references
    recorded against it are written relative to.
    """

    cache: Path = field(default_factory=lambda: create_cache_dir("dataset_server"))
    """
    The directory the dataset's files are copied into on this machine, defaulting to the
    one this package keeps everything else it downloads in.
    """

    session: requests.Session = field(default_factory=requests.Session)
    """
    The connection the requests are made on, reused so a directory and its files cost
    one handshake between them rather than one each.
    """

    timeout: timedelta = timedelta(seconds=30)
    """
    How long to wait for a single request.
    """

    completion_marker: str = ".entry-complete"
    """
    Written into a cached directory once every file in it has arrived, so a directory
    left behind by an interrupted copy is fetched again rather than read short.
    """

    @classmethod
    def from_environment(cls) -> Optional[Self]:
        """
        Build the server the environment describes.

        :return: The server, or ``None`` where no address is set.
        """
        base_url = os.environ.get(DatasetServerVariable.BASE_URL)
        if not base_url:
            return None
        dataset_root = os.environ.get(
            DatasetServerVariable.DATASET_ROOT, "/raid/users/tom_sch/datasets"
        )
        server = cls(base_url=base_url, dataset_root=PurePosixPath(dataset_root))
        cache = os.environ.get(DatasetServerVariable.CACHE_DIRECTORY)
        if cache:
            server.cache = Path(cache)
        return server

    def supports(self, uri: str) -> bool:
        return uri.startswith(f"{self.dataset_root}/")

    def resolve(self, uri: str) -> str:
        if not self.supports(uri):
            raise PathResolutionError(uri=uri, details=f"not below {self.dataset_root}")
        relative_file = PurePosixPath(uri).relative_to(self.dataset_root)
        cached_directory = self.cache / relative_file.parent
        if not (cached_directory / self.completion_marker).is_file():
            self._copy_directory(relative_file.parent, cached_directory)
        return str(cached_directory / relative_file.name)

    def list_directory(self, relative_directory: PurePosixPath) -> List[ListedItem]:
        """
        :param relative_directory: The directory to list, relative to the dataset root.
        :return: What the server reports the directory holds.
        """
        response = self._get(f"{self._url(relative_directory)}/")
        return [ListedItem.from_json(item) for item in response.json()]

    def _copy_directory(
        self, relative_directory: PurePosixPath, destination: Path
    ) -> None:
        """
        Copy every file of a directory into the cache.

        Each file is written under a temporary name and moved into place, and the marker
        is written last, so a copy that is interrupted leaves nothing that reads as
        complete.

        :param relative_directory: The directory to copy, relative to the dataset root.
        :param destination: The directory to copy it into.
        """
        destination.mkdir(parents=True, exist_ok=True)
        for item in self.list_directory(relative_directory):
            if item.kind is not ListedItemKind.FILE:
                continue
            self._copy_file(relative_directory / item.name, destination / item.name)
        (destination / self.completion_marker).write_bytes(b"")

    def _copy_file(self, relative_file: PurePosixPath, destination: Path) -> None:
        """
        :param relative_file: The file to copy, relative to the dataset root.
        :param destination: The path to write it to.
        """
        response = self._get(self._url(relative_file))
        partial = destination.with_name(f"{destination.name}.partial")
        partial.write_bytes(response.content)
        os.replace(partial, destination)

    def _url(self, relative_path: PurePosixPath) -> str:
        """
        :param relative_path: A path relative to the dataset root.
        :return: The address serving it.
        """
        return f"{self.base_url.rstrip('/')}/{quote(str(relative_path))}"

    def _get(self, url: str) -> requests.Response:
        """
        :param url: The address to request.
        :return: The answer, which is an answer of success.
        """
        response = self.session.get(url, timeout=self.timeout.total_seconds())
        if response.status_code != HTTPStatus.OK:
            raise DatasetServerError(url=url, status_code=response.status_code)
        return response
