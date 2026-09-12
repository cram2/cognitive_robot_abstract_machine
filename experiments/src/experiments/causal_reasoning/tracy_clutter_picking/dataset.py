"""
Storing recorded attempts on disk, fetching a hosted set of them, and splitting them for
fitting and evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
from krrood.adapters.json_serializer import from_json, to_json
from platformdirs import user_cache_dir
from typing_extensions import Callable, Dict, List, Self, Tuple, TypeVar

from experiments.causal_reasoning.tracy_clutter_picking.domain import ClutterPickScene

T = TypeVar("T")


@dataclass(frozen=True)
class SuccessRate:
    """
    How often a group of attempts lifted its target.
    """

    attempt_count: int
    """
    How many attempts the group holds.
    """

    lifted_count: int
    """
    How many of them lifted the target.
    """

    @property
    def rate(self) -> float:
        """
        The lifted share.
        """
        return self.lifted_count / self.attempt_count


@dataclass
class ClutterPickDataset:
    """
    A set of recorded attempts, as written by data collection and read by the pipelines.
    """

    scenes: List[ClutterPickScene] = field(default_factory=list)
    """
    The recorded attempts.
    """

    def save(self, path: Path) -> None:
        """
        Write the attempts to a JSON file.

        :param path: Where to write; parent directories are created.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(to_json(self.scenes), indent=2))

    @classmethod
    def load(cls, path: Path) -> Self:
        """
        Read attempts written by :meth:`save`.

        :param path: The file to read.
        :return: The dataset.
        """
        return cls(scenes=from_json(json.loads(path.read_text())))

    @property
    def success_rate(self) -> float:
        """
        Share of attempts whose target was lifted.
        """
        return sum(scene.lifted for scene in self.scenes) / len(self.scenes)

    def success_rate_by(
        self, key: Callable[[ClutterPickScene], T]
    ) -> Dict[T, SuccessRate]:
        """
        The share of lifted targets among the attempts sharing a value.

        :param key: What to group the attempts by.
        :return: Each value's success rate, by value.
        """
        by_value: Dict[T, List[ClutterPickScene]] = {}
        for scene in self.scenes:
            by_value.setdefault(key(scene), []).append(scene)
        return {
            value: SuccessRate(
                attempt_count=len(scenes),
                lifted_count=sum(scene.lifted for scene in scenes),
            )
            for value, scenes in sorted(by_value.items())
        }

    def split(
        self, train_fraction: float, random_state: np.random.Generator
    ) -> Tuple[Self, Self]:
        """
        Shuffle the attempts and split them in two.

        :param train_fraction: Share of attempts that go into the first part.
        :param random_state: Source of randomness for the shuffle.
        :return: The first and second part.
        """
        order = random_state.permutation(len(self.scenes))
        split_index = int(train_fraction * len(self.scenes))
        first = [self.scenes[index] for index in order[:split_index]]
        second = [self.scenes[index] for index in order[split_index:]]
        return type(self)(first), type(self)(second)


# %% hosted dataset


@dataclass(frozen=True)
class HostedDataset:
    """
    A recorded dataset kept outside the repository and fetched into the user's cache the
    first time it is needed.
    """

    url: str
    """
    Where the dataset's JSON file is hosted.
    """

    cache_directory: Path = field(
        default_factory=lambda: Path(user_cache_dir(__package__.split(".", 1)[0]))
        / "tracy_clutter_picking"
    )
    """
    Where the fetched file is kept.
    """

    download_timeout: float = 60.0
    """
    How long, in seconds, to wait for the host before giving up.
    """

    download_chunk_size: int = 1 << 16
    """
    How many bytes of the file are written at a time while it is fetched.
    """

    @property
    def path(self) -> Path:
        """
        Where the fetched file lies, whether it has been fetched yet or not.
        """
        return self.cache_directory / Path(urlparse(self.url).path).name

    def fetch(self) -> Path:
        """
        Download the file unless it has been fetched before.

        :return: The fetched file.
        :raises requests.HTTPError: If the host does not serve the file.
        """
        if self.path.exists():
            return self.path
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        with requests.get(
            self.url, stream=True, timeout=self.download_timeout
        ) as response:
            response.raise_for_status()
            with self.path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=self.download_chunk_size):
                    file.write(chunk)
        return self.path

    def load(self) -> ClutterPickDataset:
        """
        :return: The hosted attempts, fetched if need be.
        """
        return ClutterPickDataset.load(self.fetch())
