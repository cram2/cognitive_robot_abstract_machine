"""
Storing recorded attempts on disk, fetching a hosted set of them, and splitting them for
fitting and evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import requests
from krrood.adapters.json_serializer import from_json, to_json
from platformdirs import user_cache_dir
from typing_extensions import Any, Dict, List, Self

from experiments.causal_reasoning.comparison.dataset import EffectRate, ExampleDataset
from experiments.causal_reasoning.tracy_clutter_picking.domain import (
    ClutterPickScene,
    ClutterPickSceneAggregations,
    attempt_domain,
)
from experiments.causal_reasoning.tracy_clutter_picking.exceptions import (
    UnevenClutterError,
)


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

    def examples(self) -> ExampleDataset:
        """
        :return: The attempts as the comparison sees them.
        """
        return ExampleDataset(attempt_domain(), list(self.scenes))

    @property
    def recorded_neighbour_count(self) -> int:
        """
        How many neighbours every recorded attempt has.

        :raises UnevenClutterError: If the attempts do not all have the same number.
        """
        counts = {len(scene.neighbours) for scene in self.scenes}
        if len(counts) != 1:
            raise UnevenClutterError(sorted(counts))
        [count] = counts
        return count


def lift_summaries(dataset: ExampleDataset) -> Dict[str, Dict[Any, EffectRate]]:
    """
    How often the target was lifted, by the environment the clutter stood in, by the
    grasp's friction coefficient, and by how many neighbours stood adjacent to the
    target.

    :param dataset: The attempts.
    :return: The rates per summary's title.
    """
    return {
        "environment": dataset.effect_rate_by(lambda scene: scene.environment),
        "friction coefficient": dataset.effect_rate_by(
            lambda scene: scene.friction_coefficient
        ),
        "adjacent neighbours": dataset.effect_rate_by(
            lambda scene: ClutterPickSceneAggregations(instance=scene).crowding_count()
        ),
    }


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
