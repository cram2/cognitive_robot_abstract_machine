"""
Pictures and readings a scene takes a while to compute, kept on disk so a video is
rendered again without recomputing them.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import cv2
from typing_extensions import Any

from crollywood.frame import Frame

# %% where the cache lies


class EnvironmentVariable(StrEnum):
    """
    The environment variables the package reads.
    """

    CACHE_ROOT = "CROLLYWOOD_CACHE"
    """
    Where the video keeps what it computed.
    """


# %% one scene's directory of the cache


@dataclass
class SceneCache:
    """
    One scene's kept pictures and readings, under a name of its own.
    """

    name: str
    """
    What the scene's directory of the cache is called.
    """

    root: Path = field(default_factory=lambda: SceneCache.configured_root())
    """
    Where every scene's directory lies.
    """

    @staticmethod
    def configured_root(
        fallback: Path = Path.home() / ".cache" / "crollywood",
    ) -> Path:
        """
        :param fallback: Where the video keeps what it computed when the environment
            does not say.
        :return: Where the video keeps what it computed, as
            :attr:`EnvironmentVariable.CACHE_ROOT` says or else the fallback.
        """
        return Path(os.getenv(EnvironmentVariable.CACHE_ROOT, fallback))

    @property
    def directory(self) -> Path:
        """
        This scene's directory under the root, made if it is not there yet.
        """
        directory = self.root / self.name
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def picture(self, key: str) -> Frame | None:
        """
        :param key: What the picture was kept as.
        :return: The picture as red, green and blue, or None if none was kept.
        """
        path = self.directory / f"{key}.png"
        if not path.is_file():
            return None
        return Frame(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))

    def keep_picture(self, key: str, picture: Frame) -> None:
        """
        :param key: What to keep the picture as.
        :param picture: The picture, as red, green and blue.
        """
        cv2.imwrite(
            str(self.directory / f"{key}.png"),
            cv2.cvtColor(picture.pixels, cv2.COLOR_RGB2BGR),
        )

    def record(self, key: str) -> dict[str, Any] | None:
        """
        :param key: What the readings were kept as.
        :return: The readings, or None if none were kept.
        """
        path = self.directory / f"{key}.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text())

    def keep_record(self, key: str, readings: dict[str, Any]) -> None:
        """
        :param key: What to keep the readings as.
        :param readings: Readings JSON can carry.
        """
        (self.directory / f"{key}.json").write_text(json.dumps(readings, indent=1))
