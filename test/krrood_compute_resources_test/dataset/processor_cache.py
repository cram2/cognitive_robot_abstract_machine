"""
A directory laid out as Linux describes the caches of the processor cores, which the
tests of the processor cache read.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass
from pathlib import Path

# %% caches


@dataclass(kw_only=True, frozen=True)
class DescribedCache:
    """
    One cache as Linux describes it in the directory of every core that uses it.
    """

    level: int
    """
    The level of the cache.
    """

    cache_type: str
    """
    What the cache holds, ``Data``, ``Instruction`` or ``Unified``.
    """

    size: str
    """
    The size as Linux writes it, such as ``16384K``.
    """

    shared_processors: str
    """
    The cores that use the cache as Linux lists them, such as ``0-3,8``.
    """


def describe_caches(
    *, directory: Path, caches_of_cores: dict[int, list[DescribedCache]]
) -> Path:
    """
    :param directory: An empty directory.
    :param caches_of_cores: The caches every core uses.
    :return: The directory, holding ``cpu<core>/cache/index<i>`` with the four files
        Linux describes every cache by.
    """
    for core, caches in caches_of_cores.items():
        for index, cache in enumerate(caches):
            cache_directory = directory / f"cpu{core}" / "cache" / f"index{index}"
            cache_directory.mkdir(parents=True)
            (cache_directory / "level").write_text(f"{cache.level}\n")
            (cache_directory / "type").write_text(f"{cache.cache_type}\n")
            (cache_directory / "size").write_text(f"{cache.size}\n")
            (cache_directory / "shared_cpu_list").write_text(
                f"{cache.shared_processors}\n"
            )
    return directory
