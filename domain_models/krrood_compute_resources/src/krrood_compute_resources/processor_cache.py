"""
The caches of the processor cores a process runs on, as memory a computation can keep
its working set in so that it does not wait for the main memory.
"""

from __future__ import annotations

# %% imports
import os
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from pathlib import Path

from krrood.patterns.path_enumeration import PathEnumeration
from typing_extensions import Self

from krrood_compute_resources.memory import DeviceMemory

# %% how Linux describes a cache


class LinuxProcessorDirectory(PathEnumeration):
    """
    The directories Linux describes the processor cores in.
    """

    PROCESSOR_CORES = "/sys/devices/system/cpu"
    """
    The directory holding ``cpu<core>`` for every core.
    """


class CacheLevel(IntEnum):
    """
    The levels of the cache hierarchy of a processor, the first closest to a core.
    """

    FIRST = 1
    """
    The smallest and fastest caches, one of each per core.
    """

    SECOND = 2
    """
    The caches behind the first level.
    """

    THIRD = 3
    """
    The largest caches, usually shared by several cores.
    """


class CacheContent(StrEnum):
    """
    What a cache holds, as Linux writes it in the ``type`` file of the cache.
    """

    DATA = "Data"
    """
    The data the cores work on.
    """

    INSTRUCTION = "Instruction"
    """
    The instructions the cores run.
    """

    UNIFIED = "Unified"
    """
    Data and instructions.
    """


class CacheDescriptionFile(StrEnum):
    """
    The files Linux describes a cache by, in ``cpu<core>/cache/index<i>``.
    """

    LEVEL = "level"
    """
    The level of the cache.
    """

    CONTENT = "type"
    """
    What the cache holds.
    """

    SIZE = "size"
    """
    The capacity, a number followed by its unit.
    """

    SHARED_PROCESSORS = "shared_cpu_list"
    """
    The cores that use the cache, as ranges separated by commas.
    """


class CacheSizeUnit(StrEnum):
    """
    The units Linux writes the size of a cache in, as the suffix of the number.
    """

    KIBIBYTE = "K"
    """
    1024 bytes.
    """

    MEBIBYTE = "M"
    """
    1024 kibibytes.
    """

    @property
    def number_of_bytes(self) -> int:
        """
        :return: The bytes of one unit.
        """
        if self is CacheSizeUnit.KIBIBYTE:
            return 1024
        return 1024 * 1024


# %% one cache


@dataclass(kw_only=True, frozen=True)
class CacheInstance:
    """
    One cache of a processor, which the cores that use it share.
    """

    level: int
    """
    The level of the cache.
    """

    content: CacheContent
    """
    What the cache holds.
    """

    size_bytes: int
    """
    The capacity of the cache.
    """

    shared_processors: frozenset[int]
    """
    The cores that use the cache.
    """

    @classmethod
    def from_directory(cls, *, directory: Path) -> Self:
        """
        :param directory: The directory ``cpu<core>/cache/index<i>`` describing the
            cache.
        :return: The cache it describes.
        """

        def read(description_file: CacheDescriptionFile) -> str:
            """
            :param description_file: A file of the directory.
            :return: Its content without the line break.
            """
            return (directory / description_file).read_text().strip()

        size = read(CacheDescriptionFile.SIZE)
        return cls(
            level=int(read(CacheDescriptionFile.LEVEL)),
            content=CacheContent(read(CacheDescriptionFile.CONTENT)),
            size_bytes=int(size[:-1]) * CacheSizeUnit(size[-1]).number_of_bytes,
            shared_processors=frozenset(
                core
                for part in read(CacheDescriptionFile.SHARED_PROCESSORS).split(",")
                for core in range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1)
            ),
        )

    @property
    def holds_data(self) -> bool:
        """
        :return: Whether the cache can hold the data a computation works on.
        """
        return self.content is not CacheContent.INSTRUCTION


# %% caches of the cores of a process


@dataclass(kw_only=True)
class ProcessorCache(DeviceMemory):
    """
    The data caches of one level that the processor cores of a process use, each cache
    counted once however many of the cores share it; their free bytes are the capacity
    of these caches together, which a working set of that size fits in.
    """

    level: CacheLevel = CacheLevel.THIRD
    """
    The level of the caches.
    """

    processor_cores: set[int] = field(default_factory=lambda: os.sched_getaffinity(0))
    """
    The cores whose caches count, those the current process may run on by default.
    """

    system_directory: Path = LinuxProcessorDirectory.PROCESSOR_CORES
    """
    The directory in which Linux describes the cores, ``cpu<core>`` each.
    """

    def caches(self) -> set[CacheInstance]:
        """
        :return: Every data cache of :attr:`level` that one of the cores uses.
        """
        return {
            cache
            for core in self.processor_cores
            for cache in (
                CacheInstance.from_directory(directory=directory)
                for directory in (self.system_directory / f"cpu{core}" / "cache").glob(
                    "index*"
                )
            )
            if cache.level == self.level and cache.holds_data
        }

    def free_bytes(self) -> int:
        return sum(cache.size_bytes for cache in self.caches())
