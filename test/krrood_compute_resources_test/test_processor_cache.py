"""
Tests of the cache of the processor cores a process runs on, as memory a computation can
keep its working set in.
"""

from __future__ import annotations

# %% imports
import os
from pathlib import Path

import pytest

from krrood_compute_resources.processor_cache import CacheLevel, ProcessorCache

from .dataset.processor_cache import DescribedCache, describe_caches

# %% fixtures


@pytest.fixture
def two_core_complexes(tmp_path: Path) -> Path:
    """
    :return: A description of eight cores in two complexes of four, each complex sharing
        a third level cache of 16 MiB, every core with its own second level cache of
        512 KiB and first level data and instruction caches of 32 KiB.
    """
    caches_of_cores = {}
    for core in range(8):
        complex_cores = "0-3" if core < 4 else "4-7"
        caches_of_cores[core] = [
            DescribedCache(
                level=1, cache_type="Data", size="32K", shared_processors=str(core)
            ),
            DescribedCache(
                level=1,
                cache_type="Instruction",
                size="32K",
                shared_processors=str(core),
            ),
            DescribedCache(
                level=2, cache_type="Unified", size="512K", shared_processors=str(core)
            ),
            DescribedCache(
                level=3,
                cache_type="Unified",
                size="16384K",
                shared_processors=complex_cores,
            ),
        ]
    return describe_caches(directory=tmp_path, caches_of_cores=caches_of_cores)


# %% capacity


def test_cores_that_share_a_cache_count_it_once(two_core_complexes):
    """
    Two cores of one complex and one of the other use two third level caches.
    """
    cache = ProcessorCache(
        level=CacheLevel.THIRD,
        processor_cores={0, 1, 4},
        system_directory=two_core_complexes,
    )

    assert cache.free_bytes() == 2 * 16384 * 1024


def test_a_cache_of_every_core_counts_once_per_core(two_core_complexes):
    """
    Three cores use three second level caches of their own.
    """
    cache = ProcessorCache(
        level=CacheLevel.SECOND,
        processor_cores={0, 1, 4},
        system_directory=two_core_complexes,
    )

    assert cache.free_bytes() == 3 * 512 * 1024


def test_instruction_caches_hold_no_working_set(two_core_complexes):
    """
    At the first level, only the data cache of every core counts.
    """
    cache = ProcessorCache(
        level=CacheLevel.FIRST, processor_cores={0}, system_directory=two_core_complexes
    )

    assert cache.free_bytes() == 32 * 1024


def test_the_processor_cache_of_this_process_holds_something():
    """
    On a Linux machine that describes its caches, the cores of this process use some
    third level cache.
    """
    if not Path(
        f"/sys/devices/system/cpu/cpu{min(os.sched_getaffinity(0))}/cache"
    ).exists():
        pytest.skip("the machine does not describe its caches")

    assert ProcessorCache(level=CacheLevel.THIRD).free_bytes() > 0
