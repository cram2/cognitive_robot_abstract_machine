"""
Tests of working through items in several worker processes at once.
"""

from __future__ import annotations

# %% imports
import multiprocessing
import os
import signal
import time

import pytest

from krrood_compute_resources.exceptions import NotEnoughMemoryForWorkers
from krrood_compute_resources.workers import WorkerPool
from .dataset.jobs import NumbersWrittenToAFile
from .dataset.memory import FixedFreeMemory

# %% dividing a job


def test_a_single_worker_works_through_every_item_in_order(tmp_path):
    """
    A pool of one worker works through the items in this process, in order.
    """
    job = NumbersWrittenToAFile(path=tmp_path / "numbers.txt")
    numbers = list(range(5))

    assert WorkerPool().run(job=job, items=numbers) == numbers
    assert job.written_numbers() == numbers


def test_several_workers_work_through_every_item(tmp_path):
    """
    Two workers in processes of their own work through every item once.
    """
    job = NumbersWrittenToAFile(path=tmp_path / "numbers.txt")
    numbers = list(range(8))

    results = WorkerPool(number_of_workers=2).run(job=job, items=numbers)

    assert sorted(results) == numbers
    assert sorted(job.written_numbers()) == numbers


# %% memory


def test_the_workers_are_as_many_as_fit_into_the_free_memory():
    """
    Of more workers than the memory holds, only as many run as fit into it.
    """
    pool = WorkerPool(
        number_of_workers=8,
        memory_per_worker=10**6,
        memory=FixedFreeMemory(free=3 * 10**6),
    )
    assert pool.workers_that_fit() == pool.memory.free // pool.memory_per_worker


def test_no_more_workers_run_than_asked_for():
    """
    However much memory is free, no more workers run than asked for.
    """
    pool = WorkerPool(
        number_of_workers=2, memory_per_worker=1, memory=FixedFreeMemory(free=10**9)
    )
    assert pool.workers_that_fit() == pool.number_of_workers


def test_a_pool_whose_worker_does_not_fit_is_refused(tmp_path):
    """
    When not even one worker fits into the free memory, no item is worked through.
    """
    job = NumbersWrittenToAFile(path=tmp_path / "numbers.txt")
    pool = WorkerPool(
        number_of_workers=2, memory_per_worker=10**6, memory=FixedFreeMemory(free=1)
    )
    with pytest.raises(NotEnoughMemoryForWorkers):
        pool.run(job=job, items=[1, 2])
    assert job.written_numbers() == []


# %% stopping


def run_pool(*, job: NumbersWrittenToAFile, items: list[int]) -> None:
    """
    Runs a pool of two workers over the items, as the process a test terminates.

    :param job: The job to run.
    :param items: What it works through.
    """
    WorkerPool(number_of_workers=2).run(job=job, items=items)


def test_a_terminated_pool_stops_cleanly_and_keeps_what_is_done(tmp_path):
    """
    Asked to terminate, the process of a pool lets its workers finish their items and
    stop: it ends without an error, keeps the items done, and no worker writes on.
    """
    job = NumbersWrittenToAFile(path=tmp_path / "numbers.txt", seconds_per_number=0.2)
    items = list(range(40))
    process = multiprocessing.get_context("spawn").Process(
        target=run_pool, kwargs={"job": job, "items": items}
    )
    process.start()
    deadline = time.monotonic() + 60
    while len(job.written_numbers()) < 2 and time.monotonic() < deadline:
        time.sleep(0.1)

    os.kill(process.pid, signal.SIGTERM)
    process.join(timeout=30)
    written = len(job.written_numbers())
    time.sleep(1)

    assert process.exitcode == 0
    assert 2 <= written < len(items)
    assert len(job.written_numbers()) == written
