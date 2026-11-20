"""
Working through items in several worker processes at once, as many as fit into the
memory of the host, stopping cleanly when asked to.
"""

from __future__ import annotations

import signal
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from multiprocessing import get_context
from multiprocessing.managers import EventProxy

from typing_extensions import Any, Generic, TypeVar

from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood_compute_resources.exceptions import NotEnoughMemoryForWorkers
from krrood_compute_resources.memory import DeviceMemory, HostMemory

TItem = TypeVar("TItem")
"""
What a job works through, one at a time.
"""

TResult = TypeVar("TResult")
"""
What working through one item gives.
"""

# %% a job divided among workers


@dataclass(kw_only=True)
class ParallelJob(Generic[TItem, TResult], SubClassSafeGeneric, ABC):
    """
    Work that is divided among workers, each working through a share of the items.

    It is sent to each worker's process, so it holds only what can be pickled.
    """

    @abstractmethod
    def work_through(self, *, items: list[TItem], stop: EventProxy) -> list[TResult]:
        """
        :param items: The share of the items of one worker, in order.
        :param stop: Set when the job is to stop; checked before each item.
        :return: The result of every item worked through, in order.
        """


# %% workers


@dataclass(kw_only=True)
class WorkerPool:
    """
    Runs a job in as many worker processes as fit into the free memory of the host.

    An interrupt or a termination signal to this process lets every worker finish its
    current item and stop, keeping what is done.
    """

    number_of_workers: int = 1
    """
    How many processes work at once at most; a single worker works in this process.
    """

    memory_per_worker: int | None = None
    """
    How much memory a worker needs at most, in bytes; ``None`` when the workers are not
    sized by memory.
    """

    memory: DeviceMemory = field(default_factory=HostMemory)
    """
    The memory the workers share, such as a :class:`BudgetedMemory` of the host to leave
    room for other processes.
    """

    def workers_that_fit(self) -> int:
        """
        :return: How many workers run: :attr:`number_of_workers`, or fewer where not
            that many fit into the free bytes of :attr:`memory`.
        :raises NotEnoughMemoryForWorkers: If not even one worker fits.
        """
        if self.memory_per_worker is None:
            return self.number_of_workers
        free_bytes = self.memory.free_bytes()
        fitting = min(self.number_of_workers, free_bytes // self.memory_per_worker)
        if fitting == 0:
            raise NotEnoughMemoryForWorkers(
                memory_per_worker=self.memory_per_worker, free_bytes=free_bytes
            )
        return fitting

    def run(
        self, *, job: ParallelJob[TItem, TResult], items: list[TItem]
    ) -> list[TResult]:
        """
        :param job: The job to run.
        :param items: What the job works through; each worker gets every n-th item.
        :return: The results of every item worked through, worker by worker.
        :raises NotEnoughMemoryForWorkers: If not even one worker fits.
        """
        number_of_workers = self.workers_that_fit()
        context = get_context("spawn")
        with (
            context.Manager() as manager,
            self.stopping_on_signals(stop=manager.Event()) as stop,
        ):
            if number_of_workers == 1:
                return job.work_through(items=items, stop=stop)
            with ProcessPoolExecutor(
                max_workers=number_of_workers,
                mp_context=context,
                initializer=self.ignore_interrupts,
            ) as pool:
                workers = [
                    pool.submit(
                        job.work_through,
                        items=items[worker::number_of_workers],
                        stop=stop,
                    )
                    for worker in range(number_of_workers)
                ]
                return [result for worker in workers for result in worker.result()]

    @staticmethod
    @contextmanager
    def stopping_on_signals(*, stop: EventProxy) -> Iterator[EventProxy]:
        """
        :param stop: Set once this process is interrupted or asked to terminate.
        :return: A context in which the interrupt and the termination signal set
            ``stop`` instead of ending the process; outside the main thread, where no
            signal can be handled, they keep their handling.
        """
        if threading.current_thread() is not threading.main_thread():
            yield stop
            return
        signals = [signal.SIGINT, signal.SIGTERM]
        previous_handlers = {
            handled_signal: signal.getsignal(handled_signal)
            for handled_signal in signals
        }

        def request_stop(signal_number: int, frame: Any) -> None:
            """
            Handles a signal by asking the job to stop.
            """
            stop.set()

        for handled_signal in signals:
            signal.signal(handled_signal, request_stop)
        try:
            yield stop
        finally:
            for handled_signal, handler in previous_handlers.items():
                signal.signal(handled_signal, handler)

    @staticmethod
    def ignore_interrupts() -> None:
        """
        Lets a worker process ignore interrupts, which reach every process of a
        terminal, so that only this process decides when the workers stop.
        """
        signal.signal(signal.SIGINT, signal.SIG_IGN)
