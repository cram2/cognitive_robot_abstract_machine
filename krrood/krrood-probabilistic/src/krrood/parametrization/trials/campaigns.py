"""
Trying trials of a behaviour and keeping them, many at once in worker processes.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import dataclass
from multiprocessing import get_context

import numpy as np
from typing_extensions import Generic

from krrood.parametrization.model_registries import ModelRegistry
from krrood.parametrization.trials.exceptions import NotEnoughMemoryForWorkers
from krrood.parametrization.trials.parameter_spaces import ParameterSpace
from krrood.parametrization.trials.records import TParameters, TTrial
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric

# %% trying and storing


@dataclass(kw_only=True)
class TrialPerformer(Generic[TParameters, TTrial], SubClassSafeGeneric, ABC):
    """
    Tries a behaviour with given parameters, each time from the same starting state.
    """

    @abstractmethod
    def perform(self, *, parameters: TParameters) -> TTrial:
        """
        :param parameters: The parameters to try.
        :return: The trial, with what came of it.
        """


@dataclass(kw_only=True)
class TrialStore(Generic[TTrial], SubClassSafeGeneric, ABC):
    """
    Keeps trials, each as soon as it is tried.
    """

    def prepare(self) -> None:
        """
        Makes the store ready for workers to keep trials in at once, such as by creating
        its tables; by default there is nothing to make ready.
        """

    @abstractmethod
    def add(self, *, trial: TTrial) -> None:
        """
        :param trial: The trial to keep.
        """


@dataclass(kw_only=True)
class TrialSetup(Generic[TParameters, TTrial], SubClassSafeGeneric, ABC):
    """
    Everything a worker of a campaign tries trials with; it is sent to each worker's
    process, so it holds only what can be pickled and builds the rest there.
    """

    parameter_space: ParameterSpace[TParameters]
    """
    What the parameters are drawn from.
    """

    model_registry: ModelRegistry
    """
    Answers the statement of :attr:`parameter_space`.
    """

    @abstractmethod
    def performing(self) -> AbstractContextManager[TrialPerformer[TParameters, TTrial]]:
        """
        :return: A context in which the performer can try trials, such as a running
            simulation of the starting state, closed once the worker is done.
        """

    @abstractmethod
    def create_store(self) -> TrialStore[TTrial]:
        """
        :return: Where the worker keeps its trials.
        """


# %% many trials at once


@dataclass(kw_only=True)
class TrialCampaign(Generic[TParameters, TTrial], SubClassSafeGeneric):
    """
    Tries parameters drawn by a setup's model in several worker processes at once, each
    drawing its own with a seed of its own and keeping every trial as soon as it is
    tried.
    """

    setup: TrialSetup[TParameters, TTrial]
    """
    What every worker tries its trials with.
    """

    number_of_workers: int = 1
    """
    How many processes try trials at once; a single worker tries them in this process.
    """

    trials_per_worker: int = 10
    """
    How many parameters each worker draws.
    """

    first_draw_seed: int = 0
    """
    The seed the first worker draws with; each further worker's seed is one more.
    """

    memory_per_worker: int | None = None
    """
    How much working memory a worker needs at most, in bytes; ``None`` when it is not
    checked.
    """

    def run(self) -> int:
        """
        :return: How many trials were kept.
        :raises NotEnoughMemoryForWorkers: If the workers would need more working
            memory than is free.
        """
        self.raise_unless_the_workers_fit_into_memory()
        self.setup.create_store().prepare()
        seeds = [
            self.first_draw_seed + worker for worker in range(self.number_of_workers)
        ]
        if self.number_of_workers == 1:
            return self.run_worker(draw_seed=seeds[0])
        with ProcessPoolExecutor(
            max_workers=self.number_of_workers, mp_context=get_context("spawn")
        ) as pool:
            workers = [pool.submit(self.run_worker, draw_seed=seed) for seed in seeds]
            return sum(worker.result() for worker in workers)

    def run_worker(self, *, draw_seed: int) -> int:
        """
        :param draw_seed: Seeds the worker's draws.
        :return: How many trials the worker kept.
        """
        # the probabilistic models draw from numpy's global random state
        np.random.seed(draw_seed)
        drawn = self.setup.parameter_space.drawn_parameters(
            number_of_draws=self.trials_per_worker,
            model_registry=self.setup.model_registry,
        )
        store = self.setup.create_store()
        with self.setup.performing() as performer:
            for parameters in drawn:
                store.add(trial=performer.perform(parameters=parameters))
        return len(drawn)

    def raise_unless_the_workers_fit_into_memory(self) -> None:
        """
        :raises NotEnoughMemoryForWorkers: If :attr:`memory_per_worker` is given and the
            workers would need more working memory than is free.
        """
        if self.memory_per_worker is None:
            return
        free_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
        if self.number_of_workers * self.memory_per_worker > free_memory:
            raise NotEnoughMemoryForWorkers(
                number_of_workers=self.number_of_workers,
                memory_per_worker=self.memory_per_worker,
                free_memory=free_memory,
            )
