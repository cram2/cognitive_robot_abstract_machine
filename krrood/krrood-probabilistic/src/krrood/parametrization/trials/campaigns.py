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

from typing_extensions import Generic

from krrood.parametrization.trials.exceptions import (
    NotEnoughMemoryForWorkers,
    OutcomeDidNotVaryInPilot,
)
from krrood.parametrization.trials.parameter_spaces import ParameterSource
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
    Tries the parameters of a source, each as often as asked, in several worker
    processes at once, keeping every trial as soon as it is tried.
    """

    setup: TrialSetup[TParameters, TTrial]
    """
    What every worker tries its trials with.
    """

    parameter_source: ParameterSource[TParameters]
    """
    Gives the parameters to try.
    """

    pilot_trials: int = 0
    """
    How many of the first trials are tried in this process before any worker starts, to
    make sure their outcome varies; none when ``0``.
    """

    number_of_workers: int = 1
    """
    How many processes try trials at once; a single worker tries them in this process.
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
        :raises OutcomeDidNotVaryInPilot: If every pilot trial came out the same.
        """
        self.raise_unless_the_workers_fit_into_memory()
        self.setup.create_store().prepare()
        parameters = self.parameter_source.parameters_to_try()
        pilot = parameters[: self.pilot_trials]
        if pilot:
            self.raise_unless_the_outcome_varies(
                successes=self.try_trials(parameters=pilot)
            )
        remaining = parameters[len(pilot) :]
        if self.number_of_workers == 1:
            return len(pilot) + len(self.try_trials(parameters=remaining))
        shares = [
            remaining[worker :: self.number_of_workers]
            for worker in range(self.number_of_workers)
        ]
        with ProcessPoolExecutor(
            max_workers=self.number_of_workers, mp_context=get_context("spawn")
        ) as pool:
            workers = [
                pool.submit(self.try_trials, parameters=share) for share in shares
            ]
            return len(pilot) + sum(len(worker.result()) for worker in workers)

    def try_trials(self, *, parameters: list[TParameters]) -> list[bool]:
        """
        :param parameters: The parameters to try, in order.
        :return: Whether each trial succeeded, in the order they were tried.
        """
        store = self.setup.create_store()
        successes = []
        with self.setup.performing() as performer:
            for trial_parameters in parameters:
                trial = performer.perform(parameters=trial_parameters)
                store.add(trial=trial)
                successes.append(trial.succeeded)
        return successes

    @staticmethod
    def raise_unless_the_outcome_varies(*, successes: list[bool]) -> None:
        """
        :param successes: Whether each pilot trial succeeded.
        :raises OutcomeDidNotVaryInPilot: If every pilot trial came out the same.
        """
        if len(set(successes)) > 1:
            return
        raise OutcomeDidNotVaryInPilot(
            number_of_trials=len(successes), succeeded=successes[0]
        )

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
