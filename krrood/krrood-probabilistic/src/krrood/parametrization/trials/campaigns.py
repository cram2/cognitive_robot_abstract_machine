"""
Trying trials of a behaviour and keeping them, many at once in worker processes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from multiprocessing.managers import EventProxy

from typing_extensions import Generic

from krrood.parametrization.trials.exceptions import OutcomeDidNotVaryInPilot
from krrood.parametrization.trials.parameter_spaces import ParameterSource
from krrood.parametrization.trials.records import ParametersTypeVar, TrialTypeVar
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood_compute_resources.workers import ParallelJob, WorkerPool

# %% trying and storing


@dataclass(kw_only=True)
class TrialPerformer(
    Generic[ParametersTypeVar, TrialTypeVar], SubClassSafeGeneric, ABC
):
    """
    Tries a behaviour with given parameters, each time from the same starting state.
    """

    @abstractmethod
    def perform(self, *, parameters: ParametersTypeVar) -> TrialTypeVar:
        """
        :param parameters: The parameters to try.
        :return: The trial, with what came of it.
        """


@dataclass(kw_only=True)
class TrialStore(Generic[TrialTypeVar], SubClassSafeGeneric, ABC):
    """
    Keeps trials, each as soon as it is tried.
    """

    def prepare(self) -> None:
        """
        Makes the store ready for workers to keep trials in at once, such as by creating
        its tables; by default there is nothing to make ready.
        """

    @abstractmethod
    def add(self, *, trial: TrialTypeVar) -> None:
        """
        :param trial: The trial to keep.
        """


@dataclass(kw_only=True)
class TrialSetup(Generic[ParametersTypeVar, TrialTypeVar], SubClassSafeGeneric, ABC):
    """
    Everything a worker of a campaign tries trials with; it is sent to each worker's
    process, so it holds only what can be pickled and builds the rest there.
    """

    @abstractmethod
    def performing(
        self,
    ) -> AbstractContextManager[TrialPerformer[ParametersTypeVar, TrialTypeVar]]:
        """
        :return: A context in which the performer can try trials, such as a running
            simulation of the starting state, closed once the worker is done.
        """

    @abstractmethod
    def create_store(self) -> TrialStore[TrialTypeVar]:
        """
        :return: Where the worker keeps its trials.
        """


# %% many trials at once


@dataclass(kw_only=True)
class TrialCampaign(
    ParallelJob[ParametersTypeVar, bool], Generic[ParametersTypeVar, TrialTypeVar]
):
    """
    Tries the parameters of a source, each :attr:`repetitions` times, in several worker
    processes at once, keeping every trial as soon as it is tried.
    """

    setup: TrialSetup[ParametersTypeVar, TrialTypeVar]
    """
    What every worker tries its trials with.
    """

    parameter_source: ParameterSource[ParametersTypeVar]
    """
    Gives the parameters to try.
    """

    repetitions: int = 1
    """
    How often each parameters is tried, so that a success rate is learned for it where
    the same parameters do not always lead to the same outcome.
    """

    pilot_trials: int = 0
    """
    How many of the first trials are tried in this process before any worker starts, to
    make sure their outcome varies; none when ``0``.
    """

    worker_pool: WorkerPool = field(default_factory=WorkerPool)
    """
    The workers that try the trials after the pilot.
    """

    def run(self) -> int:
        """
        :return: How many trials were kept; fewer than given when the campaign was
            stopped.
        :raises NotEnoughMemoryForWorkers: If not even one worker fits into the
            working memory.
        :raises OutcomeDidNotVaryInPilot: If every pilot trial came out the same.
        """
        # refuse before anything is tried or created
        self.worker_pool.workers_that_fit()
        self.setup.create_store().prepare()
        parameters = [
            parameters_to_repeat
            for parameters_to_repeat in self.parameter_source.parameters_to_try()
            for _ in range(self.repetitions)
        ]
        pilot = parameters[: self.pilot_trials]
        if pilot:
            successes = WorkerPool().run(job=self, items=pilot)
            self.raise_unless_the_outcome_varies(successes=successes)
        remaining = parameters[len(pilot) :]
        return len(pilot) + len(self.worker_pool.run(job=self, items=remaining))

    def work_through_items(
        self, *, items: list[ParametersTypeVar], stop: EventProxy
    ) -> list[bool]:
        store = self.setup.create_store()
        successes = []
        with self.setup.performing() as performer:
            for parameters in items:
                if stop.is_set():
                    break
                trial = performer.perform(parameters=parameters)
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
