"""
Reaching the state a behaviour's trials start from, checking it is the state wanted, and
keeping every state that is.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass
from multiprocessing.managers import EventProxy

from typing_extensions import Generic, TypeVar

from krrood.parametrization.trials.database import ObjectDatabase
from krrood.parametrization.trials.workers import ParallelJob, WorkerPool
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric

TState = TypeVar("TState")
"""
A state trials start from, such as a world.
"""

# %% reaching a state


@dataclass(kw_only=True)
class StateReacher(Generic[TState], SubClassSafeGeneric, ABC):
    """
    Reaches a state by the steps that lead to it, each time anew.
    """

    @abstractmethod
    def reach(self) -> TState:
        """
        :return: The state the steps led to, which need not be the one wanted, since
            reaching it can fail and physics varies.
        """


# %% the recipe of a starting state


@dataclass(kw_only=True)
class StartingStateRecipe(ParallelJob[int, int], ObjectDatabase, Generic[TState], ABC):
    """
    How to reach the state the trials of a behaviour start from and how to tell it is
    the one wanted; attempts to reach it run in several workers at once, and every state
    that is the one wanted is kept.
    """

    @abstractmethod
    def reaching(self) -> AbstractContextManager[StateReacher[TState]]:
        """
        :return: A context in which the reacher can reach states, such as a running
            simulation, closed once the worker is done.
        """

    @abstractmethod
    def is_wanted(self, *, state: TState) -> bool:
        """
        :param state: A state the steps led to.
        :return: Whether it is the state the trials are to start from, such as an
            object being held and touching its target.
        """

    def reach_and_store(self, *, attempts: int, worker_pool: WorkerPool) -> list[int]:
        """
        :param attempts: How often to try to reach the state.
        :param worker_pool: The workers the attempts are divided among.
        :return: The database id of every state reached that is the one wanted; fewer
            than attempted when the campaign was stopped.
        :raises NotEnoughMemoryForWorkers: If not even one worker fits into the working
            memory.
        """
        worker_pool.workers_that_fit()
        self.create_tables()
        return worker_pool.run(job=self, items=list(range(attempts)))

    def work_through(self, *, items: list[int], stop: EventProxy) -> list[int]:
        stored_ids = []
        with self.open_session() as session, self.reaching() as reacher:
            for _ in items:
                if stop.is_set():
                    break
                state = reacher.reach()
                if not self.is_wanted(state=state):
                    continue
                stored_ids.append(self.store(stored_object=state, session=session))
        return stored_ids
