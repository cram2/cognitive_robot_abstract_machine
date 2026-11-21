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

from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood_compute_resources.workers import ParallelJob, WorkerPool

StateTypeVar = TypeVar("StateTypeVar")
"""
A state trials start from, such as a world.
"""

# %% reaching and keeping states


@dataclass(kw_only=True)
class StateReacher(Generic[StateTypeVar], SubClassSafeGeneric, ABC):
    """
    Reaches a state by the steps that lead to it, each time anew.
    """

    @abstractmethod
    def reach(self) -> StateTypeVar:
        """
        :return: The state the steps led to, which need not be the one wanted, since
            reaching it can fail and physics varies.
        """


@dataclass(kw_only=True)
class StateStore(Generic[StateTypeVar], SubClassSafeGeneric, ABC):
    """
    Keeps the states trials start from, each as soon as it is reached, under an id the
    trials refer to it by.
    """

    def prepare(self) -> None:
        """
        Makes the store ready for workers to keep states in at once, such as by creating
        its tables; by default there is nothing to make ready.
        """

    @abstractmethod
    def add(self, *, state: StateTypeVar) -> int:
        """
        :param state: The state to keep.
        :return: The id it is kept under, such as its database id.
        """


# %% the recipe of a starting state


@dataclass(kw_only=True)
class StartingStateRecipe(ParallelJob[int, int], Generic[StateTypeVar], ABC):
    """
    How to reach the state the trials of a behaviour start from and how to tell it is
    the one wanted; attempts to reach it run in several workers at once, and every state
    that is the one wanted is kept.
    """

    @abstractmethod
    def reaching(self) -> AbstractContextManager[StateReacher[StateTypeVar]]:
        """
        :return: A context in which the reacher can reach states, such as a running
            simulation, closed once the worker is done.
        """

    @abstractmethod
    def is_wanted(self, *, state: StateTypeVar) -> bool:
        """
        :param state: A state the steps led to.
        :return: Whether it is the state the trials are to start from, such as an
            object being held and touching its target.
        """

    @abstractmethod
    def create_store(self) -> StateStore[StateTypeVar]:
        """
        :return: Where a worker keeps the states that are the one wanted.
        """

    def reach_and_store(self, *, attempts: int, worker_pool: WorkerPool) -> list[int]:
        """
        :param attempts: How often to try to reach the state.
        :param worker_pool: The workers the attempts are divided among.
        :return: The id of every state reached that is the one wanted; fewer than
            attempted when the attempts were stopped.
        :raises NotEnoughMemoryForWorkers: If not even one worker fits into the memory.
        """
        # refuse before anything is created
        worker_pool.workers_that_fit()
        self.create_store().prepare()
        return worker_pool.run(job=self, items=list(range(attempts)))

    def work_through_items(self, *, items: list[int], stop: EventProxy) -> list[int]:
        store = self.create_store()
        stored_ids = []
        with self.reaching() as reacher:
            for _ in items:
                if stop.is_set():
                    break
                state = reacher.reach()
                if self.is_wanted(state=state):
                    stored_ids.append(store.add(state=state))
        return stored_ids
