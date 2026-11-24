"""
Keeping trials in a database through ormatic, as episodes of the state they started
from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

from sqlalchemy import select
from sqlalchemy.orm import Session

from krrood.ormatic.data_access_objects.dao import DataAccessObject
from krrood.ormatic.utils import create_engine
from krrood.parametrization.trials.campaigns import TrialStore
from krrood.parametrization.trials.records import TrialTypeVar

# %% trials in a database


@dataclass(kw_only=True)
class DatabaseTrialStore(TrialStore[TrialTypeVar], ABC):
    """
    Keeps the trials of one starting state in a database through ormatic, each in a
    transaction of its own, so that workers in other processes can keep theirs at once.
    """

    database_url: str
    """
    Where the database is, as SQLAlchemy names it.
    """

    starting_state_id: int | None = None
    """
    The database id of the state every trial kept here started from; ``None`` for trials
    of a state that is not stored.
    """

    @property
    @abstractmethod
    def data_access_object_class(self) -> type[DataAccessObject]:
        """
        The class ormatic generated for the trials.

        .. note::
            Import the generated interface inside this property, so each worker
            process loads it before converting its first trial.
        """

    @cached_property
    def session(self) -> Session:
        """
        The session the trials are kept and read in, opened in the process that uses it.
        """
        return Session(create_engine(self.database_url))

    def prepare(self) -> None:
        """
        Unlike :meth:`TrialStore.prepare`, creates the tables of the trials, unless they
        exist.
        """
        engine = create_engine(self.database_url)
        self.data_access_object_class.metadata.create_all(engine)
        engine.dispose()

    def add(self, *, trial: TrialTypeVar) -> None:
        """
        Unlike :meth:`TrialStore.add`, the trial is first marked as started from
        :attr:`starting_state_id`.
        """
        trial.starting_state_id = self.starting_state_id
        self.session.add(self.data_access_object_class.to_dao(source_object=trial))
        self.session.commit()

    def stored_trials(self) -> list[TrialTypeVar]:
        """
        :return: Every trial kept from :attr:`starting_state_id`, in the order they were
            kept.
        """
        trial_class = self.data_access_object_class
        query = (
            select(trial_class)
            .where(trial_class.starting_state_id == self.starting_state_id)
            .order_by(trial_class.database_id)
        )
        return [stored.from_dao() for stored in self.session.scalars(query)]
