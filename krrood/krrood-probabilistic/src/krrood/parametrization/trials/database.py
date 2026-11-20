"""
Keeping trials and the states they start from in a database through ormatic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

from sqlalchemy import select
from sqlalchemy.orm import Session
from typing_extensions import Any

from krrood.ormatic.data_access_objects.dao import DataAccessObject
from krrood.ormatic.utils import create_engine
from krrood.parametrization.trials.campaigns import TrialStore
from krrood.parametrization.trials.records import TTrial

# %% objects of one class in a database


@dataclass(kw_only=True)
class ObjectDatabase(ABC):
    """
    A database that keeps objects of one class through ormatic, each in a transaction of
    its own, so that processes can keep theirs at once.
    """

    database_url: str
    """
    Where the database is, as SQLAlchemy names it.
    """

    @property
    @abstractmethod
    def data_access_object_class(self) -> type[DataAccessObject]:
        """
        The class ormatic generated for the objects.

        .. note::
            Import the generated interface inside this property, so each worker
            process loads it before converting its first object.
        """

    def create_tables(self) -> None:
        """
        Creates the tables of the objects, unless they exist.
        """
        engine = create_engine(self.database_url)
        self.data_access_object_class.metadata.create_all(engine)
        engine.dispose()

    def open_session(self) -> Session:
        """
        :return: A new session of the database, for the process that opens it.
        """
        return Session(create_engine(self.database_url))

    def store(self, *, stored_object: Any, session: Session) -> int:
        """
        :param stored_object: The object to keep.
        :param session: The session to keep it in.
        :return: The database id the object is kept under.
        """
        data_access_object = self.data_access_object_class.to_dao(
            source_object=stored_object
        )
        session.add(data_access_object)
        session.commit()
        return data_access_object.database_id

    def stored_object(self, *, database_id: int) -> Any:
        """
        :param database_id: The database id an object is kept under.
        :return: The object, read back from the database.
        """
        with self.open_session() as session:
            return session.get(self.data_access_object_class, database_id).from_dao()


# %% trials in a database


@dataclass(kw_only=True)
class DatabaseTrialStore(TrialStore[TTrial], ObjectDatabase, ABC):
    """
    Keeps the trials of one starting state in a database, as episodes of that state.
    """

    starting_state_id: int | None = None
    """
    The database id of the state every trial kept here started from; ``None`` for trials
    of a state that is not stored.
    """

    @cached_property
    def session(self) -> Session:
        """
        The session the trials are kept and read in.
        """
        return self.open_session()

    def prepare(self) -> None:
        """
        Unlike :meth:`TrialStore.prepare`, creates the tables of the trials.
        """
        self.create_tables()

    def add(self, *, trial: TTrial) -> None:
        """
        Unlike :meth:`TrialStore.add`, the trial is first marked as started from
        :attr:`starting_state_id`.
        """
        trial.starting_state_id = self.starting_state_id
        self.store(stored_object=trial, session=self.session)

    def stored_trials(self) -> list[TTrial]:
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
