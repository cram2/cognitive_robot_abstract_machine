"""
Whether and where a sorting run keeps the iterations it finishes.

Recording is best effort: a run whose database will not take a write sorts anyway and
says so, rather than losing a finished sort to a database problem.
:func:`~experiments.montessori.results_database.main` reports the same thing before a
world has been built, where it costs a fraction of a second rather than a minute.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from krrood.ormatic.data_access_objects.helper import to_dao
from sqlalchemy.orm import Session
from typing_extensions import Protocol

from experiments.montessori.results_database import (
    ReadOnlyResultsDatabase,
    ResultsDatabase,
    UnreachableResultsDatabase,
    database_label,
    verify_reachable,
    verify_writable,
)
from experiments.montessori.sorting_results import SortingIterationResult

logger = logging.getLogger(__name__)


class RecordsIterations(Protocol):
    """
    Somewhere a run's finished iterations go.
    """

    def record(self, iteration_result: SortingIterationResult) -> None:
        """
        Keep one finished iteration.
        """

    def close(self) -> None:
        """
        Stop recording, releasing whatever was held open.
        """


@dataclass
class RecordsIterationsToADatabase:
    """
    Keeps every finished iteration in a database, one commit each.

    Committed as each iteration finishes rather than once at the end, so a run
    interrupted halfway keeps everything it had finished by then.
    """

    session: Session
    """
    The open session iterations are committed through.
    """

    def record(self, iteration_result: SortingIterationResult) -> None:
        """
        Commit one finished iteration.

        :param iteration_result: The iteration to keep.
        """
        self.session.add(to_dao(iteration_result))
        self.session.commit()

    def close(self) -> None:
        """
        Close the session iterations were committed through.
        """
        self.session.close()


@dataclass
class RecordsNothing:
    """
    Keeps no iteration at all, for a run that would rather sort than record.
    """

    def record(self, iteration_result: SortingIterationResult) -> None:
        """
        Keep nothing.

        :param iteration_result: The iteration that is not kept.
        """

    def close(self) -> None:
        """
        Close nothing.
        """


def open_recording(results_database: ResultsDatabase) -> RecordsIterations:
    """
    Start keeping finished iterations, or keep none if the database refuses them.

    Records through the database object it is given rather than one of its own, so
    whatever reads a run's results back reads the very rows the run is writing -- which
    for an in-memory database is only true of a shared connection.

    :param results_database: The database to record to.
    :return: A recorder writing to that database, or one keeping nothing when it cannot
        be reached or will not take a write.
    """
    try:
        verify_reachable(results_database.uri)
        verify_writable(results_database.uri)
    except (UnreachableResultsDatabase, ReadOnlyResultsDatabase) as error:
        logger.warning("%s", error)
        logger.warning("Sorting anyway; this run's results are not being recorded.")
        return RecordsNothing()
    logger.info("Recording results to %s.", database_label(results_database.uri))
    return RecordsIterationsToADatabase(session=results_database.open_session())
