"""
Tests for whether and where a Montessori run keeps the iterations it finishes.
"""

from __future__ import annotations

import logging

import pytest
from sqlalchemy import func, select

from experiments.montessori.results_database import (
    IN_MEMORY_DATABASE_URI,
    ResultsDatabase,
)
from experiments.montessori.results_recording import (
    RecordsIterationsToADatabase,
    RecordsNothing,
    open_recording,
)
from experiments.montessori.sorting_results import SortingIterationResult

UNREACHABLE_URI = (
    "postgresql+psycopg://recorder:hunter2@127.0.0.1:1/montessori_sorting_results"
)
"""
A Postgres URI on a port nothing listens on, carrying a password a log must not repeat.
"""


def recorded_iteration_count(results_database: ResultsDatabase) -> int:
    """
    How many iterations a database holds.

    :param results_database: The database to count in.
    """
    import experiments.orm.ormatic_interface as ormatic_interface

    with results_database.open_session() as session:
        return session.scalar(
            select(func.count()).select_from(
                ormatic_interface.SortingIterationResultDAO
            )
        )


# %% recording to a database that takes writes
def test_a_finished_iteration_is_kept(tmp_path):
    database = ResultsDatabase(uri="sqlite:///%s" % (tmp_path / "results.db"))

    recording = open_recording(database)
    recording.record(SortingIterationResult(iteration=1))
    recording.close()

    assert recorded_iteration_count(database) == 1


def test_a_writable_database_is_recorded_to(tmp_path):
    recording = open_recording(
        ResultsDatabase(uri="sqlite:///%s" % (tmp_path / "results.db"))
    )

    assert isinstance(recording, RecordsIterationsToADatabase)
    recording.close()


def test_an_in_memory_database_is_read_back_through_the_same_object():
    """
    An in-memory database is what a run falls back to, and the viewer's episodic-memory
    questions are answered from the very object the run recorded through.
    """
    database = ResultsDatabase(uri=IN_MEMORY_DATABASE_URI)

    recording = open_recording(database)
    recording.record(SortingIterationResult(iteration=1))

    assert recorded_iteration_count(database) == 1
    recording.close()


# %% a database that will not take them
def test_a_run_that_cannot_reach_a_database_records_nothing(caplog):
    """
    A database problem must cost a run its results, never the sort itself.
    """
    with caplog.at_level(logging.WARNING):
        recording = open_recording(ResultsDatabase(uri=UNREACHABLE_URI))

    assert isinstance(recording, RecordsNothing)


def test_a_run_that_cannot_record_says_so_without_the_password(caplog):
    """
    The URI usually comes from an environment variable that carries a password, and a
    demo's log is pasted into issues and chats.
    """
    with caplog.at_level(logging.WARNING):
        open_recording(ResultsDatabase(uri=UNREACHABLE_URI))

    assert "montessori_sorting_results" in caplog.text
    assert "hunter2" not in caplog.text


def test_a_run_that_cannot_record_is_told_it_is_not_recording(caplog):
    """
    Nothing else in the run's output would reveal that its results are being dropped.
    """
    with caplog.at_level(logging.WARNING):
        open_recording(ResultsDatabase(uri=UNREACHABLE_URI))

    assert "not being recorded" in caplog.text


def test_a_read_only_database_records_nothing(tmp_path):
    path = tmp_path / "results.db"
    ResultsDatabase(uri="sqlite:///%s" % path).open_session().close()

    recording = open_recording(
        ResultsDatabase(uri="sqlite:///file:%s?mode=ro&uri=true" % path)
    )

    assert isinstance(recording, RecordsNothing)


# %% recording nothing at all
def test_recording_nothing_keeps_nothing():
    recording = RecordsNothing()

    recording.record(SortingIterationResult(iteration=1))
    recording.close()
