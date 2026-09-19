"""
Tests for resolving and reaching the database a Montessori run records to.
"""

from __future__ import annotations

import threading

import pytest
from sqlalchemy import inspect

from experiments.montessori.results_database import (
    ConfiguredDatabase,
    DATABASE_URI_ENVIRONMENT_VARIABLE,
    DEFAULT_DATABASE_URI,
    DatabaseUriOrigin,
    IN_MEMORY_DATABASE_URI,
    ReadOnlyResultsDatabase,
    ResultsDatabase,
    UnreachableResultsDatabase,
    configured_database_uri,
    create_results_engine,
    main,
    verify_reachable,
    verify_writable,
)

UNREACHABLE_URI = (
    "postgresql+psycopg://nobody:wrong@127.0.0.1:1/montessori_sorting_results"
)
"""
A Postgres URI on a port nothing listens on, so reaching it fails without waiting.
"""


# %% which database a run records to
class TestResolvingTheUri:
    def test_the_default_is_used_when_nothing_overrides_it(self, monkeypatch):
        monkeypatch.delenv(DATABASE_URI_ENVIRONMENT_VARIABLE, raising=False)

        assert configured_database_uri() == DEFAULT_DATABASE_URI

    def test_the_environment_overrides_the_default(self, monkeypatch):
        monkeypatch.setenv(DATABASE_URI_ENVIRONMENT_VARIABLE, "sqlite://")

        assert configured_database_uri() == "sqlite://"


# %% opening it
class TestOpeningSessions:
    """
    A run writes its results and the viewer reads them back, so both open the same
    database the same way.
    """

    def test_a_session_is_opened_against_the_configured_database(self, tmp_path):
        database = ResultsDatabase(uri="sqlite:///%s" % (tmp_path / "results.db"))

        with database.open_session() as session:
            assert session.bind.url.database == str(tmp_path / "results.db")

    def test_the_schema_is_created_before_the_first_write(self, tmp_path):
        """
        A fresh database has no tables of its own, and a run must not have to be told to
        create them first.
        """
        database = ResultsDatabase(uri="sqlite:///%s" % (tmp_path / "results.db"))

        with database.open_session() as session:
            tables = inspect(session.bind).get_table_names()

        assert "ShapeInsertionResultDAO" in tables

    def test_the_default_database_is_the_configured_one(self, monkeypatch):
        monkeypatch.setenv(DATABASE_URI_ENVIRONMENT_VARIABLE, "sqlite://")

        assert ResultsDatabase().uri == "sqlite://"


# %% is it actually reachable
class TestReachingTheDatabase:
    def test_a_reachable_database_verifies_quietly(self):
        assert verify_reachable("sqlite://") is None

    def test_an_unreachable_database_raises(self):
        with pytest.raises(UnreachableResultsDatabase):
            verify_reachable(UNREACHABLE_URI)

    def test_the_failure_names_the_database_without_its_password(self):
        """
        The message is printed to a terminal, so it says which database could not be
        reached without putting that database's password there too.
        """
        with pytest.raises(UnreachableResultsDatabase) as raised:
            verify_reachable(UNREACHABLE_URI)

        message = str(raised.value)
        assert "montessori_sorting_results" in message
        assert "wrong" not in message

    def test_the_failure_says_what_to_do_about_it(self):
        with pytest.raises(UnreachableResultsDatabase) as raised:
            verify_reachable(UNREACHABLE_URI)

        assert "--database-uri" in str(raised.value)


# %% can it actually be recorded to
@pytest.fixture
def read_only_database(tmp_path) -> str:
    """
    A database that exists and accepts connections but refuses every write.

    What a role provisioned for reading only looks like from the outside, which is the
    misconfiguration a run cannot notice until its first insert.
    """
    path = tmp_path / "results.db"
    ResultsDatabase(uri="sqlite:///%s" % path).open_session().close()
    return "sqlite:///file:%s?mode=ro&uri=true" % path


class TestWritingToTheDatabase:
    def test_a_writable_database_verifies_quietly(self, tmp_path):
        assert verify_writable("sqlite:///%s" % (tmp_path / "results.db")) is None

    def test_a_read_only_database_raises(self, read_only_database):
        """
        Connecting says nothing about recording: this database is reachable, and every
        table a run writes to is already there.
        """
        assert verify_reachable(read_only_database) is None

        with pytest.raises(ReadOnlyResultsDatabase):
            verify_writable(read_only_database)

    def test_the_failure_names_the_database_without_its_password(self):
        with pytest.raises(ReadOnlyResultsDatabase) as raised:
            verify_writable(
                "postgresql+psycopg://reader:secret@127.0.0.1:1/"
                "montessori_sorting_results"
            )

        message = str(raised.value)
        assert "montessori_sorting_results" in message
        assert "secret" not in message

    def test_the_failure_points_at_the_environment_variable(self, read_only_database):
        """
        A URI that came from the environment appears nowhere in the command that was
        run, so a failure has to say where else to look for it.
        """
        with pytest.raises(ReadOnlyResultsDatabase) as raised:
            verify_writable(read_only_database)

        assert DATABASE_URI_ENVIRONMENT_VARIABLE in str(raised.value)

    def test_verifying_leaves_nothing_behind(self, tmp_path):
        """
        The check runs before every run, so it must not accumulate anything in the
        database it is checking.
        """
        uri = "sqlite:///%s" % (tmp_path / "results.db")
        tables_before = inspect(create_results_engine(uri)).get_table_names()

        verify_writable(uri)

        assert inspect(create_results_engine(uri)).get_table_names() == tables_before


# %% where the database came from
class TestResolvingWithItsOrigin:
    def test_a_uri_given_on_the_command_line_says_so(self, monkeypatch):
        monkeypatch.setenv(DATABASE_URI_ENVIRONMENT_VARIABLE, "sqlite://")

        resolved = ConfiguredDatabase.resolve("sqlite:///given.db")

        assert resolved.uri == "sqlite:///given.db"
        assert resolved.origin is DatabaseUriOrigin.COMMAND_LINE

    def test_a_uri_from_the_environment_says_so(self, monkeypatch):
        monkeypatch.setenv(DATABASE_URI_ENVIRONMENT_VARIABLE, "sqlite:///from_env.db")

        resolved = ConfiguredDatabase.resolve(None)

        assert resolved.uri == "sqlite:///from_env.db"
        assert resolved.origin is DatabaseUriOrigin.ENVIRONMENT

    def test_the_built_in_default_says_so(self, monkeypatch):
        monkeypatch.delenv(DATABASE_URI_ENVIRONMENT_VARIABLE, raising=False)

        resolved = ConfiguredDatabase.resolve(None)

        assert resolved.uri == DEFAULT_DATABASE_URI
        assert resolved.origin is DatabaseUriOrigin.BUILT_IN_DEFAULT


# %% falling back to memory when the configured database is not there
class TestFallingBackToMemory:
    """
    A database that is simply not running must cost a run its recorded history, never
    the sort it was started for.
    """

    def test_a_reachable_database_is_kept(self, tmp_path):
        uri = "sqlite:///%s" % (tmp_path / "results.db")

        resolved = ConfiguredDatabase.resolve_reachable(uri)

        assert resolved.uri == uri
        assert resolved.fell_back_from is None

    def test_an_unreachable_database_is_replaced_by_one_in_memory(self):
        resolved = ConfiguredDatabase.resolve_reachable(UNREACHABLE_URI)

        assert resolved.uri == IN_MEMORY_DATABASE_URI
        assert resolved.origin is DatabaseUriOrigin.IN_MEMORY_FALLBACK

    def test_the_fallback_remembers_which_database_it_stands_in_for(self):
        """
        The run still has to be able to say which database was missing, and why.
        """
        resolved = ConfiguredDatabase.resolve_reachable(UNREACHABLE_URI)

        assert resolved.fell_back_from.database_uri == UNREACHABLE_URI

    def test_the_fallback_says_it_keeps_nothing_past_the_run(self):
        described = ConfiguredDatabase.resolve_reachable(UNREACHABLE_URI).describe()

        assert "only until this run exits" in described

    def test_an_unreachable_database_from_the_environment_is_replaced_too(
        self, monkeypatch
    ):
        """
        The variable is usually set in a shell profile, so its database being down must
        not be a reason for every run on that host to refuse to start.
        """
        monkeypatch.setenv(DATABASE_URI_ENVIRONMENT_VARIABLE, UNREACHABLE_URI)

        assert ConfiguredDatabase.resolve_reachable(None).uri == IN_MEMORY_DATABASE_URI


# %% a database that lives only in this process
class TestTheInMemoryDatabase:
    def test_the_schema_is_created_in_it_too(self):
        database = ResultsDatabase(uri=IN_MEMORY_DATABASE_URI)

        with database.open_session() as session:
            assert "ShapeInsertionResultDAO" in inspect(session.bind).get_table_names()

    def test_another_thread_opens_the_same_database(self):
        """
        An in-memory SQLite database lives inside the connection that created it, so the
        run recording on the planning thread and the viewer reading on another are
        looking at the same rows only when one connection is shared between them.
        """
        database = ResultsDatabase(uri=IN_MEMORY_DATABASE_URI)
        database.open_session().close()
        tables_seen_elsewhere = []

        def read_the_schema() -> None:
            with database.open_session() as session:
                tables_seen_elsewhere.extend(inspect(session.bind).get_table_names())

        reader = threading.Thread(target=read_the_schema)
        reader.start()
        reader.join()

        assert "ShapeInsertionResultDAO" in tables_seen_elsewhere


# %% the pre-flight a launcher runs
class TestPreflight:
    def test_it_reports_success_for_a_reachable_database(self, capsys, tmp_path):
        assert main(["--database-uri", "sqlite:///%s" % (tmp_path / "r.db")]) == 0
        assert "sqlite" in capsys.readouterr().out

    def test_an_unreachable_database_does_not_stop_the_run(self, capsys):
        assert main(["--database-uri", UNREACHABLE_URI]) == 0
        assert "Cannot reach the results database" in capsys.readouterr().err

    def test_it_records_in_memory_when_the_database_is_unreachable(self, capsys):
        main(["--database-uri", UNREACHABLE_URI])

        assert IN_MEMORY_DATABASE_URI in capsys.readouterr().out

    def test_a_database_it_cannot_record_to_does_not_stop_the_run(
        self, capsys, read_only_database
    ):
        """
        Naming a database is not the same as demanding to write to it: the live query
        panel reads recorded runs from the same place, which a read-only role serves
        perfectly well.
        """
        assert main(["--database-uri", read_only_database]) == 0

        assert "will not be recorded" in capsys.readouterr().err

    def test_it_says_a_database_it_cannot_record_to_is_read_only(
        self, capsys, read_only_database
    ):
        """
        The pre-flight is where a run learns this, in a fraction of a second rather
        than after a world has been built.
        """
        main(["--database-uri", read_only_database])

        assert "Cannot record results to" in capsys.readouterr().err

    def test_it_ignores_arguments_meant_for_the_demo(self, tmp_path):
        """
        A launcher forwards the run's whole argument list, most of which is the demo's.
        """
        uri = "sqlite:///%s" % (tmp_path / "r.db")

        assert main(["--viewer", "--no-rviz", "--database-uri", uri]) == 0

    def test_it_falls_back_to_the_configured_database(self, monkeypatch, tmp_path):
        monkeypatch.setenv(
            DATABASE_URI_ENVIRONMENT_VARIABLE, "sqlite:///%s" % (tmp_path / "r.db")
        )

        assert main([]) == 0

    def test_a_run_that_records_nothing_needs_no_database(self, monkeypatch, capsys):
        monkeypatch.setenv(DATABASE_URI_ENVIRONMENT_VARIABLE, UNREACHABLE_URI)

        assert main(["--no-record"]) == 0

        assert "Not recording" in capsys.readouterr().out

    def test_it_says_the_environment_chose_the_database(
        self, monkeypatch, capsys, tmp_path
    ):
        """
        A URI set in a shell profile is invisible in the command that was run, so a run
        recording somewhere unexpected has to be traceable from what it printed.
        """
        monkeypatch.setenv(
            DATABASE_URI_ENVIRONMENT_VARIABLE, "sqlite:///%s" % (tmp_path / "r.db")
        )

        main([])

        assert DATABASE_URI_ENVIRONMENT_VARIABLE in capsys.readouterr().out
