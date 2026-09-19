"""
Where a sorting run records its results, and whether that is reachable.

Kept apart from the run itself so a launcher can check the database before paying for
the CRAM stack's import and a whole world build: reaching it costs a fraction of a
second, and finding out afterwards costs a minute and buries the reason under a hundred-
line traceback.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from enum import Enum

from krrood.exceptions import DataclassException
from krrood.ormatic.utils import create_engine
from sqlalchemy import Column, Engine, Integer, MetaData, Table
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.sql.sqltypes import NullType
from typing_extensions import List, Optional

# %% which database a run records to

PROVISIONING_SCRIPT = (
    "semantic_digital_twin/scripts/create_postgres_database_and_user_if_not_exists.sql"
)
"""
Script that creates the role and the database a run records to, with the grants it
needs.
"""

DEFAULT_DATABASE_URI = (
    "postgresql+psycopg://semantic_digital_twin:montessori@localhost:5432/"
    "montessori_sorting_results"
)
"""
Database URI used when neither ``--database-uri`` nor
:data:`DATABASE_URI_ENVIRONMENT_VARIABLE` is given.

Reuses the ``semantic_digital_twin`` role already provisioned on this host for the other
experiments in this workspace; only the database itself, ``montessori_sorting_results``,
is dedicated to the shape-sorting runs. Names the ``psycopg`` (v3) driver explicitly
since only that, not ``psycopg2``, is installed in this environment.

Provision the role and this database once, before the first run, with
:data:`PROVISIONING_SCRIPT` (see that script's own header for its ``psql`` invocation).
"""

DATABASE_URI_ENVIRONMENT_VARIABLE = "MONTESSORI_SORTING_DATABASE_URI"
"""
Environment variable overriding :data:`DEFAULT_DATABASE_URI` for every run.
"""

IN_MEMORY_DATABASE_URI = "sqlite://"
"""
Database a run records to when the configured one cannot be reached.

Lives inside the running process, so a run keeps its results only until it exits, and
anything querying them in the meantime is answered from it.
"""

WRITE_PROBE_TABLE_NAME = "montessori_write_probe"
"""
Table :func:`verify_writable` creates and drops again to prove a run can record.
"""


def configured_database_uri() -> str:
    """
    The database a run records to when it is not given one on the command line.
    """
    return os.getenv(DATABASE_URI_ENVIRONMENT_VARIABLE, DEFAULT_DATABASE_URI)


class DatabaseUriOrigin(Enum):
    """
    What decided the database a run records to.
    """

    COMMAND_LINE = "--database-uri"
    ENVIRONMENT = DATABASE_URI_ENVIRONMENT_VARIABLE
    BUILT_IN_DEFAULT = "the built-in default"
    IN_MEMORY_FALLBACK = "an in-memory fallback"


@dataclass(frozen=True)
class ConfiguredDatabase:
    """
    The database a run records to, together with what decided it.
    """

    uri: str
    """
    Where the results go.
    """

    origin: DatabaseUriOrigin
    """
    What settled on :attr:`uri`.
    """

    fell_back_from: Optional[UnreachableResultsDatabase] = None
    """
    The unreachable database this one stands in for, or None when it is the configured
    one.
    """

    @classmethod
    def resolve(cls, requested_uri: Optional[str]) -> ConfiguredDatabase:
        """
        Settle on a database the way a run does.

        :param requested_uri: The URI given on the command line, or None.
        """
        if requested_uri is not None:
            return cls(uri=requested_uri, origin=DatabaseUriOrigin.COMMAND_LINE)
        from_environment = os.getenv(DATABASE_URI_ENVIRONMENT_VARIABLE)
        if from_environment is not None:
            return cls(uri=from_environment, origin=DatabaseUriOrigin.ENVIRONMENT)
        return cls(uri=DEFAULT_DATABASE_URI, origin=DatabaseUriOrigin.BUILT_IN_DEFAULT)

    @classmethod
    def resolve_reachable(cls, requested_uri: Optional[str]) -> ConfiguredDatabase:
        """
        Settle on a database a run can actually open.

        Stands :data:`IN_MEMORY_DATABASE_URI` in for a configured database that cannot
        be reached, so a database nobody started costs a run its recorded history rather
        than the sort it was started for.

        :param requested_uri: The URI given on the command line, or None.
        """
        configured = cls.resolve(requested_uri)
        try:
            verify_reachable(configured.uri)
        except UnreachableResultsDatabase as unreachable:
            return cls(
                uri=IN_MEMORY_DATABASE_URI,
                origin=DatabaseUriOrigin.IN_MEMORY_FALLBACK,
                fell_back_from=unreachable,
            )
        return configured

    def describe(self) -> str:
        """
        This database as it can be shown to someone, saying where it came from.
        """
        if self.fell_back_from is None:
            return "Recording results to %s, from %s" % (
                database_label(self.uri),
                self.origin.value,
            )
        return "Recording results to %s, from %s, kept only until this run exits" % (
            database_label(self.uri),
            self.origin.value,
        )


def database_label(database_uri: str) -> str:
    """
    A database URI as it can be shown to someone, with its password withheld.

    :param database_uri: The URI to render.
    """
    return make_url(database_uri).render_as_string(hide_password=True)


# %% opening it


def is_in_memory(database_uri: str) -> bool:
    """
    Whether a URI names a SQLite database that exists only inside this process.

    :param database_uri: The URI to read.
    """
    url = make_url(database_uri)
    return url.get_backend_name() == "sqlite" and url.database in (None, "", ":memory:")


def create_results_engine(database_uri: str) -> Engine:
    """
    An engine on a results database.

    An in-memory database lives inside the connection that created it, so one connection
    is shared across every thread for it: a run records on the planning thread while
    another thread reads, and they must see the same rows.

    :param database_uri: The database to open.
    """
    if not is_in_memory(database_uri):
        return create_engine(database_uri)
    return create_engine(
        database_uri,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )


@dataclass
class ResultsDatabase:
    """
    The database a Montessori run records its results to, and reads them back from.
    """

    uri: str = field(default_factory=configured_database_uri)
    """
    Where the results live.
    """

    _sessions: Optional[sessionmaker] = field(default=None, init=False, repr=False)
    """
    Opens sessions on the engine this database has already prepared, once it has one.
    """

    def open_session(self) -> Session:
        """
        Open a session against the results, preparing the database on the first call.
        """
        if self._sessions is None:
            self._sessions = self._prepared_sessions()
        return self._sessions()

    def _prepared_sessions(self) -> sessionmaker:
        """
        Connect, creating this module's tables if they are not there yet.

        Done once per database rather than once per session: reading the whole generated
        ``experiments`` schema and issuing its ``CREATE TABLE`` statements takes the best
        part of a minute, which a query answered while a run sorts cannot pay.

        Skips any table ORMatic could not assign a real column type to (surfaced as
        SQLAlchemy's ``NullType``, e.g. ``EpisodePlayerDAO.rdr_viewer`` for the
        ``RDRCaseViewer`` field it has no mapping for), together with every table that
        depends on a skipped one through a foreign key -- transitively, since joined-
        table inheritance chains more than one table deep. One unrelated, pre-existing
        gap in the huge generated ``experiments`` schema must not stop every other
        table, including this module's own, from being created; a table left out purely
        because it depends on a skipped one would otherwise fail with an "undefined
        table" error the moment ``CREATE TABLE`` tried to reference it.
        """
        engine = create_results_engine(self.uri)
        metadata = self._schema()
        metadata.create_all(engine, tables=self._creatable_tables(metadata))
        return sessionmaker(engine)

    @staticmethod
    def _schema() -> MetaData:
        """
        The generated ``experiments`` schema every result table belongs to.
        """
        import experiments.orm.ormatic_interface as ormatic_interface

        return ormatic_interface.Base.metadata

    @staticmethod
    def _creatable_tables(metadata: MetaData) -> List[Table]:
        """
        Every table of a schema that can actually be created.

        :param metadata: The schema to filter.
        """
        excluded = {
            name
            for name, table in metadata.tables.items()
            if any(isinstance(column.type, NullType) for column in table.columns)
        }
        spreading = True
        while spreading:
            spreading = False
            for name, table in metadata.tables.items():
                if name in excluded:
                    continue
                if any(
                    foreign_key.column.table.name in excluded
                    for foreign_key in table.foreign_keys
                ):
                    excluded.add(name)
                    spreading = True
        return [
            table for name, table in metadata.tables.items() if name not in excluded
        ]


# %% is it actually reachable


@dataclass
class UnreachableResultsDatabase(DataclassException):
    """
    Raised when the database a run would record its results to cannot be connected to.
    """

    database_uri: str
    """
    The database that could not be reached.
    """

    reason: str
    """
    What the driver said went wrong.
    """

    def error_message(self) -> str:
        return "Cannot reach the results database %s: %s" % (
            database_label(self.database_uri),
            self.reason,
        )

    def suggest_correction(self) -> str:
        return (
            "Provision it once with %s, or point the run at another database with "
            "--database-uri (for a throwaway one, --database-uri "
            "sqlite:///montessori.db). An authentication failure on an existing role "
            "usually means that role's password is not the one DEFAULT_DATABASE_URI "
            "assumes."
        ) % PROVISIONING_SCRIPT


def verify_reachable(database_uri: str) -> None:
    """
    Open and close one connection to the database a run would record to.

    :param database_uri: The database to reach.
    :raises UnreachableResultsDatabase: When no connection can be opened.
    """
    try:
        with create_results_engine(database_uri).connect():
            pass
    except SQLAlchemyError as error:
        raise UnreachableResultsDatabase(
            database_uri=database_uri, reason=str(error.orig or error).strip()
        ) from error


# %% can it actually be recorded to


@dataclass
class ReadOnlyResultsDatabase(DataclassException):
    """
    Raised when the database a run would record its results to refuses to be written to.
    """

    database_uri: str
    """
    The database that would not take a write.
    """

    reason: str
    """
    What the driver said went wrong.
    """

    def error_message(self) -> str:
        return "Cannot record results to %s: %s" % (
            database_label(self.database_uri),
            self.reason,
        )

    def suggest_correction(self) -> str:
        return (
            "The database is reachable but will not take a write, which usually means "
            "the role in that URI was provisioned for reading only. If no "
            "--database-uri was given, the URI came from the %s environment variable "
            "-- check whether a shell profile sets it -- or from DEFAULT_DATABASE_URI. "
            "Grant the role write access as %s does, or point the run at another "
            "database with --database-uri (for a throwaway one, --database-uri "
            "sqlite:///montessori.db)."
        ) % (DATABASE_URI_ENVIRONMENT_VARIABLE, PROVISIONING_SCRIPT)


def verify_writable(database_uri: str) -> None:
    """
    Write a table of its own to the database a run would record to, and drop it again.

    Reaching a database says nothing about recording to it: a read-only role connects,
    finds every table it needs already there, and is refused only by the first insert --
    a world build and a whole sort later.

    Dropped rather than rolled back because SQLite's driver commits a ``CREATE TABLE``
    whatever transaction it was issued in.

    ..note:: Proves the role can create and write a table of its own. A role that can do
        that but holds no insert privilege on a table someone else owns would still be
        refused at record time.

    :param database_uri: The database to write to.
    :raises ReadOnlyResultsDatabase: When the write is refused.
    """
    engine = create_results_engine(database_uri)
    probe = Table(
        WRITE_PROBE_TABLE_NAME, MetaData(), Column("value", Integer, primary_key=True)
    )
    try:
        with engine.begin() as connection:
            probe.create(connection)
            connection.execute(probe.insert().values(value=1))
    except SQLAlchemyError as error:
        raise ReadOnlyResultsDatabase(
            database_uri=database_uri, reason=str(error.orig or error).strip()
        ) from error
    probe.drop(engine)


# %% the pre-flight a launcher runs


def main(argument_list: Optional[List[str]] = None) -> int:
    """
    Check the database a run would record to, before the run itself starts.

    Reads only ``--database-uri`` and ignores everything else, so a launcher can forward
    the run's whole argument list without knowing which parts this check reads.

    Reports rather than refuses: an unreachable database is replaced by one in memory
    and a read-only one is recorded nothing to, so the only thing a database problem
    costs is the run's recorded history.

    :param argument_list: Arguments to read; the process's own when omitted.
    :return: 0, the run may go ahead.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-uri", default=None)
    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)
    arguments, _ = parser.parse_known_args(argument_list)
    if not arguments.record:
        print("Not recording this run's results.")
        return 0
    database = ConfiguredDatabase.resolve_reachable(arguments.database_uri)
    if database.fell_back_from is not None:
        print(database.fell_back_from, file=sys.stderr)
    try:
        verify_writable(database.uri)
    except ReadOnlyResultsDatabase as error:
        print(error, file=sys.stderr)
        print(
            "Sorting anyway; this run's results will not be recorded. Pass --no-record "
            "to ask for no database at all.",
            file=sys.stderr,
        )
        return 0
    print(database.describe())
    return 0


if __name__ == "__main__":
    sys.exit(main())
