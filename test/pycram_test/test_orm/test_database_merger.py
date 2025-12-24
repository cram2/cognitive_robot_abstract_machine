import json
import pathlib
from dataclasses import dataclass, field

import pytest
import sqlalchemy

import pycram.orm.utils


# Skip entire module if the merger configuration is missing (mirrors original unittest.skipIf)
pytestmark = pytest.mark.skipif(
    not (pathlib.Path("test_database_merger.json").resolve().is_file()),
    reason="Config File not found: test_database_merger.json",
)


@dataclass
class Configuration:
    user: str = field(default="alice")
    password: str = field(default="alice123")
    ipaddress: str = field(default="localhost")
    port: int = field(default=5432)
    database: str = field(default="pycram")


@pytest.fixture(scope="module")
def orm_session_makers():
    """Create SQLAlchemy engines and session makers for source (SQLite in-memory) and destination (Postgres).

    Reads connection from test_database_merger.json and prepares empty schemas.
    """
    with open("test_database_merger.json") as f:
        json_data = json.load(f)
        config = Configuration(**json_data)

    connection_string = f"postgresql+psycopg2://{config.user}:{config.password}@{config.ipaddress}:{config.port}/{config.database}"
    destination_engine = sqlalchemy.create_engine(connection_string, echo=False)
    source_engine = sqlalchemy.create_engine("sqlite+pysqlite:///:memory:", echo=False)

    source_session_maker = sqlalchemy.orm.sessionmaker(bind=source_engine)
    destination_session_maker = sqlalchemy.orm.sessionmaker(bind=destination_engine)

    # Ensure schemas exist
    pycram.orm.utils.mapper_registry.metadata.create_all(destination_engine)

    yield source_session_maker, destination_session_maker, source_engine


@pytest.fixture()
def populated_source_db(orm_session_makers):
    """Populate the source (SQLite) database. Here we just ensure the schema exists; content may be empty."""
    source_session_maker, destination_session_maker, source_engine = orm_session_makers

    # Create tables for the in-memory database each time (new connection resets)
    pycram.orm.utils.mapper_registry.metadata.create_all(source_engine)

    # No content insertion needed for merge mechanics tests
    return source_session_maker, destination_session_maker


def _snapshot_tables(session_maker):
    """Return a mapping table -> set(rows) for easy equality checks."""
    content = {}
    with session_maker() as session:
        for table in pycram.orm.utils.mapper_registry.metadata.sorted_tables:
            table_content_set = set()
            table_content_set.update(session.query(table).all())
            content[table] = table_content_set
    return content


def test_merge_databases(populated_source_db):
    source_session_maker, destination_session_maker = populated_source_db

    pycram.orm.utils.update_primary_key_constrains(destination_session_maker)
    pycram.orm.utils.update_primary_key(source_session_maker, destination_session_maker)
    pycram.orm.utils.copy_database(source_session_maker, destination_session_maker)

    destination_content = _snapshot_tables(destination_session_maker)
    source_content = _snapshot_tables(source_session_maker)

    for key in destination_content:
        assert destination_content[key] == destination_content[key].union(
            source_content[key]
        )


def test_migrate_neems(populated_source_db):
    source_session_maker, destination_session_maker = populated_source_db

    pycram.orm.utils.migrate_neems(source_session_maker, destination_session_maker)

    destination_content = _snapshot_tables(destination_session_maker)
    source_content = _snapshot_tables(source_session_maker)

    for key in destination_content:
        assert destination_content[key] == destination_content[key].union(
            source_content[key]
        )
