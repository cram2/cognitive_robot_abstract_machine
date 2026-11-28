"""
A database written before the table of every stored object, brought to it.

Its tables numbered their rows each on its own, so a storage and a holder share ids; the
rows are written as such a database held them.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import Engine, insert, select
from sqlalchemy.orm import Session

from krrood.ormatic.stored_object import StoredObjectDAO
from krrood.ormatic.stored_object_backfill import StoredObjectBackfill
from krrood.ormatic.utils import create_engine

from ..dataset.ormatic_interface import (
    Base,
    HolderOfValueKeptInStorageDAO,
    StorageDAO,
    SubclassOfHolderDAO,
)

# %% a database written before the table of every stored object


@pytest.fixture
def database_before_stored_objects(tmp_path: Path) -> Engine:
    """
    Two storages and two holders, numbered 1 and 2 in each table: the plain holder keeps
    its value in the second storage, the subclass's holder in the first.
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'before.sqlite'}")
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(
            insert(StorageDAO.__table__),
            [
                {
                    "database_id": 1,
                    "value": 7,
                    "number_of_writes": 1,
                    "polymorphic_type": StorageDAO.__tablename__,
                },
                {
                    "database_id": 2,
                    "value": 9,
                    "number_of_writes": 1,
                    "polymorphic_type": StorageDAO.__tablename__,
                },
            ],
        )
        connection.execute(
            insert(HolderOfValueKeptInStorageDAO.__table__),
            [
                {
                    "database_id": 1,
                    "polymorphic_type": HolderOfValueKeptInStorageDAO.__tablename__,
                    "_storage_id": 2,
                },
                {
                    "database_id": 2,
                    "polymorphic_type": SubclassOfHolderDAO.__tablename__,
                    "_storage_id": 1,
                },
            ],
        )
        connection.execute(
            insert(SubclassOfHolderDAO.__table__),
            [{"database_id": 2, "other_value": 5}],
        )
    return engine


def stored_objects(engine: Engine) -> list[tuple[int, str]]:
    """
    :return: The rows of the table of every stored object, by id.
    """
    with engine.connect() as connection:
        return list(
            connection.execute(
                select(
                    StoredObjectDAO.__table__.c.database_id,
                    StoredObjectDAO.__table__.c.polymorphic_type,
                ).order_by(StoredObjectDAO.__table__.c.database_id)
            )
        )


# %% backfilling


def test_every_row_gets_an_id_of_its_own_and_its_stored_object_row(
    database_before_stored_objects,
):
    """
    The four rows that shared two ids get four, each with its row in the table of every
    stored object naming its class's table.
    """
    with database_before_stored_objects.begin() as connection:
        StoredObjectBackfill(metadata=Base.metadata).backfill(connection=connection)

    rows = stored_objects(database_before_stored_objects)
    assert len({database_id for database_id, _ in rows}) == 4
    assert sorted(polymorphic_type for _, polymorphic_type in rows) == sorted(
        [
            HolderOfValueKeptInStorageDAO.__tablename__,
            StorageDAO.__tablename__,
            StorageDAO.__tablename__,
            SubclassOfHolderDAO.__tablename__,
        ]
    )


def test_each_holder_still_refers_to_its_own_storage(database_before_stored_objects):
    """
    Renumbering moves the references with the rows: each holder loads with the storage
    it held, and the subclass's holder with its own field.
    """
    with database_before_stored_objects.begin() as connection:
        StoredObjectBackfill(metadata=Base.metadata).backfill(connection=connection)

    with Session(database_before_stored_objects) as session:
        holders = session.scalars(select(HolderOfValueKeptInStorageDAO)).all()
        loaded = sorted(
            (type(holder).__name__, holder.storage.value) for holder in holders
        )
        (subclass_holder,) = session.scalars(select(SubclassOfHolderDAO)).all()
        other_value = subclass_holder.other_value
    assert loaded == [
        (HolderOfValueKeptInStorageDAO.__name__, 9),
        (SubclassOfHolderDAO.__name__, 7),
    ]
    assert other_value == 5


def test_backfilling_again_changes_nothing(database_before_stored_objects):
    """
    Rows that have their row in the table of every stored object keep their ids.
    """
    backfill = StoredObjectBackfill(metadata=Base.metadata)
    with database_before_stored_objects.begin() as connection:
        backfill.backfill(connection=connection)
    backfilled = stored_objects(database_before_stored_objects)

    with database_before_stored_objects.begin() as connection:
        backfill.backfill(connection=connection)

    assert stored_objects(database_before_stored_objects) == backfilled
