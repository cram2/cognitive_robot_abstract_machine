from __future__ import annotations

from sqlalchemy import create_engine, inspect

from segmind.orm.ormatic_interface import Base

# %% database tables


def test_every_table_of_segmind_can_be_created_in_a_database() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    assert set(inspect(engine).get_table_names()) == set(Base.metadata.tables)
    engine.dispose()
