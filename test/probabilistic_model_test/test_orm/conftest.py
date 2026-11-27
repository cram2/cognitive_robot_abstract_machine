"""
Fixtures shared by the tests that store probabilistic models in a database.
"""

from __future__ import annotations

# %% ORM interface

# Built before the imports below, which read a mapped datastructure: pytest imports every
# conftest of a run before calling any hook, so a hook would fire too late. The build runs
# once per process and never on an xdist worker.
from cognitive_robot_abstract_machine.orm_interfaces import OrmPackage

from ...orm_interface_build import regenerate_orm_interfaces

regenerate_orm_interfaces(OrmPackage.PROBABILISTIC_MODEL)


import pytest
from sqlalchemy.orm import Session
from typing_extensions import Iterator

from krrood.ormatic.utils import create_engine

# %% database session


@pytest.fixture
def session() -> Iterator[Session]:
    """
    A session of an empty in memory database with the tables of the probabilistic model
    interface.
    """
    from probabilistic_model.orm.ormatic_interface import Base

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as database_session:
        yield database_session
