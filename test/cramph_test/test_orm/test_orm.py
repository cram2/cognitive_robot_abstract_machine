import pytest
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine
from sqlalchemy import select
from sqlalchemy.orm import Session

from cramph.monitors import CountTicks
from cramph.orm.ormatic_interface import Base, CountTicksDAO

# %% fixtures


@pytest.fixture
def session():
    session = Session(create_engine("sqlite:///:memory:"))
    Base.metadata.create_all(bind=session.bind)
    yield session
    Base.metadata.drop_all(session.bind)
    session.close()


@pytest.fixture
def tick_counting_node() -> CountTicks:
    return CountTicks(name="count three ticks", ticks=3)


# %% statechart node persistence


class TestStatechartNodeSurvivesRoundTrip:
    """
    The nodes of a statechart are mapped by cramph's own interface.
    """

    def test_node_fields_are_read_back_unchanged(
        self, session, tick_counting_node
    ) -> None:
        session.add(to_dao(tick_counting_node))
        session.commit()
        session.expunge_all()

        loaded = session.scalars(select(CountTicksDAO)).one()

        assert loaded.ticks == tick_counting_node.ticks
        assert loaded.name == tick_counting_node.name
