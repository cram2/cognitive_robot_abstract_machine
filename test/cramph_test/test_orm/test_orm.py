"""
Statecharts and their nodes survive being stored through cramph's ORM interface.
"""

import pytest
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine
from sqlalchemy import select
from sqlalchemy.orm import Session

from cramph.composites import Sequence
from cramph.executor import StatechartExecutor
from cramph.monitors import CountTicks
from cramph.node import CancelStatechart, EndStatechart
from cramph.nodes_for_testing import (
    ConstFalseNode,
    ConstTrueNode,
    NodeAssertionError,
    NodeAssertionErrorNamingANode,
)
from cramph.orm.ormatic_interface import Base, CountTicksDAO
from cramph.statechart import Statechart

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


@pytest.fixture
def statechart_run_to_its_end(statechart_executor: StatechartExecutor) -> Statechart:
    """
    :return: A statechart counting ticks in two steps one after another, ticked until it
        ended.
    """
    steps = Sequence(
        nodes=[
            CountTicks(name="count two ticks", ticks=2),
            CountTicks(name="count three ticks", ticks=3),
        ]
    )
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(steps)
    statechart.add_node(EndStatechart.when_true(steps))
    statechart_executor.compile(statechart)
    statechart_executor.tick_until_end(timeout=100)
    return statechart


def store_and_read_back(session: Session, statechart: Statechart) -> Statechart:
    """
    :param session: The session to store `statechart` through.
    :param statechart: The statechart to store.
    :return: `statechart` as read back from the database.
    """
    stored = to_dao(statechart)
    session.add(stored)
    session.commit()
    session.expunge_all()
    return session.scalars(select(type(stored))).one().from_dao()


@pytest.fixture
def statechart_read_back(session, statechart_run_to_its_end: Statechart) -> Statechart:
    """
    :return: :func:`statechart_run_to_its_end` as read back from the database.
    """
    return store_and_read_back(session, statechart_run_to_its_end)


@pytest.fixture
def cancelled_statechart(statechart_executor: StatechartExecutor) -> Statechart:
    """
    :return: A statechart with a node still running, ticked until a cancel node ended
        it with an error naming that node.
    """
    running = ConstFalseNode()
    trigger = ConstTrueNode()
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_nodes(
        [
            running,
            trigger,
            CancelStatechart.when_true(
                trigger,
                NodeAssertionErrorNamingANode(reason="cancelled", node=running),
            ),
        ]
    )
    statechart_executor.compile(statechart)
    with pytest.raises(NodeAssertionError):
        statechart_executor.tick_until_end(timeout=100)
    return statechart


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


# %% statechart persistence


class TestStatechartSurvivesRoundTrip:
    """
    A statechart is read back with its nodes, the transitions between them and the
    history of its state.
    """

    def test_nodes_are_read_back_in_index_order(
        self, statechart_run_to_its_end: Statechart, statechart_read_back: Statechart
    ) -> None:
        assert [
            (type(node), node.name, node.parent_node_index)
            for node in statechart_read_back.nodes
        ] == [
            (type(node), node.name, node.parent_node_index)
            for node in statechart_run_to_its_end.nodes
        ]

    def test_transitions_connect_the_same_nodes(
        self, statechart_run_to_its_end: Statechart, statechart_read_back: Statechart
    ) -> None:
        def transitions(statechart: Statechart) -> list[tuple[int, int, str]]:
            graph = statechart.rx_graph
            return sorted(
                (source, destination, condition.kind.name)
                for source, destination, condition in graph.edge_index_map().values()
            )

        assert transitions(statechart_read_back) == transitions(
            statechart_run_to_its_end
        )

    def test_state_history_is_read_back(
        self, statechart_run_to_its_end: Statechart, statechart_read_back: Statechart
    ) -> None:
        def history(statechart: Statechart) -> list[tuple[int, list, list]]:
            return [
                (
                    item.tick_count,
                    item.life_cycle_state.data.tolist(),
                    item.observation_state.data.tolist(),
                )
                for item in statechart.state_history.items
            ]

        assert history(statechart_read_back) == history(statechart_run_to_its_end)

    def test_current_state_is_the_last_one_recorded(
        self, statechart_run_to_its_end: Statechart, statechart_read_back: Statechart
    ) -> None:
        assert (
            statechart_read_back.life_cycle_state
            == statechart_run_to_its_end.life_cycle_state
        )
        assert (
            statechart_read_back.observation_state
            == statechart_run_to_its_end.observation_state
        )

    def test_cancelling_node_is_read_back(
        self, session, cancelled_statechart: Statechart
    ) -> None:
        read_back = store_and_read_back(session, cancelled_statechart)

        assert [
            None if item.cancelled_by is None else item.cancelled_by.index
            for item in read_back.state_history.items
        ] == [
            None if item.cancelled_by is None else item.cancelled_by.index
            for item in cancelled_statechart.state_history.items
        ]

    def test_cancellation_error_is_read_back_naming_the_same_node(
        self, session, cancelled_statechart: Statechart
    ) -> None:
        read_back = store_and_read_back(session, cancelled_statechart)

        original = cancelled_statechart.state_history.latest_item.cancellation_error
        error = read_back.state_history.latest_item.cancellation_error
        assert (type(error), error.reason, error.node) == (
            type(original),
            original.reason,
            read_back.get_node_by_index(original.node.index),
        )
