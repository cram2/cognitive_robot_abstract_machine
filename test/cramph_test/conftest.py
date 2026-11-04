# %% ORM interfaces

# Built before the imports below, which read a mapped datastructure: pytest imports every
# conftest of a run before calling any hook, so a hook would fire too late. The build runs
# once per process and never on an xdist worker.
from ..orm_interface_build import regenerate_orm_interfaces

regenerate_orm_interfaces()


import pytest

from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor
from semantic_digital_twin.world import World

# %% statechart context and executor

TICK_DURATION = 0.05
"""
How many seconds one tick of :func:`statechart_context` stands for.
"""


@pytest.fixture()
def empty_world() -> World:
    """
    :return: A world without any body.
    """
    return World()


@pytest.fixture()
def statechart_context(empty_world: World) -> StatechartContext:
    """
    :return: A context on an empty world whose ticks last :data:`TICK_DURATION`.
    """
    return StatechartContext(world=empty_world, tick_duration=TICK_DURATION)


@pytest.fixture()
def statechart_context_without_tick_duration(
    empty_world: World,
) -> StatechartContext:
    """
    :return: A context on an empty world that does not know how long a tick lasts.
    """
    return StatechartContext(world=empty_world)


@pytest.fixture()
def statechart_executor(
    statechart_context: StatechartContext,
) -> StatechartExecutor:
    """
    :return: An executor ticking statecharts in :func:`statechart_context`.
    """
    return StatechartExecutor(context=statechart_context)
