"""
Fixtures shared by the cramph tests: worlds, statechart contexts and executors.
"""

from __future__ import annotations

# %% ORM interfaces

# Built before the imports below, which read a mapped datastructure: pytest imports every
# conftest of a run before calling any hook, so a hook would fire too late. The build runs
# once per process and never on an xdist worker.
from cognitive_robot_abstract_machine.orm_interfaces import OrmPackage

from ..orm_interface_build import regenerate_orm_interfaces

regenerate_orm_interfaces(OrmPackage.CRAMPH)


import pytest

from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor
from krrood_spatial_math.spatial_types import Vector3
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    ScrewAxisConnection,
)
from semantic_digital_twin.world_description.world_entity import Body

from ..dataset.sliding_world import SlidingWorld

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


# %% the ORM mappers the tests use


@pytest.fixture(scope="session", autouse=True)
def register_orm_mappers() -> None:
    """
    Load cramph's ORM interface, whose import registers the mappers of it and the
    interfaces it builds on for every test.
    """
    import cramph.orm.ormatic_interface  # noqa: F401


# %% a world with a slider


@pytest.fixture()
def sliding_world() -> SlidingWorld:
    """
    :return: A world with a slider moving along x and a box fixed beside it.
    """
    world = World()
    root = Body(name=PrefixedName("root"))
    slider = Body(name=PrefixedName("slider"))
    box = Body(name=PrefixedName("box"))
    with world.modify_world():
        world.add_kinematic_structure_entity(root)
        slider_connection = ScrewAxisConnection.create_with_dofs(
            screw_axis=ScrewAxisConnection.create_prismatic_screw_axis(
                axis=Vector3.X()
            ),
            world=world,
            parent=root,
            child=slider,
        )
        world.add_connection(slider_connection)
        world.add_connection(FixedConnection(parent=root, child=box))
    return SlidingWorld(
        world=world, slider=slider, box=box, slider_connection=slider_connection
    )
