"""
The guard that fails a test module which left worlds in memory.

The guard runs when a module finishes, so pytest reports it under whichever test of that
module happened to run last. What it says therefore has to carry the whole finding: how
many worlds are still there, how many a module may leave, and which tests created the
ones that survived.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import pytest

from ..living_worlds import (
    BEFORE_THE_FIRST_TEST,
    MAXIMUM_LIVING_WORLDS,
    LeakedWorldsError,
    LivingWorlds,
    UnwatchableWorldTypeError,
    WorldsLeftBehind,
)
from .dataset.leakable_object import LeakableObject, ObjectMakingItsOwnInstances


@dataclass(frozen=True)
class StandInTestNames:
    """
    Names a test below attributes created and surviving worlds to, standing in for the
    node ids a real test run would report.
    """

    finished_module: str
    """
    Stands in for the module whose end the guard checks.
    """

    leaking_test: str
    """
    Stands in for a test whose worlds are still in memory afterwards.
    """

    tidy_test: str
    """
    Stands in for a test that releases every world it created.
    """


@pytest.fixture()
def stand_in_test_names() -> StandInTestNames:
    """
    The stand-in names the tests below attribute worlds to.
    """
    finished_module = "test/semantic_digital_twin_test/test_world.py"
    return StandInTestNames(
        finished_module=finished_module,
        leaking_test=f"{finished_module}::test_that_leaks",
        tidy_test=f"{finished_module}::test_that_leaves_nothing",
    )


@pytest.fixture(scope="module")
def watched_objects() -> LivingWorlds:
    """
    The one record watching the stand-in, since watching a type cannot be undone.
    """
    worlds = LivingWorlds(world_type=LeakableObject)
    worlds.watch()
    return worlds


@pytest.fixture()
def living_worlds(watched_objects: LivingWorlds) -> LivingWorlds:
    """
    The record, holding nothing but what the test using it creates.
    """
    watched_objects.creations.clear()
    watched_objects.current_test = BEFORE_THE_FIRST_TEST
    return watched_objects


# %% which test the surviving worlds are attributed to


def test_the_tests_that_created_the_surviving_worlds_are_named(
    living_worlds: LivingWorlds, stand_in_test_names: StandInTestNames
):
    """
    The report names the test the surviving worlds came from, not the one that happened
    to be running when the count was taken.
    """
    living_worlds.current_test = stand_in_test_names.leaking_test
    leaked = [LeakableObject() for _ in range(3)]
    living_worlds.current_test = stand_in_test_names.tidy_test

    with pytest.raises(LeakedWorldsError) as leak:
        living_worlds.enforce_limit(module=stand_in_test_names.finished_module, limit=2)

    assert leak.value.left_behind == (
        WorldsLeftBehind(stand_in_test_names.leaking_test, len(leaked)),
    )


def test_a_test_whose_worlds_were_collected_is_not_named(
    living_worlds: LivingWorlds, stand_in_test_names: StandInTestNames
):
    living_worlds.current_test = stand_in_test_names.tidy_test
    for _ in range(5):
        LeakableObject()
    living_worlds.current_test = stand_in_test_names.leaking_test
    leaked = [LeakableObject() for _ in range(3)]

    with pytest.raises(LeakedWorldsError) as leak:
        living_worlds.enforce_limit(module=stand_in_test_names.finished_module, limit=2)

    assert leak.value.worlds_in_memory == len(leaked)
    assert [left_behind.test for left_behind in leak.value.left_behind] == [
        stand_in_test_names.leaking_test
    ]


def test_the_test_that_left_the_most_worlds_is_named_first(
    living_worlds: LivingWorlds, stand_in_test_names: StandInTestNames
):
    living_worlds.current_test = stand_in_test_names.tidy_test
    few = [LeakableObject()]
    living_worlds.current_test = stand_in_test_names.leaking_test
    many = [LeakableObject() for _ in range(3)]

    with pytest.raises(LeakedWorldsError) as leak:
        living_worlds.enforce_limit(module=stand_in_test_names.finished_module, limit=2)

    assert leak.value.left_behind == (
        WorldsLeftBehind(stand_in_test_names.leaking_test, len(many)),
        WorldsLeftBehind(stand_in_test_names.tidy_test, len(few)),
    )


def test_worlds_created_before_any_test_ran_are_not_blamed_on_a_test(
    living_worlds: LivingWorlds, stand_in_test_names: StandInTestNames
):
    leaked = [LeakableObject() for _ in range(3)]

    with pytest.raises(LeakedWorldsError) as leak:
        living_worlds.enforce_limit(module=stand_in_test_names.finished_module, limit=2)

    assert leak.value.left_behind == (
        WorldsLeftBehind(BEFORE_THE_FIRST_TEST, len(leaked)),
    )


def test_a_world_reaches_the_record_however_it_was_made(
    living_worlds: LivingWorlds, stand_in_test_names: StandInTestNames
):
    living_worlds.current_test = stand_in_test_names.leaking_test
    original = LeakableObject()
    copies = [copy.deepcopy(original), copy.copy(original)]

    with pytest.raises(LeakedWorldsError) as leak:
        living_worlds.enforce_limit(module=stand_in_test_names.finished_module, limit=2)

    assert leak.value.left_behind == (
        WorldsLeftBehind(stand_in_test_names.leaking_test, len(copies) + 1),
    )


# %% the limit the guard reports is the one it enforces


def test_a_module_within_the_limit_is_let_through(
    living_worlds: LivingWorlds, stand_in_test_names: StandInTestNames
):
    living_worlds.current_test = stand_in_test_names.leaking_test
    kept = [LeakableObject() for _ in range(MAXIMUM_LIVING_WORLDS)]

    living_worlds.enforce_limit(module=stand_in_test_names.finished_module)

    assert living_worlds.surviving_worlds() == (
        WorldsLeftBehind(stand_in_test_names.leaking_test, len(kept)),
    )


def test_the_reported_limit_is_the_enforced_one(
    living_worlds: LivingWorlds, stand_in_test_names: StandInTestNames
):
    """
    The number the report states is the number that made it fail, so that a reader is
    not sent looking for a leak of a size the guard never enforced.
    """
    living_worlds.current_test = stand_in_test_names.leaking_test
    leaked = [LeakableObject() for _ in range(MAXIMUM_LIVING_WORLDS + 1)]

    with pytest.raises(LeakedWorldsError) as leak:
        living_worlds.enforce_limit(module=stand_in_test_names.finished_module)

    assert leak.value.limit == MAXIMUM_LIVING_WORLDS
    assert leak.value.worlds_in_memory == len(leaked)
    assert str(MAXIMUM_LIVING_WORLDS) in str(leak.value)


# %% the watched type goes on creating its objects


def test_a_watched_type_creates_the_objects_it_is_asked_for(
    living_worlds: LivingWorlds,
):
    created = LeakableObject("the one asked for")

    assert created == LeakableObject(name="the one asked for")


def test_a_type_creating_its_own_instances_is_left_alone():
    worlds = LivingWorlds(world_type=ObjectMakingItsOwnInstances)
    creates_objects_itself = ObjectMakingItsOwnInstances.__new__

    with pytest.raises(UnwatchableWorldTypeError):
        worlds.watch()

    assert ObjectMakingItsOwnInstances.__new__ is creates_objects_itself
