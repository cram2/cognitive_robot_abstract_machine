"""
Tests for the anatomy of fruits: which parts each fruit takes and where they go.
"""

from __future__ import annotations

# %% Imports

import pytest

from semantic_digital_twin.exceptions import CannotBeAPartOf
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Apple,
    Avocado,
    Banana,
    CitrusSegment,
    Core,
    Cucumber,
    Flesh,
    Fruit,
    FruitPart,
    Orange,
    Peach,
    Peel,
    Pit,
    Pith,
    Seed,
    SeedChamber,
    Stem,
    Strawberry,
    Watermelon,
)
from semantic_digital_twin.world import World

# %% Fixtures


@pytest.fixture
def world() -> World:
    """
    An empty world with a root body to spawn fruits and their parts into.
    """
    return World.create_with_root_body("root")


# %% Anatomy of each fruit


@pytest.mark.parametrize(
    "fruit_type, part_types",
    [
        (Apple, [Peel, Flesh, Core, Seed, Stem]),
        (Avocado, [Peel, Flesh, Pit]),
        (Banana, [Peel, Flesh]),
        (Cucumber, [Peel, Flesh, SeedChamber]),
        (Orange, [Peel, CitrusSegment]),
        (Peach, [Peel, Flesh, Pit, Stem]),
        (Strawberry, [Peel, Flesh, Pith]),
        (Watermelon, [Peel, Flesh, Seed]),
    ],
)
def test_fruit_takes_each_part_of_its_anatomy(
    world: World, fruit_type: type[Fruit], part_types: list[type[FruitPart]]
) -> None:
    """
    Each fruit accepts one part of every kind its anatomy has, and lists exactly those
    as its parts.
    """
    with world.modify_world():
        fruit = fruit_type.create_with_new_body_in_world(name="fruit", world=world)
        parts = [
            part_type.create_with_new_body_in_world(
                name=part_type.__name__, world=world
            )
            for part_type in part_types
        ]
        for part in parts:
            fruit.add(part)

    assert set(fruit.parts) == set(parts)


# %% Routing of parts


def test_parts_of_an_apple_land_in_their_fields(world: World) -> None:
    """
    Each part added to an apple lands in the field of its kind, seeds in order.
    """
    with world.modify_world():
        apple = Apple.create_with_new_body_in_world(name="apple", world=world)
        peel = Peel.create_with_new_body_in_world(name="peel", world=world)
        flesh = Flesh.create_with_new_body_in_world(name="flesh", world=world)
        core = Core.create_with_new_body_in_world(name="core", world=world)
        stem = Stem.create_with_new_body_in_world(name="stem", world=world)
        first_seed = Seed.create_with_new_body_in_world(name="seed_1", world=world)
        second_seed = Seed.create_with_new_body_in_world(name="seed_2", world=world)
        for part in (peel, flesh, core, stem, first_seed, second_seed):
            apple.add(part)

    assert apple.peel is peel
    assert apple.flesh is flesh
    assert apple.core is core
    assert apple.stem is stem
    assert apple.seeds == [first_seed, second_seed]


def test_citrus_segments_of_an_orange_land_in_their_field(world: World) -> None:
    """
    Citrus segments added to an orange land in its citrus segments field, in order.
    """
    with world.modify_world():
        orange = Orange.create_with_new_body_in_world(name="orange", world=world)
        segments = [
            CitrusSegment.create_with_new_body_in_world(
                name=f"segment_{number}", world=world
            )
            for number in range(3)
        ]
        for segment in segments:
            orange.add(segment)

    assert orange.citrus_segments == segments


def test_part_hangs_from_its_fruit(world: World) -> None:
    """
    An added part becomes a kinematic child of its fruit.
    """
    with world.modify_world():
        peach = Peach.create_with_new_body_in_world(name="peach", world=world)
        pit = Pit.create_with_new_body_in_world(name="pit", world=world)
        peach.add(pit)

    assert pit.root.parent_kinematic_structure_entity == peach.root


# %% Parts a fruit does not have


@pytest.mark.parametrize(
    "fruit_type, part_type",
    [(Apple, Pit), (Orange, Flesh), (Banana, Seed)],
)
def test_fruit_rejects_a_part_outside_its_anatomy(
    world: World, fruit_type: type[Fruit], part_type: type[FruitPart]
) -> None:
    """
    A fruit refuses a part its anatomy does not have.
    """
    with world.modify_world():
        fruit = fruit_type.create_with_new_body_in_world(name="fruit", world=world)
        part = part_type.create_with_new_body_in_world(name="part", world=world)
        with pytest.raises(CannotBeAPartOf):
            fruit.add(part)
