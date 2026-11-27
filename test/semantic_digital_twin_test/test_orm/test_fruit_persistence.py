"""
A fruit keeps its parts, its ripeness and the material qualities of its parts when its
world is stored in the database and loaded back.
"""

from __future__ import annotations

# %% Imports

import pytest
from random_events.interval import Bound, SimpleInterval
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from krrood.ormatic.data_access_objects.helper import to_dao
from semantic_digital_twin.semantic_annotations.qualities.quality import (
    MassDensity,
    MaterialElasticity,
    Ripeness,
    RipenessStage,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Apple,
    Peel,
    Seed,
)
from semantic_digital_twin.world import World

# %% Fruits


def stored_and_loaded_apple(
    apple_world: World, session_maker: sessionmaker[Session]
) -> Apple:
    """
    :param apple_world: A world holding one apple.
    :param session_maker: Opens sessions on the database to store the world in.
    :return: The apple of the world after storing the world and loading it back.
    """
    stored_world = to_dao(apple_world)
    with session_maker() as session:
        session.add(stored_world)
        session.commit()
        session.expunge_all()
        loaded_world = session.scalar(select(type(stored_world))).from_dao()
    [apple] = loaded_world.get_semantic_annotations_by_type(Apple)
    return apple


def test_apple_keeps_its_ripeness_stage(
    in_memory_session_maker: sessionmaker[Session],
) -> None:
    """
    An apple's ripeness stage survives storing its world and loading it back.
    """
    world = World.create_with_root_body("root")
    with world.modify_world():
        apple = Apple.create_with_new_body_in_world(name="apple", world=world)
        apple.set_quality(quality=Ripeness.from_stage(stage=RipenessStage.RIPE))

    loaded_apple = stored_and_loaded_apple(world, in_memory_session_maker)

    assert loaded_apple.ripeness.stage == apple.ripeness.stage


def test_apple_keeps_its_parts(
    in_memory_session_maker: sessionmaker[Session],
) -> None:
    """
    An apple's parts survive storing its world and loading it back.
    """
    world = World.create_with_root_body("root")
    with world.modify_world():
        apple = Apple.create_with_new_body_in_world(name="apple", world=world)
        peel = Peel.create_with_new_body_in_world(name="peel", world=world)
        seed = Seed.create_with_new_body_in_world(name="seed", world=world)
        apple.add(part=peel)
        apple.add(part=seed)

    loaded_apple = stored_and_loaded_apple(world, in_memory_session_maker)

    assert loaded_apple.peel.name == peel.name
    assert [loaded_seed.name for loaded_seed in loaded_apple.seeds] == [seed.name]


def test_apple_peel_keeps_its_density_and_elasticity(
    in_memory_session_maker: sessionmaker[Session],
) -> None:
    """
    The density and elasticity of an apple's peel survive storing its world and loading
    it back, up to the single precision the interval ends are stored in.
    """
    world = World.create_with_root_body("root")
    with world.modify_world():
        apple = Apple.create_with_new_body_in_world(name="apple", world=world)
        peel = Peel.create_with_new_body_in_world(name="peel", world=world)
        apple.add(part=peel)
        peel.set_quality(
            quality=MassDensity(
                value=SimpleInterval.from_data(840.0, 860.0, Bound.CLOSED, Bound.CLOSED)
            )
        )
        peel.set_quality(
            quality=MaterialElasticity(
                youngs_modulus=SimpleInterval.from_data(
                    11.6e6, 11.6e6, Bound.CLOSED, Bound.CLOSED
                ),
                poissons_ratio=SimpleInterval.from_data(
                    0.33, 0.33, Bound.CLOSED, Bound.CLOSED
                ),
            )
        )

    loaded_peel = stored_and_loaded_apple(world, in_memory_session_maker).peel

    single_precision = 1e-6
    for loaded, stored in (
        (loaded_peel.mass_density.value, peel.mass_density.value),
        (
            loaded_peel.material_elasticity.youngs_modulus,
            peel.material_elasticity.youngs_modulus,
        ),
        (
            loaded_peel.material_elasticity.poissons_ratio,
            peel.material_elasticity.poissons_ratio,
        ),
    ):
        assert (loaded.lower, loaded.upper) == pytest.approx(
            (stored.lower, stored.upper), rel=single_precision
        )
