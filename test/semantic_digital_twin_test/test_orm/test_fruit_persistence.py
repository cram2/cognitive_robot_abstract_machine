"""
A fruit keeps its parts and its ripeness when its world is stored in the database and
loaded back.
"""

from __future__ import annotations

# %% Imports

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from krrood.ormatic.data_access_objects.helper import to_dao
from semantic_digital_twin.semantic_annotations.qualities import Ripe, Ripeness
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
        apple.set_quality(quality=Ripeness.from_stage(stage=Ripe()))

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
        apple.add(peel)
        apple.add(seed)

    loaded_apple = stored_and_loaded_apple(world, in_memory_session_maker)

    assert loaded_apple.peel.name == peel.name
    assert [loaded_seed.name for loaded_seed in loaded_apple.seeds] == [seed.name]
