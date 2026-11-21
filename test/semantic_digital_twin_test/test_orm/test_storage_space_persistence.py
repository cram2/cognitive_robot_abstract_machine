"""
A storage space keeps objects whose classes are primarily other annotations, such as the
cabinets a counter top holds, and gets them back from the database.
"""

from __future__ import annotations

# %% Imports

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from krrood.ormatic.data_access_objects.helper import to_dao
from semantic_digital_twin.predefined_maps.kitchen_environment import KitchenEnvironment
from semantic_digital_twin.semantic_annotations.mixins import IsStorageSpace
from semantic_digital_twin.world import World

# %% Storage spaces


def held_objects(world: World) -> dict[str, list[tuple[str, str]]]:
    """
    :param world: A world.
    :return: The class and name of every object each storage space of the world holds,
        by the name of the storage space.
    """
    return {
        str(annotation.name): [
            (type(held).__name__, str(held.name)) for held in annotation.objects
        ]
        for annotation in world.semantic_annotations
        if isinstance(annotation, IsStorageSpace) and annotation.objects
    }


def test_storage_spaces_keep_objects_of_other_annotation_classes(
    in_memory_session_maker: sessionmaker[Session],
) -> None:
    """
    The kitchen's counter tops, cupboards and tables hold cabinets, ovens and other
    annotations whose primary class is not the type the storage space lists them as.
    """
    world = KitchenEnvironment().create_world()
    stored_world = to_dao(world)
    with in_memory_session_maker() as session:
        session.add(stored_world)
        session.commit()
        session.expunge_all()
        restored = session.scalar(select(type(stored_world))).from_dao()

    assert held_objects(restored) == held_objects(world)
