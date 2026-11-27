"""
Tests for the qualities an annotation starts with, taken from the literature, which its
parts take when they are added to it and which a fresher source, such as perception,
replaces.
"""

from __future__ import annotations

# %% Imports

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from krrood.ormatic.data_access_objects.helper import to_dao
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.part_whole.relationship import (
    HasParts,
)
from semantic_digital_twin.semantic_annotations.qualities.quality import (
    MassDensity,
    Quality,
)
from semantic_digital_twin.semantic_annotations.qualities.relationship import (
    HasQualities,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Apple,
    Avocado,
    Cereal,
    Core,
    Egg,
    Peel,
    Pit,
)
from semantic_digital_twin.world import World

# %% Helpers


def every_subclass_of(cls: type) -> set[type]:
    """
    :param cls: A class.
    :return: Every class that inherits from ``cls``, directly or not.
    """
    subclasses = set(cls.__subclasses__())
    for subclass in list(subclasses):
        subclasses |= every_subclass_of(subclass)
    return subclasses


def by_name(classes: set[type]) -> list[type]:
    """
    :param classes: Classes.
    :return: The classes sorted by name, so that test parameters are stable.
    """
    return sorted(classes, key=lambda cls: cls.__name__)


def by_type(qualities: list[Quality]) -> list[Quality]:
    """
    :param qualities: Qualities of one annotation, at most one of each type.
    :return: The qualities in the order of their type names.
    """
    return sorted(qualities, key=lambda quality: type(quality).__name__)


def interval_ends(qualities: list[Quality]) -> list[float]:
    """
    :param qualities: Qualities of one annotation, at most one of each type.
    :return: The lower and upper end of every interval the qualities hold, in the order
        of their type names and fields.
    """
    return [
        end
        for quality in by_type(qualities)
        for interval in vars(quality).values()
        if interval is not None
        for end in (interval.lower, interval.upper)
    ]


def whole_with_its_parts(world: World, whole_type: type[HasParts]) -> HasParts:
    """
    :param world: The world to create the whole in, inside its modification.
    :param whole_type: A kind of whole, such as a kind of fruit.
    :return: A whole of ``whole_type`` with one part of each kind it has default
        qualities for.
    """
    whole = whole_type.create_with_new_body_in_world(name="whole", world=world)
    for part_type in whole_type.default_part_qualities:
        whole.add(
            part=part_type.create_with_new_body_in_world(
                name=part_type.__name__, world=world
            )
        )
    return whole


ANNOTATIONS_WITH_DEFAULTS = by_name(
    {
        annotation_type
        for annotation_type in every_subclass_of(HasRootBody)
        if annotation_type.default_qualities
    }
)

WHOLES_WITH_PART_DEFAULTS = by_name(
    {
        whole_type
        for whole_type in every_subclass_of(HasParts)
        if whole_type.default_part_qualities
    }
)

# %% The default qualities


@pytest.mark.parametrize(
    "annotation_type",
    ANNOTATIONS_WITH_DEFAULTS,
    ids=lambda annotation_type: annotation_type.__name__,
)
def test_the_defaults_of_an_annotation_are_new_on_each_call(
    annotation_type: type[HasQualities],
) -> None:
    """
    Every default lies on the scale of its quality, since creating it raises no :class:`
    ~semantic_digital_twin.semantic_annotations.qualities.exceptions.QualityOutsideScale
    `, and each call creates new qualities, so that no two annotations share one.
    """
    first, second = (
        annotation_type.default_qualities,
        annotation_type.default_qualities,
    )

    assert first == second
    assert all(
        first_quality is not second_quality
        for first_quality, second_quality in zip(first, second)
    )


@pytest.mark.parametrize(
    "annotation_type",
    ANNOTATIONS_WITH_DEFAULTS,
    ids=lambda annotation_type: annotation_type.__name__,
)
def test_a_new_annotation_has_the_defaults_of_its_kind(
    annotation_type: type[HasRootBody],
) -> None:
    """
    A new annotation starts with the default qualities of its kind.
    """
    world = World.create_with_root_body("root")
    with world.modify_world():
        annotation = annotation_type.create_with_new_body_in_world(
            name="annotation", world=world
        )

    assert by_type(annotation.qualities) == by_type(annotation_type.default_qualities)


def test_an_annotation_without_defaults_starts_without_qualities() -> None:
    """
    A kind of annotation whose qualities were not measured, such as a box of cereal,
    starts without qualities.
    """
    world = World.create_with_root_body("root")
    with world.modify_world():
        cereal = Cereal.create_with_new_body_in_world(name="cereal", world=world)

    assert cereal.qualities == []


def test_a_perceived_quality_replaces_the_default() -> None:
    """
    A quality set later, such as a perceived density, replaces the default of its type,
    since it is fresher.
    """
    perceived = MassDensity(value=Quality.closed_range(lower=950.0, upper=950.0))
    world = World.create_with_root_body("root")
    with world.modify_world():
        apple = Apple.create_with_new_body_in_world(name="apple", world=world)
        apple.set_quality(quality=perceived)

    assert apple.mass_density is perceived


# %% The default qualities of parts


@pytest.mark.parametrize(
    "whole_type",
    WHOLES_WITH_PART_DEFAULTS,
    ids=lambda whole_type: whole_type.__name__,
)
def test_every_added_part_takes_the_defaults_of_its_whole(
    whole_type: type[HasParts],
) -> None:
    """
    A part added to a whole, such as a peel to an apple, takes the qualities the whole
    has for its kind of part.
    """
    world = World.create_with_root_body("root")
    with world.modify_world():
        whole = whole_with_its_parts(world, whole_type)

    assert {type(part): by_type(part.qualities) for part in whole.parts} == {
        part_type: by_type(qualities)
        for part_type, qualities in whole_type.default_part_qualities.items()
    }


def test_a_part_keeps_a_quality_it_already_has() -> None:
    """
    A part that already has a quality, such as a perceived density, keeps it when it is
    added to a whole whose default is of the same type.
    """
    perceived = MassDensity(value=Quality.closed_range(lower=700.0, upper=700.0))
    world = World.create_with_root_body("root")
    with world.modify_world():
        egg = Egg.create_with_new_body_in_world(name="egg", world=world)
        part_type = next(
            part_type
            for part_type, qualities in Egg.default_part_qualities.items()
            if any(isinstance(quality, MassDensity) for quality in qualities)
        )
        part = part_type.create_with_new_body_in_world(name="part", world=world)
        part.set_quality(quality=perceived)
        egg.add(part=part)

    assert part.mass_density is perceived


@pytest.mark.parametrize(
    "whole_type, part_type",
    [(Apple, Core), (Avocado, Pit)],
)
def test_a_part_without_defaults_starts_without_qualities(
    whole_type: type[HasParts], part_type: type[HasRootBody]
) -> None:
    """
    A part of a kind the whole has no defaults for, such as an apple's core, starts
    without qualities.
    """
    world = World.create_with_root_body("root")
    with world.modify_world():
        whole = whole_type.create_with_new_body_in_world(name="whole", world=world)
        part = part_type.create_with_new_body_in_world(name="part", world=world)
        whole.add(part=part)

    assert part.qualities == []


def test_a_part_alone_has_no_defaults() -> None:
    """
    A peel serves many fruits, so a peel that belongs to no fruit has no qualities.
    """
    world = World.create_with_root_body("root")
    with world.modify_world():
        peel = Peel.create_with_new_body_in_world(name="peel", world=world)

    assert peel.qualities == []


# %% Persistence


@pytest.mark.parametrize("whole_type", [Apple, Egg], ids=["Apple", "Egg"])
def test_the_defaults_of_a_whole_and_its_parts_survive_the_database(
    whole_type: type[HasParts],
    in_memory_session_maker: sessionmaker[Session],
) -> None:
    """
    The default qualities of a whole and of its parts survive storing the world and
    loading it back, up to the single precision the interval ends are stored in.
    """
    world = World.create_with_root_body("root")
    with world.modify_world():
        whole = whole_with_its_parts(world, whole_type)
    stored_world = to_dao(world)
    with in_memory_session_maker() as session:
        session.add(stored_world)
        session.commit()
        session.expunge_all()
        loaded_world = session.scalar(select(type(stored_world))).from_dao()
    [loaded_whole] = loaded_world.get_semantic_annotations_by_type(whole_type)
    loaded_parts = {type(part): part for part in loaded_whole.parts}

    for loaded, stored in [(loaded_whole, whole)] + [
        (loaded_parts[type(part)], part) for part in whole.parts
    ]:
        assert interval_ends(loaded.qualities) == pytest.approx(
            interval_ends(stored.qualities), rel=1e-6
        )
