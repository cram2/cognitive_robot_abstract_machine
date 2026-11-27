"""
Tests for the qualities annotations start with, taken from the literature: defaults of
their quality fields, which a fresher source, such as perception, replaces.
"""

from __future__ import annotations

# %% Imports

from dataclasses import MISSING

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from krrood.ormatic.data_access_objects.helper import to_dao
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.qualities.quality import (
    MassDensity,
    MaterialElasticity,
    Quality,
)
from semantic_digital_twin.semantic_annotations.qualities.relationship_marker import (
    IsQualityRelationship,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Apple,
    AppleFlesh,
    ApplePeel,
    Cereal,
    Egg,
    EggShell,
    Flesh,
    Peel,
    Yolk,
    Albumen,
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


def default_qualities_of(annotation_type: type[HasRootBody]) -> list[Quality]:
    """
    :param annotation_type: A kind of annotation.
    :return: New qualities from the defaults of its quality fields, in field order.
    """
    return [
        wrapped_field.field.default_factory()
        for wrapped_field in annotation_type.semantic_relationship_fields(
            relationship_type=IsQualityRelationship
        )
        if wrapped_field.field.default_factory is not MISSING
    ]


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


ANNOTATIONS_WITH_DEFAULTS = sorted(
    (
        annotation_type
        for annotation_type in every_subclass_of(HasRootBody)
        if default_qualities_of(annotation_type)
    ),
    key=lambda annotation_type: annotation_type.__name__,
)

# %% Defaults of the quality fields


@pytest.mark.parametrize(
    "annotation_type",
    ANNOTATIONS_WITH_DEFAULTS,
    ids=lambda annotation_type: annotation_type.__name__,
)
def test_a_new_annotation_has_the_defaults_of_its_kind(
    annotation_type: type[HasRootBody],
) -> None:
    """
    A new annotation starts with the defaults of its quality fields, each of which lies
    on the scale of its quality.
    """
    world = World.create_with_root_body("root")
    with world.modify_world():
        annotation = annotation_type.create_with_new_body_in_world(
            name="annotation", world=world
        )

    assert by_type(annotation.qualities) == by_type(
        default_qualities_of(annotation_type)
    )


def test_two_annotations_share_no_default_quality() -> None:
    """
    Every annotation gets its own default qualities, so changing one changes no other.
    """
    world = World.create_with_root_body("root")
    with world.modify_world():
        first = Apple.create_with_new_body_in_world(name="first", world=world)
        second = Apple.create_with_new_body_in_world(name="second", world=world)

    assert first.mass_density == second.mass_density
    assert first.mass_density is not second.mass_density


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


# %% Parts of a particular kind of whole


@pytest.mark.parametrize(
    "part_type, generic_part_type",
    [(ApplePeel, Peel), (AppleFlesh, Flesh)],
    ids=["peel", "flesh"],
)
def test_the_part_of_an_apple_has_defaults_a_generic_part_does_not(
    part_type: type[HasRootBody], generic_part_type: type[HasRootBody]
) -> None:
    """
    A peel serves many fruits and has no defaults, while an apple's peel has those
    measured on apples, and either can be added to an apple.
    """
    world = World.create_with_root_body("root")
    with world.modify_world():
        apple = Apple.create_with_new_body_in_world(name="apple", world=world)
        part = part_type.create_with_new_body_in_world(name="part", world=world)
        apple.add(part=part)

    assert issubclass(part_type, generic_part_type)
    assert default_qualities_of(generic_part_type) == []
    assert by_type(part.qualities) == by_type(default_qualities_of(part_type))


# %% Materials


@pytest.mark.parametrize(
    "material",
    [
        lambda: MassDensity.steel,
        lambda: MassDensity.glass,
        lambda: MassDensity.polyurethane_foam,
        lambda: MassDensity.paper,
        lambda: MassDensity.cardboard,
        lambda: MaterialElasticity.steel,
        lambda: MaterialElasticity.glass,
    ],
    ids=[
        "density of steel",
        "density of glass",
        "density of polyurethane foam",
        "density of paper",
        "density of cardboard",
        "elasticity of steel",
        "elasticity of glass",
    ],
)
def test_a_material_gives_a_new_quality_on_each_access(material) -> None:
    """
    The quality of a material lies on its scale and is new on each access, so that no
    two annotations made of it share one.
    """
    first, second = material(), material()

    assert first == second
    assert first is not second


# %% Persistence


@pytest.mark.parametrize(
    "whole_type, part_types",
    [(Apple, [ApplePeel, AppleFlesh]), (Egg, [EggShell, Albumen, Yolk])],
    ids=["Apple", "Egg"],
)
def test_the_defaults_of_a_whole_and_its_parts_survive_the_database(
    whole_type: type[HasRootBody],
    part_types: list[type[HasRootBody]],
    in_memory_session_maker: sessionmaker[Session],
) -> None:
    """
    The default qualities of a whole and of its parts survive storing the world and
    loading it back, up to the single precision the interval ends are stored in.
    """
    world = World.create_with_root_body("root")
    with world.modify_world():
        whole = whole_type.create_with_new_body_in_world(name="whole", world=world)
        for part_type in part_types:
            whole.add(
                part=part_type.create_with_new_body_in_world(
                    name=part_type.__name__, world=world
                )
            )
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
