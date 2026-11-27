"""
Tests for the qualities the literature reports for the materials of semantic
annotations, and for setting them on an annotation and on the parts of a whole, such as
a fruit or an egg.
"""

from __future__ import annotations

# %% Imports

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from krrood.ormatic.data_access_objects.helper import to_dao
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.part_whole.measured_parts import (
    HasMeasuredParts,
)
from semantic_digital_twin.semantic_annotations.qualities.measured_material import (
    MeasuredMaterial,
)
from semantic_digital_twin.semantic_annotations.qualities.quality import Quality
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Apple,
    Avocado,
    Banana,
    Cereal,
    Core,
    Cucumber,
    Egg,
    Orange,
    Peach,
    Pit,
    Strawberry,
    Tomato,
    Watermelon,
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


def whole_with_mapped_parts(
    world: World, whole_type: type[HasMeasuredParts]
) -> HasMeasuredParts:
    """
    :param world: The world to create the whole in, inside its modification.
    :param whole_type: A kind of whole, such as a kind of fruit.
    :return: A whole of ``whole_type`` with one part of each kind its measured part
        materials map.
    """
    whole = whole_type.create_with_new_body_in_world(name="whole", world=world)
    for part_type in whole_type.measured_part_materials:
        whole.add(
            part=part_type.create_with_new_body_in_world(
                name=part_type.__name__, world=world
            )
        )
    return whole


# %% The measured materials


@pytest.mark.parametrize("material", list(MeasuredMaterial))
def test_every_measured_material_lies_on_the_scales_of_its_qualities(
    material: MeasuredMaterial,
) -> None:
    """
    Every reported value lies on the scale of its quality, so that creating the
    qualities of a measured material raises no
    :class:`~semantic_digital_twin.semantic_annotations.qualities.exceptions.QualityOutsideScale`.
    """
    material.value.create_qualities()


@pytest.mark.parametrize("material", list(MeasuredMaterial))
def test_every_measured_material_reports_a_quality(
    material: MeasuredMaterial,
) -> None:
    """
    A measured material reports at least one quality.
    """
    assert material.value.create_qualities()


def test_no_measured_material_is_another_under_a_second_name() -> None:
    """
    Two materials whose reported values were equal would make the enumeration alias one
    to the other.
    """
    assert len(MeasuredMaterial) == len(MeasuredMaterial.__members__)


def test_every_measured_material_is_the_material_of_some_annotation() -> None:
    """
    Every measured material is the material of a kind of annotation or of a part of a
    kind of whole.
    """
    annotation_types = every_subclass_of(HasRootBody)
    used = {
        annotation_type.measured_material for annotation_type in annotation_types
    } | {
        material
        for whole_type in every_subclass_of(HasMeasuredParts)
        for material in whole_type.measured_part_materials.values()
    }

    assert used - {None} == set(MeasuredMaterial)


def test_the_qualities_of_a_material_are_new_each_time() -> None:
    """
    Each annotation gets its own qualities, so no two share one.
    """
    first, second = (
        MeasuredMaterial.APPLE.value.create_qualities(),
        MeasuredMaterial.APPLE.value.create_qualities(),
    )

    assert first == second
    assert all(
        first_quality is not second_quality
        for first_quality, second_quality in zip(first, second)
    )


# %% Setting the measured qualities on an annotation


@pytest.mark.parametrize(
    "annotation_type",
    sorted(
        (
            annotation_type
            for annotation_type in every_subclass_of(HasRootBody)
            if annotation_type.measured_material is not None
        ),
        key=lambda annotation_type: annotation_type.__name__,
    ),
    ids=lambda annotation_type: annotation_type.__name__,
)
def test_an_annotation_gets_the_qualities_of_its_material(
    annotation_type: type[HasRootBody],
) -> None:
    """
    Setting the measured qualities of an annotation sets those reported for its
    material.
    """
    world = World.create_with_root_body("root")
    with world.modify_world():
        annotation = annotation_type.create_with_new_body_in_world(
            name="annotation", world=world
        )
        annotation.set_measured_qualities()

    assert by_type(annotation.qualities) == by_type(
        annotation_type.measured_material.value.create_qualities()
    )


def test_an_annotation_without_a_measured_material_keeps_no_qualities() -> None:
    """
    A kind of annotation whose material was not measured, such as a box of cereal, keeps
    its qualities empty.
    """
    world = World.create_with_root_body("root")
    with world.modify_world():
        cereal = Cereal.create_with_new_body_in_world(name="cereal", world=world)
        cereal.set_measured_qualities()

    assert cereal.qualities == []


# %% Setting the measured qualities on a whole and its parts

WHOLE_TYPES = [
    Apple,
    Avocado,
    Banana,
    Cucumber,
    Orange,
    Peach,
    Strawberry,
    Tomato,
    Watermelon,
    Egg,
]


@pytest.mark.parametrize(
    "whole_type", WHOLE_TYPES, ids=lambda whole_type: whole_type.__name__
)
def test_every_part_of_a_whole_gets_the_qualities_of_its_material(
    whole_type: type[HasMeasuredParts],
) -> None:
    """
    Setting the measured qualities of a whole, such as a fruit, sets on each part of a
    mapped kind the qualities reported for that part's material.
    """
    world = World.create_with_root_body("root")
    with world.modify_world():
        whole = whole_with_mapped_parts(world, whole_type)
        whole.set_measured_qualities()

    assert {type(part): by_type(part.qualities) for part in whole.parts} == {
        part_type: by_type(material.value.create_qualities())
        for part_type, material in whole_type.measured_part_materials.items()
    }


@pytest.mark.parametrize(
    "whole_type", WHOLE_TYPES, ids=lambda whole_type: whole_type.__name__
)
def test_a_whole_gets_the_qualities_of_its_own_material(
    whole_type: type[HasMeasuredParts],
) -> None:
    """
    The whole itself gets the qualities reported for it, and none if only its parts were
    measured.
    """
    world = World.create_with_root_body("root")
    with world.modify_world():
        whole = whole_with_mapped_parts(world, whole_type)
        whole.set_measured_qualities()

    expected = (
        []
        if whole_type.measured_material is None
        else whole_type.measured_material.value.create_qualities()
    )
    assert by_type(whole.qualities) == by_type(expected)


@pytest.mark.parametrize(
    "fruit_type, part_type",
    [(Apple, Core), (Avocado, Pit)],
)
def test_a_part_whose_material_was_not_measured_keeps_no_qualities(
    fruit_type: type[HasMeasuredParts], part_type: type[HasRootBody]
) -> None:
    """
    A part of a kind the fruit does not map, such as an apple's core, keeps its
    qualities empty.
    """
    world = World.create_with_root_body("root")
    with world.modify_world():
        fruit = fruit_type.create_with_new_body_in_world(name="fruit", world=world)
        part = part_type.create_with_new_body_in_world(name="part", world=world)
        fruit.add(part=part)
        fruit.set_measured_qualities()

    assert part.qualities == []


# %% Persistence


@pytest.mark.parametrize("whole_type", [Apple, Egg], ids=["Apple", "Egg"])
def test_the_measured_qualities_of_a_whole_survive_the_database(
    whole_type: type[HasMeasuredParts],
    in_memory_session_maker: sessionmaker[Session],
) -> None:
    """
    The measured qualities of a whole and of its parts survive storing the world and
    loading it back, up to the single precision the interval ends are stored in.
    """
    world = World.create_with_root_body("root")
    with world.modify_world():
        whole = whole_with_mapped_parts(world, whole_type)
        whole.set_measured_qualities()
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
