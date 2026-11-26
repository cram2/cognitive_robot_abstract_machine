"""
Tests for the qualities of semantic annotations: ripeness, its stages and the fruits
that carry it.
"""

from __future__ import annotations

# %% Imports

from dataclasses import dataclass, field

import pytest
from random_events.interval import Bound, SimpleInterval

from semantic_digital_twin.semantic_annotations.exceptions import (
    AmbiguousQuality,
    CannotHaveQuality,
    RipenessOutsideScale,
    RipenessSpansSeveralStages,
)
from semantic_digital_twin.semantic_annotations.mixins import HasRipeness, HasRootBody
from semantic_digital_twin.semantic_annotations.qualities import (
    IsQualityRelationship,
    Overripe,
    Quality,
    Ripe,
    Ripeness,
    RipenessStage,
    Unripe,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Apple,
    Avocado,
    Banana,
    Cucumber,
    Fruit,
    Orange,
    Peach,
    Strawberry,
    Watermelon,
)
from semantic_digital_twin.world import World

# %% Fixtures


@pytest.fixture
def world() -> World:
    """
    An empty world with a root body to spawn fruits into.
    """
    return World.create_with_root_body("root")


# %% Ripeness stages


@pytest.mark.parametrize("stage", [Unripe(), Ripe(), Overripe()])
def test_ripeness_known_only_by_its_stage_lies_in_that_stage(
    stage: RipenessStage,
) -> None:
    """
    A ripeness made from a stage alone is classified as that stage.
    """
    assert Ripeness.from_stage(stage=stage).stage == stage


def test_measured_ripeness_lies_in_the_stage_containing_it() -> None:
    """
    A measured ripeness, a single point, is classified as the stage around it.
    """
    ripe_interval = Ripe.ripeness_interval
    centre = (ripe_interval.lower + ripe_interval.upper) / 2
    ripeness = Ripeness(
        value=SimpleInterval.from_data(centre, centre, Bound.CLOSED, Bound.CLOSED)
    )

    assert ripeness.stage == Ripe()


def test_ripeness_straddling_two_stages_has_no_single_stage() -> None:
    """
    A ripeness reaching into two stages raises instead of picking one of them.
    """
    ripeness = Ripeness(
        value=SimpleInterval.from_data(
            Unripe.ripeness_interval.lower,
            Ripe.ripeness_interval.upper,
            Bound.CLOSED,
            Bound.OPEN,
        )
    )

    with pytest.raises(RipenessSpansSeveralStages):
        ripeness.stage


def test_ripeness_beyond_the_scale_is_rejected() -> None:
    """
    A ripeness value beyond the ripeness scale is refused when it is made.
    """
    with pytest.raises(RipenessOutsideScale):
        Ripeness(
            value=SimpleInterval.from_data(
                Ripeness.scale.upper, Ripeness.scale.upper + 1, Bound.OPEN, Bound.CLOSED
            )
        )


def test_stages_cover_the_ripeness_scale_without_overlap() -> None:
    """
    The stages divide the whole ripeness scale between them, without overlapping.
    """
    stage_intervals = [
        stage.ripeness_interval.as_composite_set() for stage in (Unripe, Ripe, Overripe)
    ]
    covered = stage_intervals[0]
    for stage_interval in stage_intervals[1:]:
        assert covered.intersection_with(stage_interval).is_empty()
        covered = covered.union_with(stage_interval)

    assert covered == Ripeness.scale.as_composite_set()


# %% Ripeness of fruits


@pytest.mark.parametrize(
    "fruit_type",
    [Apple, Avocado, Banana, Cucumber, Orange, Peach, Strawberry, Watermelon],
)
def test_every_fruit_starts_without_a_ripeness(
    world: World, fruit_type: type[Fruit]
) -> None:
    """
    Every fruit has a ripeness field, empty until a ripeness is set.
    """
    with world.modify_world():
        fruit = fruit_type.create_with_new_body_in_world(name="fruit", world=world)

    assert isinstance(fruit, HasRipeness)
    assert fruit.ripeness is None


def test_fruit_without_a_ripeness_has_no_qualities(world: World) -> None:
    """
    A fruit whose ripeness is unknown lists no qualities.
    """
    with world.modify_world():
        apple = Apple.create_with_new_body_in_world(name="apple", world=world)

    assert apple.qualities == []


def test_ripeness_of_a_fruit_is_its_quality(world: World) -> None:
    """
    Setting a ripeness on a fruit fills its ripeness field and lists it among its qualities.
    """
    ripeness = Ripeness.from_stage(stage=Ripe())
    with world.modify_world():
        apple = Apple.create_with_new_body_in_world(name="apple", world=world)
        apple.set_quality(quality=ripeness)

    assert apple.ripeness is ripeness
    assert apple.qualities == [ripeness]


# %% Setting qualities


@dataclass(kw_only=True)
class Firmness(Quality):
    """
    A quality no fruit has a field for.
    """


@dataclass(eq=False, kw_only=True)
class AnnotationWithTwoRipenessFields(HasRipeness, HasRootBody):
    """
    An annotation with a second field that also accepts a ripeness.
    """

    second_ripeness: Ripeness | None = field(
        default=None,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    Another field accepting a ripeness.
    """


def test_setting_a_quality_without_a_field_for_it_is_rejected(world: World) -> None:
    """
    Setting a quality the annotation has no field for raises.
    """
    with world.modify_world():
        apple = Apple.create_with_new_body_in_world(name="apple", world=world)
        with pytest.raises(CannotHaveQuality):
            apple.set_quality(quality=Firmness())


def test_setting_a_quality_two_fields_accept_is_rejected(world: World) -> None:
    """
    Setting a quality that two fields of the annotation accept raises instead of picking one.
    """
    with world.modify_world():
        annotation = AnnotationWithTwoRipenessFields.create_with_new_body_in_world(
            name="annotation", world=world
        )
        with pytest.raises(AmbiguousQuality):
            annotation.set_quality(quality=Ripeness.from_stage(stage=Ripe()))
