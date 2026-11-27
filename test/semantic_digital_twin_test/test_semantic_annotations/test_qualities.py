"""
Tests for the qualities of semantic annotations: ripeness, its stages and the fruits
that carry it.
"""

from __future__ import annotations

# %% Imports

from dataclasses import dataclass, field

import pytest
from random_events.interval import Bound, SimpleInterval

from semantic_digital_twin.exceptions import (
    NoFieldAcceptsEntity,
    SeveralFieldsAcceptEntity,
)
from semantic_digital_twin.semantic_annotations.qualities.exceptions import (
    QualityOutsideScale,
    RipenessInNoSingleStage,
)
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.qualities.mixins import HasRipeness
from semantic_digital_twin.semantic_annotations.qualities.relationship_marker import (
    IsQualityRelationship,
)
from semantic_digital_twin.semantic_annotations.qualities.quality import (
    Quality,
    Ripeness,
    RipenessStage,
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


@pytest.mark.parametrize("stage", list(RipenessStage))
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
    ripe_interval = RipenessStage.RIPE.value
    centre = (ripe_interval.lower + ripe_interval.upper) / 2
    ripeness = Ripeness(
        value=SimpleInterval.from_data(
            lower=centre, upper=centre, left=Bound.CLOSED, right=Bound.CLOSED
        )
    )

    assert ripeness.stage == RipenessStage.RIPE


def test_ripeness_straddling_two_stages_has_no_single_stage() -> None:
    """
    A ripeness reaching into two stages raises instead of picking one of them.
    """
    ripeness = Ripeness(
        value=SimpleInterval.from_data(
            lower=RipenessStage.UNRIPE.value.lower,
            upper=RipenessStage.RIPE.value.upper,
            left=Bound.CLOSED,
            right=Bound.OPEN,
        )
    )

    with pytest.raises(RipenessInNoSingleStage):
        ripeness.stage


def test_ripeness_beyond_the_scale_is_rejected() -> None:
    """
    A ripeness value beyond the ripeness scale is refused when it is made.
    """
    with pytest.raises(QualityOutsideScale):
        Ripeness(
            value=SimpleInterval.from_data(
                lower=Ripeness.scale.upper,
                upper=Ripeness.scale.upper + 1,
                left=Bound.OPEN,
                right=Bound.CLOSED,
            )
        )


def test_stages_cover_the_ripeness_scale_without_overlap() -> None:
    """
    The stages divide the whole ripeness scale between them, without overlapping.
    """
    stage_intervals = [stage.value.as_composite_set() for stage in RipenessStage]
    covered = stage_intervals[0]
    for stage_interval in stage_intervals[1:]:
        assert covered.intersection_with(other=stage_interval).is_empty()
        covered = covered.union_with(other=stage_interval)

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


def test_fruit_without_a_ripeness_has_only_its_default_qualities(world: World) -> None:
    """
    A fruit whose ripeness is unknown lists only the qualities its kind starts with.
    """
    with world.modify_world():
        apple = Apple.create_with_new_body_in_world(name="apple", world=world)

    assert apple.qualities == [apple.mass_density, apple.material_elasticity]


def test_ripeness_of_a_fruit_is_its_quality(world: World) -> None:
    """
    Setting a ripeness on a fruit fills its ripeness field and lists it among its qualities.
    """
    ripeness = Ripeness.from_stage(stage=RipenessStage.RIPE)
    with world.modify_world():
        apple = Apple.create_with_new_body_in_world(name="apple", world=world)
        apple.set_quality(quality=ripeness)

    assert apple.ripeness is ripeness
    assert apple.qualities == [
        ripeness,
        apple.mass_density,
        apple.material_elasticity,
    ]


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
        with pytest.raises(NoFieldAcceptsEntity):
            apple.set_quality(quality=Firmness())


def test_setting_a_quality_two_fields_accept_is_rejected(world: World) -> None:
    """
    Setting a quality that two fields of the annotation accept raises instead of picking one.
    """
    with world.modify_world():
        annotation = AnnotationWithTwoRipenessFields.create_with_new_body_in_world(
            name="annotation", world=world
        )
        with pytest.raises(SeveralFieldsAcceptEntity):
            annotation.set_quality(
                quality=Ripeness.from_stage(stage=RipenessStage.RIPE)
            )


def test_field_name_chooses_between_quality_fields_that_accept_a_quality(
    world: World,
) -> None:
    """
    Naming a field sets the quality there when several fields accept it.
    """
    ripeness = Ripeness.from_stage(stage=RipenessStage.RIPE)
    with world.modify_world():
        annotation = AnnotationWithTwoRipenessFields.create_with_new_body_in_world(
            name="annotation", world=world
        )
        annotation.set_quality(quality=ripeness, field_name="second_ripeness")

    assert annotation.second_ripeness is ripeness
    assert annotation.ripeness is None
