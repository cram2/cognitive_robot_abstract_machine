"""
Tests for the material qualities of fruit tissues: their density and elasticity.
"""

from __future__ import annotations

# %% Imports

import pytest
from random_events.interval import Bound, SimpleInterval

from semantic_digital_twin.semantic_annotations.qualities.exceptions import (
    QualityOutsideScale,
)
from semantic_digital_twin.semantic_annotations.qualities.mixins import (
    HasDensity,
    HasElasticity,
)
from semantic_digital_twin.semantic_annotations.qualities.quality import (
    Density,
    Elasticity,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    CitrusSegment,
    Core,
    Flesh,
    FruitPart,
    Peel,
    Pit,
    Pith,
    Seed,
    SeedChamber,
    Stem,
)
from semantic_digital_twin.world import World

# %% Fixtures


@pytest.fixture
def world() -> World:
    """
    An empty world with a root body to spawn fruit parts into.
    """
    return World.create_with_root_body("root")


def point(value: float) -> SimpleInterval:
    """
    :param value: A measured value.
    :return: The interval holding only ``value``.
    """
    return SimpleInterval.from_data(value, value, Bound.CLOSED, Bound.CLOSED)


def measured_elasticity() -> Elasticity:
    """
    :return: An elasticity measured at a point within both of its scales.
    """
    return Elasticity(youngs_modulus=point(3.0e6), poissons_ratio=point(0.33))


# %% Scales


def test_density_at_the_lower_end_of_its_scale_is_rejected() -> None:
    """
    A density of zero, the open lower end of the density scale, is refused.
    """
    with pytest.raises(QualityOutsideScale):
        Density(value=point(Density.scale.lower))


def test_youngs_modulus_at_the_lower_end_of_its_scale_is_rejected() -> None:
    """
    A Young's modulus of zero, the open lower end of its scale, is refused.
    """
    with pytest.raises(QualityOutsideScale):
        Elasticity(
            youngs_modulus=point(Elasticity.youngs_modulus_scale.lower),
            poissons_ratio=measured_elasticity().poissons_ratio,
        )


def test_poissons_ratio_at_the_upper_end_of_its_scale_is_rejected() -> None:
    """
    A Poisson's ratio of one half, the open upper end of its scale, is refused.
    """
    with pytest.raises(QualityOutsideScale):
        Elasticity(
            youngs_modulus=measured_elasticity().youngs_modulus,
            poissons_ratio=point(Elasticity.poissons_ratio_scale.upper),
        )


# %% Material qualities of fruit parts


@pytest.mark.parametrize(
    "part_type",
    [Peel, Flesh, CitrusSegment, Core, Seed, Stem, Pit, SeedChamber, Pith],
)
def test_every_fruit_part_starts_without_density_and_elasticity(
    world: World, part_type: type[FruitPart]
) -> None:
    """
    Every fruit part has a density and an elasticity field, empty until they are set.
    """
    with world.modify_world():
        part = part_type.create_with_new_body_in_world(name="part", world=world)

    assert isinstance(part, HasElasticity)
    assert isinstance(part, HasDensity)
    assert part.density is None
    assert part.elasticity is None


def test_density_and_elasticity_of_a_peel_are_its_qualities(world: World) -> None:
    """
    Setting a density and an elasticity on a peel fills their fields and lists both
    among its qualities.
    """
    density = Density(value=point(850.0))
    elasticity = measured_elasticity()
    with world.modify_world():
        peel = Peel.create_with_new_body_in_world(name="peel", world=world)
        peel.set_quality(quality=density)
        peel.set_quality(quality=elasticity)

    assert peel.density is density
    assert peel.elasticity is elasticity
    assert peel.qualities == [density, elasticity]
