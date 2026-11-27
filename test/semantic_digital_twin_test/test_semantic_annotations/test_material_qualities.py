"""
Tests for the material qualities of annotations with a root body, such as fruit tissues:

their mass density and elasticity.
"""

from __future__ import annotations

# %% Imports

import pytest
from random_events.interval import Bound, SimpleInterval

from semantic_digital_twin.semantic_annotations.qualities.exceptions import (
    QualityOutsideScale,
)
from semantic_digital_twin.semantic_annotations.qualities.mixins import (
    HasMassDensity,
    HasMaterialElasticity,
)
from semantic_digital_twin.semantic_annotations.qualities.quality import (
    MassDensity,
    MaterialElasticity,
)
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
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
    Table,
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
    return SimpleInterval.from_data(
        lower=value, upper=value, left=Bound.CLOSED, right=Bound.CLOSED
    )


def measured_material_elasticity() -> MaterialElasticity:
    """
    :return: An elasticity measured at a point within both of its scales.
    """
    return MaterialElasticity(youngs_modulus=point(3.0e6), poissons_ratio=point(0.33))


# %% Scales


def test_density_at_the_lower_end_of_its_scale_is_rejected() -> None:
    """
    A density of zero, the open lower end of the density scale, is refused.
    """
    with pytest.raises(QualityOutsideScale):
        MassDensity(value=point(MassDensity.scale.lower))


def test_youngs_modulus_at_the_lower_end_of_its_scale_is_rejected() -> None:
    """
    A Young's modulus of zero, the open lower end of its scale, is refused.
    """
    with pytest.raises(QualityOutsideScale):
        MaterialElasticity(
            youngs_modulus=point(MaterialElasticity.youngs_modulus_scale.lower),
            poissons_ratio=measured_material_elasticity().poissons_ratio,
        )


def test_poissons_ratio_at_the_upper_end_of_its_scale_is_rejected() -> None:
    """
    A Poisson's ratio of one half, the open upper end of its scale, is refused.
    """
    with pytest.raises(QualityOutsideScale):
        MaterialElasticity(
            youngs_modulus=measured_material_elasticity().youngs_modulus,
            poissons_ratio=point(MaterialElasticity.poissons_ratio_scale.upper),
        )


# %% Material qualities of annotations with a root body


def test_every_annotation_with_a_root_body_has_material_qualities() -> None:
    """
    An annotation with a root body is made of a material, so it has a mass density and
    an elasticity, whatever kind of annotation it is.
    """
    assert issubclass(HasRootBody, HasMaterialElasticity)
    assert issubclass(HasRootBody, HasMassDensity)


def test_furniture_starts_without_mass_density_and_elasticity(world: World) -> None:
    """
    A table, no fruit part, has mass density and elasticity fields too, empty until set.
    """
    with world.modify_world():
        table = Table.create_with_new_body_in_world(name="table", world=world)

    assert table.mass_density is None
    assert table.material_elasticity is None


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

    assert isinstance(part, HasMaterialElasticity)
    assert isinstance(part, HasMassDensity)
    assert part.mass_density is None
    assert part.material_elasticity is None


def test_density_and_elasticity_of_a_peel_are_its_qualities(world: World) -> None:
    """
    Setting a density and an elasticity on a peel fills their fields and lists both
    among its qualities.
    """
    mass_density = MassDensity(value=point(850.0))
    material_elasticity = measured_material_elasticity()
    with world.modify_world():
        peel = Peel.create_with_new_body_in_world(name="peel", world=world)
        peel.set_quality(quality=mass_density)
        peel.set_quality(quality=material_elasticity)

    assert peel.mass_density is mass_density
    assert peel.material_elasticity is material_elasticity
    assert peel.qualities == [mass_density, material_elasticity]
