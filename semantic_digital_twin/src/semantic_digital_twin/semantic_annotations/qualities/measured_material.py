"""
Mass densities, elasticities and thicknesses of the materials semantic annotations are
made of, as published measurements report them, each with its source.
"""

from __future__ import annotations

# %% Imports

from dataclasses import dataclass
from enum import Enum

from random_events.interval import Bound, SimpleInterval
from semantic_digital_twin.semantic_annotations.qualities.quality import (
    MassDensity,
    MaterialElasticity,
    Quality,
    LayerThickness,
)

# %% A measurement


@dataclass(frozen=True, kw_only=True)
class MeasuredRange:
    """
    The closed range the reported values of one quantity spread over, in SI units; a
    single reported value is a range whose ends are equal.
    """

    lower: float
    """
    The smallest reported value.
    """

    upper: float
    """
    The largest reported value.
    """

    def to_interval(self) -> SimpleInterval:
        """
        :return: The closed interval from :attr:`lower` to :attr:`upper`.
        """
        return SimpleInterval.from_data(
            lower=self.lower, upper=self.upper, left=Bound.CLOSED, right=Bound.CLOSED
        )


@dataclass(frozen=True, kw_only=True)
class MeasuredElasticity:
    """
    The elastic constants measured for a material: always a Young's modulus, and a
    Poisson's ratio only where one was measured too.
    """

    youngs_modulus: MeasuredRange
    """
    The Young's modulus, in pascals.
    """

    poissons_ratio: MeasuredRange | None = None
    """
    The Poisson's ratio, or ``None`` when it was not measured.
    """


@dataclass(frozen=True, kw_only=True)
class MaterialMeasurement:
    """
    What the literature reports about one material; a quantity no source reports is
    ``None``.
    """

    mass_density: MeasuredRange | None = None
    """
    The mass density, in kilograms per cubic metre.
    """

    material_elasticity: MeasuredElasticity | None = None
    """
    The elastic constants.
    """

    layer_thickness: MeasuredRange | None = None
    """
    The thickness of a layer of the material, in metres.
    """

    def create_qualities(self) -> list[Quality]:
        """
        :return: A new quality for each quantity that was measured, so that no two
            annotations share one.
        """
        qualities: list[Quality] = []
        if self.mass_density is not None:
            qualities.append(MassDensity(value=self.mass_density.to_interval()))
        if self.material_elasticity is not None:
            poissons_ratio = self.material_elasticity.poissons_ratio
            qualities.append(
                MaterialElasticity(
                    youngs_modulus=self.material_elasticity.youngs_modulus.to_interval(),
                    poissons_ratio=(
                        None if poissons_ratio is None else poissons_ratio.to_interval()
                    ),
                )
            )
        if self.layer_thickness is not None:
            qualities.append(LayerThickness(value=self.layer_thickness.to_interval()))
        return qualities


# %% The measured materials


class MeasuredMaterial(Enum):
    """
    The materials of semantic annotations whose qualities were measured, each with what
    the cited sources report.
    """

    # %% Fruits

    APPLE = MaterialMeasurement(
        mass_density=MeasuredRange(lower=929.0, upper=946.0),
        material_elasticity=MeasuredElasticity(
            youngs_modulus=MeasuredRange(lower=1.63e6, upper=3.94e6),
            poissons_ratio=MeasuredRange(lower=0.173, upper=0.221),
        ),
    )
    """
    Whole apples: the density of three early cultivars :cite:`ozturk2011apple`, and the
    elastic constants of Golden Delicious fruit from acoustic and puncture tests
    :cite:`grotte2002golden`.
    """

    APPLE_PEEL = MaterialMeasurement(
        material_elasticity=MeasuredElasticity(
            youngs_modulus=MeasuredRange(lower=17.79e6, upper=24.00e6)
        ),
        layer_thickness=MeasuredRange(lower=33.3e-6, upper=78.3e-6),
    )
    """
    Apple skin: tensile strips of Fuji and Starkrimson peel in both directions
    :cite:`wang2015apple`, and the histological skin thickness across cultivars
    :cite:`homutova2006apple`.
    """

    APPLE_FLESH = MaterialMeasurement(
        material_elasticity=MeasuredElasticity(
            youngs_modulus=MeasuredRange(lower=2.7e6, upper=4.6e6)
        ),
    )
    """
    Apple flesh in quasi-static compression; under impact it measures about 9 MPa
    :cite:`stropek2022apple`.
    """

    AVOCADO_PEEL = MaterialMeasurement(
        layer_thickness=MeasuredRange(lower=1.6e-3, upper=1.6e-3),
    )
    """
    The skin of Hass avocados, from a plant-patent comparison of cultivars
    :cite:`uspp14239avocado`.
    """

    AVOCADO_FLESH = MaterialMeasurement(
        material_elasticity=MeasuredElasticity(
            youngs_modulus=MeasuredRange(lower=0.16e6, upper=2.29e6)
        ),
    )
    """
    Slices of Hass avocado flesh, from 2.29 MPa unripe down to 0.16 MPa after five days
    of ripening :cite:`farinas2021avocado`.
    """

    BANANA_PEEL = MaterialMeasurement(
        layer_thickness=MeasuredRange(lower=2.95e-3, upper=3.65e-3),
    )
    """
    The peel of green Nendran and Dwarf Cavendish bananas, measured with calipers
    :cite:`kachru1995banana`.
    """

    BANANA_FLESH = MaterialMeasurement(
        material_elasticity=MeasuredElasticity(
            youngs_modulus=MeasuredRange(lower=1.09e6, upper=1.75e6)
        ),
    )
    """
    Cylinders of banana pulp in compression, from day 1 to day 12 after harvest
    :cite:`ciro2005banana`.
    """

    CUCUMBER = MaterialMeasurement(
        mass_density=MeasuredRange(lower=980.0, upper=980.0),
    )
    """
    The specific gravity of whole Kurume-H cucumbers :cite:`kojima1975cucumber`.
    """

    CUCUMBER_FLESH = MaterialMeasurement(
        material_elasticity=MeasuredElasticity(
            youngs_modulus=MeasuredRange(lower=2.844e6, upper=2.844e6)
        ),
    )
    """
    Cucumber flesh in compression with skin and seeds removed, 29 kgf/cm²
    :cite:`kojima1975cucumber`.
    """

    ORANGE = MaterialMeasurement(
        mass_density=MeasuredRange(lower=999.0, upper=1046.0),
        material_elasticity=MeasuredElasticity(
            youngs_modulus=MeasuredRange(lower=0.584e6, upper=0.584e6),
            poissons_ratio=MeasuredRange(lower=0.24, upper=0.24),
        ),
    )
    """
    Whole oranges: the density from large to small grades :cite:`sharifi2007orange`, and
    the elastic constants in whole-fruit compression :cite:`namdari2018orange`.
    """

    ORANGE_PEEL = MaterialMeasurement(
        mass_density=MeasuredRange(lower=746.0, upper=746.0),
        material_elasticity=MeasuredElasticity(
            youngs_modulus=MeasuredRange(lower=1.985e4, upper=1.985e4),
            poissons_ratio=MeasuredRange(lower=-0.05, upper=-0.05),
        ),
        layer_thickness=MeasuredRange(lower=5.4e-3, upper=7.0e-3),
    )
    """
    Orange peel in quasi-static compression with digital image correlation; the
    thickness is the mean 6.2 mm within one standard deviation :cite:`jentzsch2022citrus`.
    """

    PEACH = MaterialMeasurement(
        material_elasticity=MeasuredElasticity(
            youngs_modulus=MeasuredRange(lower=0.89e6, upper=0.89e6),
            poissons_ratio=MeasuredRange(lower=0.29, upper=0.29),
        ),
    )
    """
    Halved peaches in compression, for a drop simulation :cite:`kabas2015peach`.
    """

    PEACH_PEEL = MaterialMeasurement(
        material_elasticity=MeasuredElasticity(
            youngs_modulus=MeasuredRange(lower=1.08e6, upper=1.08e6)
        ),
    )
    """
    Tensile strips of Red Haven peach skin; the strips include flesh, so their thickness
    is no peel thickness :cite:`cermak2011peach`.
    """

    STRAWBERRY_FLESH = MaterialMeasurement(
        material_elasticity=MeasuredElasticity(
            youngs_modulus=MeasuredRange(lower=0.274e6, upper=0.614e6)
        ),
    )
    """
    The cortex of strawberries in uniaxial compression from 35 °C down to 1 °C
    :cite:`an2022strawberry`.
    """

    STRAWBERRY_PITH = MaterialMeasurement(
        material_elasticity=MeasuredElasticity(
            youngs_modulus=MeasuredRange(lower=0.333e6, upper=0.695e6)
        ),
    )
    """
    The pith of strawberries, in the same compression tests :cite:`an2022strawberry`.
    """

    TOMATO_PEEL = MaterialMeasurement(
        material_elasticity=MeasuredElasticity(
            youngs_modulus=MeasuredRange(lower=2.297e6, upper=9.59e6)
        ),
    )
    """
    Tensile strips of tomato skin :cite:`li2012tomato,kunabroniowska2011tomato`; its
    measured Poisson's ratio of 0.38 to 0.73 lies beyond the scale and is left out.
    """

    TOMATO_FLESH = MaterialMeasurement(
        material_elasticity=MeasuredElasticity(
            youngs_modulus=MeasuredRange(lower=0.726e6, upper=0.846e6)
        ),
    )
    """
    Tomato mesocarp in compression :cite:`li2012tomato`.
    """

    TOMATO_SEED_CHAMBER = MaterialMeasurement(
        material_elasticity=MeasuredElasticity(
            youngs_modulus=MeasuredRange(lower=0.048e6, upper=0.124e6)
        ),
    )
    """
    The locular gel around tomato seeds in compression :cite:`li2012tomato`.
    """

    WATERMELON = MaterialMeasurement(
        mass_density=MeasuredRange(lower=927.84, upper=973.57),
    )
    """
    Whole watermelons across storage conditions :cite:`khater2016watermelon`.
    """

    WATERMELON_PEEL = MaterialMeasurement(
        material_elasticity=MeasuredElasticity(
            youngs_modulus=MeasuredRange(lower=4.937e6, upper=5.357e6)
        ),
        layer_thickness=MeasuredRange(lower=12.11e-3, upper=20.28e-3),
    )
    """
    The green outer rind of Crimson Sweet and Charleston Gray watermelons
    :cite:`sadrnia2008watermelon`, and the thickness of the whole rind
    :cite:`khater2016watermelon`.
    """

    WATERMELON_FLESH = MaterialMeasurement(
        material_elasticity=MeasuredElasticity(
            youngs_modulus=MeasuredRange(lower=0.396e6, upper=0.536e6)
        ),
    )
    """
    Red watermelon flesh in compression :cite:`sadrnia2008watermelon`.
    """

    # %% Vegetables

    POTATO = MaterialMeasurement(
        mass_density=MeasuredRange(lower=1080.0, upper=1095.0),
        material_elasticity=MeasuredElasticity(
            youngs_modulus=MeasuredRange(lower=2.17e6, upper=5.18e6),
            poissons_ratio=MeasuredRange(lower=0.406, upper=0.441),
        ),
    )
    """
    Potato flesh: the specific gravity targeted for processing cultivars
    :cite:`wang2017potato`, cylinders in axial compression :cite:`solomon2006potato`, and a
    Poisson's ratio back-calculated with a finite element model :cite:`celik2026potato`.
    """

    CARROT = MaterialMeasurement(
        material_elasticity=MeasuredElasticity(
            youngs_modulus=MeasuredRange(lower=3.0e6, upper=8.31e6),
            poissons_ratio=MeasuredRange(lower=0.35, upper=0.40),
        ),
    )
    """
    Raw carrot root in compression: a Japanese carrot before boiling
    :cite:`ogawa2015carrot`, and Chantenay and Nairobi along the root
    :cite:`morrell2019carrot`.
    """

    LETTUCE = MaterialMeasurement(
        material_elasticity=MeasuredElasticity(
            youngs_modulus=MeasuredRange(lower=2.23e6, upper=5.08e6)
        ),
    )
    """
    Tensile strips of leaves of var. Crispa lettuce :cite:`xavier2022lettuce`.
    """

    # %% Food with contents

    MILK = MaterialMeasurement(
        mass_density=MeasuredRange(lower=1027.0, upper=1033.0),
    )
    """
    Milk at 20 °C, which dominates a full container of it; being liquid, it has no
    elasticity :cite:`goff_milk_physical`.
    """

    BREAD = MaterialMeasurement(
        mass_density=MeasuredRange(lower=130.0, upper=340.0),
    )
    """
    The apparent density of the crumb of five commercial breads
    :cite:`nussinovitch1990bread`.
    """

    # %% Eggs

    EGG = MaterialMeasurement(
        mass_density=MeasuredRange(lower=1085.0, upper=1085.0),
    )
    """
    Whole fresh hen's eggs, weighed in air and in water; the density falls as an egg
    ages and loses mass :cite:`tsarenko2017egg`.
    """

    EGGSHELL = MaterialMeasurement(
        material_elasticity=MeasuredElasticity(
            youngs_modulus=MeasuredRange(lower=19.0e9, upper=19.0e9)
        ),
        layer_thickness=MeasuredRange(lower=0.403e-3, upper=0.444e-3),
    )
    """
    Eggshell: full rings in compression, around the egg :cite:`lin1995eggshell`, and the
    shell thickness of four hen genotypes :cite:`hammershoj2021egg`.
    """

    ALBUMEN = MaterialMeasurement(
        mass_density=MeasuredRange(lower=1027.5, upper=1030.9),
    )
    """
    Egg white from 25 °C down to 8 °C; being liquid, it has no elasticity
    :cite:`tsarenko2017egg`.
    """

    YOLK = MaterialMeasurement(
        mass_density=MeasuredRange(lower=1026.7, upper=1031.6),
    )
    """
    Egg yolk from 25 °C down to 8 °C; being liquid, it has no elasticity
    :cite:`tsarenko2017egg`.
    """

    # %% Household objects

    STEEL = MaterialMeasurement(
        mass_density=MeasuredRange(lower=7820.0, upper=7820.0),
        material_elasticity=MeasuredElasticity(
            youngs_modulus=MeasuredRange(lower=200e9, upper=200e9),
            poissons_ratio=MeasuredRange(lower=0.305, upper=0.305),
        ),
    )
    """
    Steel, from handbook compilations; the Poisson's ratio is that of 18-8 stainless
    steel :cite:`engineeringtoolbox_density_solids,wikipedia_youngs_modulus,engineeringtoolbox_poissons_ratio`.
    """

    GLASS = MaterialMeasurement(
        mass_density=MeasuredRange(lower=2400.0, upper=2800.0),
        material_elasticity=MeasuredElasticity(
            youngs_modulus=MeasuredRange(lower=47.7e9, upper=83.6e9),
            poissons_ratio=MeasuredRange(lower=0.22, upper=0.22),
        ),
    )
    """
    Glass, from handbook compilations: the Young's modulus of float glass and the
    Poisson's ratio of soda glass
    :cite:`engineeringtoolbox_density_solids,wikipedia_youngs_modulus,engineeringtoolbox_poissons_ratio`.
    """

    POLYURETHANE_FOAM = MaterialMeasurement(
        mass_density=MeasuredRange(lower=30.0, upper=30.0),
    )
    """
    Polyurethane foam, from a handbook compilation
    :cite:`engineeringtoolbox_density_solids`.
    """

    PAPER = MaterialMeasurement(
        mass_density=MeasuredRange(lower=700.0, upper=1150.0),
    )
    """
    Paper, from a handbook compilation :cite:`engineeringtoolbox_density_solids`.
    """

    CARDBOARD = MaterialMeasurement(
        mass_density=MeasuredRange(lower=700.0, upper=700.0),
    )
    """
    Cardboard, from a handbook compilation :cite:`engineeringtoolbox_density_solids`.
    """
