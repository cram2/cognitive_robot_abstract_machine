"""
Mixins for each kind of quality an annotation can have, such as the ripeness of a fruit
or the mass density of its flesh.
"""

from __future__ import annotations

# %% Imports

from dataclasses import dataclass, field

from krrood.patterns.classproperty import classproperty
from krrood.patterns.mixin import Mixin
from semantic_digital_twin.semantic_annotations.qualities.relationship_marker import (
    IsQualityRelationship,
)
from semantic_digital_twin.semantic_annotations.qualities.quality import (
    MassDensity,
    MaterialElasticity,
    Quality,
    Ripeness,
    SkinThickness,
)
from semantic_digital_twin.semantic_annotations.qualities.relationship import (
    HasQualities,
)

# %% Qualities of food


@dataclass(eq=False, kw_only=True)
class HasRipeness(HasQualities, Mixin):
    """
    A mixin class for semantic annotations that ripen.
    """

    ripeness: Ripeness | None = field(
        default=None,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    How far the semantic annotation has ripened, or ``None`` when that is unknown.
    """


# %% Material qualities


@dataclass(eq=False, kw_only=True)
class HasMassDensity(HasQualities, Mixin):
    """
    A mixin class for semantic annotations made of a material with a mass density.
    """

    mass_density: MassDensity | None = field(
        default=None,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    The mass density of the semantic annotation's material, or ``None`` when that is unknown.
    """


@dataclass(eq=False, kw_only=True)
class HasMaterialElasticity(HasMassDensity, Mixin):
    """
    A mixin class for semantic annotations made of an elastic material, which, being
    matter, also has a mass density.
    """

    material_elasticity: MaterialElasticity | None = field(
        default=None,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    The elasticity of the semantic annotation's material, or ``None`` when that is
    unknown.
    """


# %% Qualities of outer coverings


@dataclass(eq=False, kw_only=True)
class HasSkinThickness(HasQualities, Mixin):
    """
    A mixin class for semantic annotations that are the outer covering of an object,
    such as the peel of a fruit or the shell of an egg.
    """

    skin_thickness: SkinThickness | None = field(
        default=None,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    How thick the semantic annotation is, or ``None`` when that is unknown.
    """


# %% Materials


@dataclass(eq=False, kw_only=True)
class MadeOfSteel(HasQualities, Mixin):
    """
    A mixin class for semantic annotations made of steel, such as cutlery.
    """

    @classproperty
    def default_qualities(cls) -> list[Quality]:
        """
        :return: The qualities of steel from handbook compilations; the Poisson's ratio
            is that of 18-8 stainless steel
            :cite:`engineeringtoolbox_density_solids,wikipedia_youngs_modulus,engineeringtoolbox_poissons_ratio`.
        """
        return [
            MassDensity(value=Quality.closed_range(lower=7820.0, upper=7820.0)),
            MaterialElasticity(
                youngs_modulus=Quality.closed_range(lower=200e9, upper=200e9),
                poissons_ratio=Quality.closed_range(lower=0.305, upper=0.305),
            ),
        ]
