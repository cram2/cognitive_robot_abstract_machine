"""
Qualities of semantic annotations: aspects of a single bearer, such as the ripeness of a
fruit or the mass density of its flesh, that can change while the bearer stays the same.

Follows DUL and SOMA: a quality has a value in a region of its quality space, and named
regions of that space, such as :attr:`RipenessStage.RIPE`, classify the value.
"""

from __future__ import annotations

# %% Imports

from abc import ABC
from dataclasses import dataclass
from enum import Enum

from typing_extensions import Self

from krrood.patterns.classproperty import classproperty
from random_events.interval import Bound, SimpleInterval
from semantic_digital_twin.semantic_annotations.qualities.exceptions import (
    QualityOutsideScale,
    RipenessInNoSingleStage,
)

# %% Qualities


@dataclass(kw_only=True)
class Quality(ABC):
    """
    An aspect of exactly one bearer that cannot exist without it and is not a part of it,
    such as the ripeness of a fruit. Its value changes while the bearer stays the same.

    .. note:: Dispositions, such as being graspable, are not qualities here; they are
        mixins of the annotations that have them.
    """

    def _reject_value_outside_scale(
        self, *, value: SimpleInterval, scale: SimpleInterval
    ) -> None:
        """
        :param value: A value of this quality.
        :param scale: The scale ``value`` has to lie within.
        :raises QualityOutsideScale: If ``value`` reaches beyond ``scale``.
        """
        if scale.intersection_with(other=value) != value:
            raise QualityOutsideScale(quality=self, value=value, scale=scale)

    @staticmethod
    def closed_range(*, lower: float, upper: float) -> SimpleInterval:
        """
        :param lower: The smallest measured value.
        :param upper: The largest measured value, equal to ``lower`` for a single one.
        :return: The closed interval the measured values of a quality spread over.
        """
        return SimpleInterval.from_data(
            lower=lower, upper=upper, left=Bound.CLOSED, right=Bound.CLOSED
        )


# %% Ripeness


class RipenessStage(Enum):
    """
    The named stages of ripening, each the interval of :attr:`Ripeness.scale` it covers.

    .. note:: The boundaries between the stages are a convention of this model; no
        ontology or horticultural source fixes them.
    """

    UNRIPE = SimpleInterval.from_data(
        lower=0.0, upper=1 / 3, left=Bound.CLOSED, right=Bound.OPEN
    )
    """
    The first third of ripening, such as a green banana or a firm avocado.
    """

    RIPE = SimpleInterval.from_data(
        lower=1 / 3, upper=2 / 3, left=Bound.CLOSED, right=Bound.OPEN
    )
    """
    The middle third of ripening, in which a fruit is usually eaten.
    """

    OVERRIPE = SimpleInterval.from_data(
        lower=2 / 3, upper=1.0, left=Bound.CLOSED, right=Bound.CLOSED
    )
    """
    The last third of ripening, in which a fruit softens and starts to spoil.
    """


@dataclass(kw_only=True)
class Ripeness(Quality):
    """
    How far a fruit has ripened, on one :attr:`scale` shared by every fruit.

    Ripening is a gradient: each :class:`RipenessStage` is an interval of the scale, and
    :attr:`stage` classifies the value.
    """

    value: SimpleInterval
    """
    The position of the ripeness on :attr:`scale`: a single point when it was measured,
    or the interval of a :class:`RipenessStage` when only the stage is known.
    """

    def __post_init__(self) -> None:
        """
        :raises QualityOutsideScale: If the value reaches beyond :attr:`scale`.
        """
        self._reject_value_outside_scale(value=self.value, scale=self.scale)

    @classproperty
    def scale(cls) -> SimpleInterval:
        """
        :return: The ripeness scale, from 0 at the start of ripening to 1 at its end.
        """
        return SimpleInterval.from_data(
            lower=0.0, upper=1.0, left=Bound.CLOSED, right=Bound.CLOSED
        )

    @classmethod
    def from_stage(cls, *, stage: RipenessStage) -> Self:
        """
        :param stage: The stage the ripeness is known to lie in.
        :return: A ripeness whose value is the whole interval of ``stage``.
        """
        return cls(value=stage.value)

    @property
    def stage(self) -> RipenessStage:
        """
        :return: The stage whose interval holds the whole value.
        :raises RipenessInNoSingleStage: If no single stage holds the whole value.
        """
        for stage in RipenessStage:
            if stage.value.intersection_with(other=self.value) == self.value:
                return stage
        raise RipenessInNoSingleStage(ripeness=self)


# %% Physics


@dataclass(kw_only=True)
class PhysicsQuality(Quality, ABC):
    """
    A quality of its bearer's matter that physics acts on, such as its mass density or
    its elasticity, which a physics simulation of the bearer reads.

    .. note:: Not every quality is one: the ripeness of a fruit is not.
    """


# %% Material


@dataclass(kw_only=True)
class MassDensity(PhysicsQuality):
    """
    The mass per volume of its bearer's material, in kilograms per cubic metre
    :cite:`wikipedia_density`.
    """

    value: SimpleInterval
    """
    The mass density on :attr:`scale`: a single point when it was measured, or the range its
    measurements spread over.
    """

    def __post_init__(self) -> None:
        """
        :raises QualityOutsideScale: If the value reaches beyond :attr:`scale`.
        """
        self._reject_value_outside_scale(value=self.value, scale=self.scale)

    @classproperty
    def scale(cls) -> SimpleInterval:
        """
        :return: The mass density scale: every positive mass density.
        """
        return SimpleInterval.from_data(
            lower=0.0, upper=float("inf"), left=Bound.OPEN, right=Bound.OPEN
        )

    @classproperty
    def steel(cls) -> MassDensity:
        """
        :return: The mass density of steel, from a handbook compilation
            :cite:`engineeringtoolbox_density_solids`.
        """
        return cls(value=Quality.closed_range(lower=7820.0, upper=7820.0))

    @classproperty
    def glass(cls) -> MassDensity:
        """
        :return: The mass density of glass, from a handbook compilation
            :cite:`engineeringtoolbox_density_solids`.
        """
        return cls(value=Quality.closed_range(lower=2400.0, upper=2800.0))

    @classproperty
    def polyurethane_foam(cls) -> MassDensity:
        """
        :return: The mass density of polyurethane foam, from a handbook compilation
            :cite:`engineeringtoolbox_density_solids`.
        """
        return cls(value=Quality.closed_range(lower=30.0, upper=30.0))

    @classproperty
    def paper(cls) -> MassDensity:
        """
        :return: The mass density of paper, from a handbook compilation
            :cite:`engineeringtoolbox_density_solids`.
        """
        return cls(value=Quality.closed_range(lower=700.0, upper=1150.0))

    @classproperty
    def cardboard(cls) -> MassDensity:
        """
        :return: The mass density of cardboard, from a handbook compilation
            :cite:`engineeringtoolbox_density_solids`.
        """
        return cls(value=Quality.closed_range(lower=700.0, upper=700.0))


@dataclass(kw_only=True)
class MaterialElasticity(PhysicsQuality):
    """
    How its bearer's material deforms under load and springs back, given by the two
    constants that describe an isotropic linear elastic material
    :cite:`wikipedia_linear_elasticity`.

    .. note:: Plant tissue is neither isotropic nor linear; the constants approximate it.

    .. seealso:: :cite:t:`mohsenin1971physical`, §4.5 on ideal elastic behaviour, §4.9
        on viscoelasticity and §5.4 on Poisson's ratio.
    """

    youngs_modulus: SimpleInterval
    """
    The stiffness of the material when stretched or compressed along one axis, in pascals,
    on :attr:`youngs_modulus_scale`: a single point when measured, or a range
    :cite:`wikipedia_youngs_modulus`.
    """

    poissons_ratio: SimpleInterval | None = None
    """
    How much the material narrows across a stretch relative to how much it lengthens, on
    :attr:`poissons_ratio_scale`: a single point when measured, a range, or ``None`` when
    it was not measured :cite:`wikipedia_poissons_ratio`.
    """

    def __post_init__(self) -> None:
        """
        :raises QualityOutsideScale: If a constant reaches beyond its scale.
        """
        self._reject_value_outside_scale(
            value=self.youngs_modulus, scale=self.youngs_modulus_scale
        )
        if self.poissons_ratio is None:
            return
        self._reject_value_outside_scale(
            value=self.poissons_ratio, scale=self.poissons_ratio_scale
        )

    @classproperty
    def youngs_modulus_scale(cls) -> SimpleInterval:
        """
        :return: The scale of Young's modulus: every positive stiffness.
        """
        return SimpleInterval.from_data(
            lower=0.0, upper=float("inf"), left=Bound.OPEN, right=Bound.OPEN
        )

    @classproperty
    def poissons_ratio_scale(cls) -> SimpleInterval:
        """
        :return: The scale of Poisson's ratio: the range, from -1 to 1/2, in which an
            isotropic linear elastic material is stable :cite:`mott2009limits`.
        """
        return SimpleInterval.from_data(
            lower=-1.0, upper=0.5, left=Bound.OPEN, right=Bound.OPEN
        )

    @classproperty
    def steel(cls) -> MaterialElasticity:
        """
        :return: The elastic constants of steel, from handbook compilations; the Poisson's
            ratio is that of 18-8 stainless steel
            :cite:`wikipedia_youngs_modulus,engineeringtoolbox_poissons_ratio`.
        """
        return cls(
            youngs_modulus=Quality.closed_range(lower=200e9, upper=200e9),
            poissons_ratio=Quality.closed_range(lower=0.305, upper=0.305),
        )

    @classproperty
    def glass(cls) -> MaterialElasticity:
        """
        :return: The elastic constants of glass, from handbook compilations: the Young's
            modulus of float glass and the Poisson's ratio of soda glass
            :cite:`wikipedia_youngs_modulus,engineeringtoolbox_poissons_ratio`.
        """
        return cls(
            youngs_modulus=Quality.closed_range(lower=47.7e9, upper=83.6e9),
            poissons_ratio=Quality.closed_range(lower=0.22, upper=0.22),
        )


# %% Outer coverings


@dataclass(kw_only=True)
class SkinThickness(PhysicsQuality):
    """
    How thick its bearer, the outer covering of an object such as the peel of a fruit or
    the shell of an egg, is across its surface, in metres.
    """

    value: SimpleInterval
    """
    The thickness on :attr:`scale`: a single point when it was measured, or the range its
    measurements spread over.
    """

    def __post_init__(self) -> None:
        """
        :raises QualityOutsideScale: If the value reaches beyond :attr:`scale`.
        """
        self._reject_value_outside_scale(value=self.value, scale=self.scale)

    @classproperty
    def scale(cls) -> SimpleInterval:
        """
        :return: The thickness scale: every positive thickness.
        """
        return SimpleInterval.from_data(
            lower=0.0, upper=float("inf"), left=Bound.OPEN, right=Bound.OPEN
        )
