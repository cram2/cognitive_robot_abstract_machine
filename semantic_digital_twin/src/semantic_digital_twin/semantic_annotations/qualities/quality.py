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
        if scale.intersection_with(value) != value:
            raise QualityOutsideScale(quality=self, value=value, scale=scale)


# %% Ripeness


class RipenessStage(Enum):
    """
    The named stages of ripening, each the interval of :attr:`Ripeness.scale` it covers.

    .. note:: The boundaries between the stages are a convention of this model; no
        ontology or horticultural source fixes them.
    """

    UNRIPE = SimpleInterval.from_data(0.0, 1 / 3, Bound.CLOSED, Bound.OPEN)
    """
    The first third of ripening, such as a green banana or a firm avocado.
    """

    RIPE = SimpleInterval.from_data(1 / 3, 2 / 3, Bound.CLOSED, Bound.OPEN)
    """
    The middle third of ripening, in which a fruit is usually eaten.
    """

    OVERRIPE = SimpleInterval.from_data(2 / 3, 1.0, Bound.CLOSED, Bound.CLOSED)
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
        return SimpleInterval.from_data(0.0, 1.0, Bound.CLOSED, Bound.CLOSED)

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
            if stage.value.intersection_with(self.value) == self.value:
                return stage
        raise RipenessInNoSingleStage(ripeness=self)


# %% Material


@dataclass(kw_only=True)
class MassDensity(Quality):
    """
    The mass per volume of its bearer's material, in kilograms per cubic metre.

    .. seealso:: `Density <https://en.wikipedia.org/wiki/Density>`_ (Wikipedia)
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
        return SimpleInterval.from_data(0.0, float("inf"), Bound.OPEN, Bound.OPEN)


@dataclass(kw_only=True)
class MaterialElasticity(Quality):
    """
    How its bearer's material deforms under load and springs back, given by the two
    constants that describe an isotropic linear elastic material.

    .. note:: Plant tissue is neither isotropic nor linear; the constants approximate it.

    .. seealso::
        - `Linear elasticity <https://en.wikipedia.org/wiki/Linear_elasticity>`_
          (Wikipedia)
        - N. N. Mohsenin, `Physical Properties of Plant and Animal Materials, vol. 1
          <https://www.routledge.com/Physical-Properties-of-Plant-and-Animal-Materials-v-1-Physical-Characteristics-and-Mechanical-Properties/Mohsenin/p/book/9780677023007>`_,
          ISBN 978-0-677-02300-7: §4.5 on ideal elastic behaviour, §4.9 on
          viscoelasticity, §5.4 on Poisson's ratio
    """

    youngs_modulus: SimpleInterval
    """
    The stiffness of the material when stretched or compressed along one axis, in pascals,
    on :attr:`youngs_modulus_scale`: a single point when measured, or a range.

    .. seealso:: `Young's modulus <https://en.wikipedia.org/wiki/Young%27s_modulus>`_
        (Wikipedia)
    """

    poissons_ratio: SimpleInterval
    """
    How much the material narrows across a stretch relative to how much it lengthens, on
    :attr:`poissons_ratio_scale`: a single point when measured, or a range.

    .. seealso:: `Poisson's ratio <https://en.wikipedia.org/wiki/Poisson%27s_ratio>`_
        (Wikipedia)
    """

    def __post_init__(self) -> None:
        """
        :raises QualityOutsideScale: If a constant reaches beyond its scale.
        """
        self._reject_value_outside_scale(
            value=self.youngs_modulus, scale=self.youngs_modulus_scale
        )
        self._reject_value_outside_scale(
            value=self.poissons_ratio, scale=self.poissons_ratio_scale
        )

    @classproperty
    def youngs_modulus_scale(cls) -> SimpleInterval:
        """
        :return: The scale of Young's modulus: every positive stiffness.
        """
        return SimpleInterval.from_data(0.0, float("inf"), Bound.OPEN, Bound.OPEN)

    @classproperty
    def poissons_ratio_scale(cls) -> SimpleInterval:
        """
        :return: The scale of Poisson's ratio: the range, from -1 to 1/2, in which an
            isotropic linear elastic material is stable.

        .. seealso:: P. H. Mott and C. M. Roland, `Limits to Poisson's ratio in isotropic
            materials <https://doi.org/10.1103/PhysRevB.80.132104>`_, Physical Review B
            80, 132104
        """
        return SimpleInterval.from_data(-1.0, 0.5, Bound.OPEN, Bound.OPEN)
