"""
Qualities of semantic annotations: aspects of a single bearer, such as the ripeness of a
fruit, whose value lies on a scale and can change while the bearer stays the same.

Follows DUL and SOMA: a quality has a value in a region of its quality space, and named
regions of that space, such as :class:`Ripe`, classify the value.
"""

from __future__ import annotations

# %% Imports

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import Self

from krrood.patterns.classproperty import classproperty
from krrood.patterns.field_metadata import FieldMetadata
from krrood.utils import recursive_subclasses
from random_events.interval import Bound, SimpleInterval
from semantic_digital_twin.semantic_annotations.exceptions import (
    RipenessOutsideScale,
    RipenessSpansSeveralStages,
)

# %% Relation metadata


@dataclass(kw_only=True)
class IsQualityRelationship(FieldMetadata):
    """
    Marks a field as holding a :class:`Quality` of its owner.

    The relation is signalled by the presence of an instance of this class in the
    field's ``metadata`` mapping (attach it with :meth:`~FieldMetadata.as_dict`).
    """


# %% Qualities


@dataclass(kw_only=True)
class Quality(ABC):
    """
    An aspect of exactly one bearer that cannot exist without it and is not a part of
    it, such as the ripeness of a fruit. Its value changes while the bearer stays the
    same.

    .. note:: Dispositions, such as being graspable, are not qualities here; they are
        mixins of the annotations that have them.
    """


# %% Ripeness


@dataclass(kw_only=True)
class RipenessStage(ABC):
    """
    A named stage of ripening: a fixed interval of :attr:`Ripeness.scale` that a
    ripeness lies in.

    .. note:: The boundaries between the stages are a convention of this model; no
        ontology or horticultural source fixes them.
    """

    @classproperty
    @abstractmethod
    def ripeness_interval(cls) -> SimpleInterval:
        """
        :return: The part of :attr:`Ripeness.scale` this stage covers.
        """


@dataclass(kw_only=True)
class Unripe(RipenessStage):
    """
    The first third of ripening, such as a green banana or a firm avocado.
    """

    @classproperty
    def ripeness_interval(cls) -> SimpleInterval:
        return SimpleInterval.from_data(0.0, 1 / 3, Bound.CLOSED, Bound.OPEN)


@dataclass(kw_only=True)
class Ripe(RipenessStage):
    """
    The middle third of ripening, in which a fruit is usually eaten.
    """

    @classproperty
    def ripeness_interval(cls) -> SimpleInterval:
        return SimpleInterval.from_data(1 / 3, 2 / 3, Bound.CLOSED, Bound.OPEN)


@dataclass(kw_only=True)
class Overripe(RipenessStage):
    """
    The last third of ripening, in which a fruit softens and starts to spoil.
    """

    @classproperty
    def ripeness_interval(cls) -> SimpleInterval:
        return SimpleInterval.from_data(2 / 3, 1.0, Bound.CLOSED, Bound.CLOSED)


@dataclass(kw_only=True)
class Ripeness(Quality):
    """
    How far a fruit has ripened, on one :attr:`scale` shared by every fruit.

    Ripening is a gradient: the stages :class:`Unripe`, :class:`Ripe` and
    :class:`Overripe` are intervals of the scale, and :attr:`stage` classifies the value.
    """

    value: SimpleInterval
    """
    The position of the ripeness on :attr:`scale`: a single point when it was measured,
    or the interval of a :class:`RipenessStage` when only the stage is known.
    """

    def __post_init__(self) -> None:
        """
        :raises RipenessOutsideScale: If the value reaches beyond :attr:`scale`.
        """
        if self.scale.intersection_with(self.value) != self.value:
            raise RipenessOutsideScale(ripeness=self)

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
        return cls(value=stage.ripeness_interval)

    @property
    def stage(self) -> RipenessStage:
        """
        :return: The stage whose interval holds the whole value.
        :raises RipenessSpansSeveralStages: If the value reaches into several stages.
        """
        for stage_type in recursive_subclasses(RipenessStage):
            if stage_type.ripeness_interval.intersection_with(self.value) == self.value:
                return stage_type()
        raise RipenessSpansSeveralStages(ripeness=self)
