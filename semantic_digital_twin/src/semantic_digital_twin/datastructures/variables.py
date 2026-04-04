from __future__ import annotations

from enum import Enum
from functools import cached_property

from krrood.ormatic.utils import classproperty
from random_events.variable import Continuous
from sortedcontainers import SortedSet


class SpatialVariables(Enum):
    """
    Enum for spatial variables used in the semantic digital twin. Used in the context of random events.
    """

    x = Continuous(name="x")
    y = Continuous(name="y")
    z = Continuous(name="z")

    @classproperty
    def xy(cls):
        return SortedSet([cls.x.value, cls.y.value])

    @classproperty
    def xz(cls):
        return SortedSet([cls.x.value, cls.z.value])

    @classproperty
    def yz(cls):
        return SortedSet([cls.y.value, cls.z.value])
