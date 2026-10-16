"""
Physical and digital quantities: amounts kept apart by their unit, and how fast they flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from numbers import Integral

from typing_extensions import Generic, Self, TypeVar

from krrood.exceptions import DifferentQuantities
from krrood.ormatic.utils import classproperty
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric

# %% quantities

MagnitudeT = TypeVar("MagnitudeT", int, float)
"""
The kind of number a :class:`Quantity` counts its unit in.
"""


@dataclass(order=True)
class Quantity(Generic[MagnitudeT], SubClassSafeGeneric):
    """
    An amount of one unit, which only adds to, subtracts from and compares with an amount
    of the same quantity.
    """

    magnitude: MagnitudeT
    """
    How many of the quantity's unit it is.
    """

    @classmethod
    def nearest(cls, magnitude: float) -> Self:
        """
        :param magnitude: How many of the unit, possibly a fraction.
        :return: The amount of that many units, to the nearest whole unit if the
            quantity counts in whole numbers.
        """
        magnitude_type = cls.get_type_of_generic_parameter(MagnitudeT)
        if issubclass(magnitude_type, Integral):
            return cls(round(magnitude))
        return cls(magnitude_type(magnitude))

    def scaled(self, factor: float) -> Self:
        """
        :param factor: What to multiply the amount by.
        :return: The amount that many times over.
        """
        return self.nearest(self.magnitude * factor)

    def __add__(self, other: Self) -> Self:
        self.ensure_same_quantity(other)
        return type(self)(self.magnitude + other.magnitude)

    def __sub__(self, other: Self) -> Self:
        self.ensure_same_quantity(other)
        return type(self)(self.magnitude - other.magnitude)

    def __truediv__(self, other: Self) -> float:
        """
        :param other: An amount of the same quantity.
        :return: How many times the other amount fits into this one.
        :raises DifferentQuantities: If the other amount is of another quantity.
        """
        self.ensure_same_quantity(other)
        return self.magnitude / other.magnitude

    def ensure_same_quantity(self, other: Quantity) -> None:
        """
        :param other: Another amount.
        :raises DifferentQuantities: If it is of another quantity than this one.
        """
        if type(other) is not type(self):
            raise DifferentQuantities(first=self, second=other)


@dataclass(order=True)
class DataSize(Quantity[int]):
    """
    An amount of data, such as a file takes on disk, counted in bits.
    """

    @classproperty
    def bits_per_byte(cls) -> int:
        """
        How many bits make a byte.
        """
        return 8

    @classmethod
    def of_bytes(cls, count: float) -> DataSize:
        """
        :param count: How many bytes.
        :return: That amount of data, to the nearest bit.
        """
        return cls.nearest(count * cls.bits_per_byte)

    @property
    def bytes(self) -> int:
        """
        How many whole bytes it is.
        """
        return self.magnitude // self.bits_per_byte


# %% rates

QuantityT = TypeVar("QuantityT", bound=Quantity)
"""
The quantity a :class:`Rate` measures the flow of.
"""


@dataclass
class Rate(Generic[QuantityT], SubClassSafeGeneric):
    """
    How much of a quantity flows in each second.
    """

    per_second: QuantityT
    """
    The amount that flows in one second.
    """

    def over(self, duration: timedelta) -> QuantityT:
        """
        :param duration: How long it flows.
        :return: The amount that flows in that time.
        """
        return self.per_second.scaled(duration.total_seconds())

    def scaled(self, factor: float) -> Self:
        """
        :param factor: What to multiply the rate by.
        :return: The rate that many times over.
        """
        return type(self)(self.per_second.scaled(factor))

    @classmethod
    def filling(cls, amount: QuantityT, duration: timedelta) -> Self:
        """
        :param amount: An amount.
        :param duration: How long it may take to flow.
        :return: The rate at which exactly that amount flows in that time.
        """
        return cls(amount.scaled(1 / duration.total_seconds()))


@dataclass
class BitRate(Rate[DataSize]):
    """
    How much data flows in each second, such as a stream of a video takes.
    """
