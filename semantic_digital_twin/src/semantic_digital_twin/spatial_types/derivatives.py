from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Self

from typing_extensions import Generic, TypeVar, List, Optional

T = TypeVar("T")


class Derivatives(IntEnum):
    """
    Enumeration of interpretation for the order of derivatives on spatial positions.
    """

    position = 0
    velocity = 1
    acceleration = 2
    jerk = 3

    @classmethod
    def range(cls, start: Derivatives, stop: Derivatives, step: int = 1):
        """
        Includes stop!
        """
        return [item for item in cls if start <= item <= stop][::step]


@dataclass
class DerivativeMap(Generic[T]):
    """
    A container class that maps derivatives (position, velocity, acceleration, jerk) to values of type T.

    This class provides a structured way to store and access different orders of derivatives.
    Each derivative order can hold a value of type T or None.
    """

    position: Optional[T] = None
    velocity: Optional[T] = None
    acceleration: Optional[T] = None
    jerk: Optional[T] = None

    @property
    def data(self) -> List[Optional[T]]:
        """
        :return: A list of all derivative values.
        """
        return [self[d] for d in Derivatives]

    def __hash__(self):
        return hash(tuple(self.data))

    def _broadcast_callable(self, operand: Callable[[Optional[T]], T]) -> Self:
        """
        Apply a callable to each derivative value and return a new instance with the resulting values.

        :param operand: The callable to apply. Make sure it can deal with None values.
        :return: The new instance with the resulting values.
        """
        return type(self)(
            operand(self.position),
            operand(self.velocity),
            operand(self.acceleration),
            operand(self.jerk),
        )

    def __mul__(self, other: float) -> DerivativeMap[T]:
        return self._broadcast_callable(lambda v: v * other if v is not None else None)

    def __add__(self, other: float) -> DerivativeMap[T]:
        return self._broadcast_callable(lambda v: v + other if v is not None else None)

    def __setitem__(self, key: Derivatives, value: T):
        """
        Set an attribute using the `Derivatives` Enum.

        :param key: The derivative to set.
        :param value: The value to set.
        """
        assert hasattr(self, key.name)
        self.__setattr__(key.name, value)

    def __getitem__(self, item: Derivatives) -> T:
        """
        Get an attribute using the `Derivatives` Enum.

        :param item: The derivative.
        :return: The value.
        """
        assert hasattr(self, item.name)
        return self.__getattribute__(item.name)
