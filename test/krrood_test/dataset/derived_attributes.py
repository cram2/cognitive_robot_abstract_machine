"""
A class whose instances expose attributes beyond the ones they are constructed with.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Rectangle:
    """
    A rectangle constructed from its sides, exposing its area as a derived attribute.
    """

    width: float
    """
    The length of the horizontal sides.
    """

    height: float
    """
    The length of the vertical sides.
    """

    layer: int = 0
    """
    The drawing layer the rectangle sits on, a constructor parameter that is also a
    class attribute.
    """

    @classmethod
    def from_sides(cls, first_side: float, second_side: float) -> Rectangle:
        """
        :param first_side: The length of the horizontal sides.
        :param second_side: The length of the vertical sides.
        :return: The rectangle with those sides, which keeps neither under the name of
            its parameter.
        """
        return cls(first_side, second_side)

    @property
    def area(self) -> float:
        """
        :return: The area the sides enclose.
        """
        return self.width * self.height

    def perimeter(self) -> float:
        """
        :return: The length of the boundary.
        """
        return 2 * (self.width + self.height)
