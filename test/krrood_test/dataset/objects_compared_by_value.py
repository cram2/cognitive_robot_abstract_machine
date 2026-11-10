"""
Objects that compare and hash by value, so two distinct objects can be equal.
"""

from __future__ import annotations

from dataclasses import dataclass

# %% objects compared by value


@dataclass(frozen=True)
class PointComparedByValue:
    """
    A point on a line, equal to every other point at the same coordinate.
    """

    coordinate: float
    """
    Where the point lies on the line.
    """

    def shifted_by(self, distance: float) -> float:
        """
        :param distance: How far to move along the line.
        :return: The coordinate *distance* away from this point.
        """
        return self.coordinate + distance
