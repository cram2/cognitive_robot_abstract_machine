"""
Values that hash by identity and are created anew every time they are asked for, so two
evaluations of the same expression yield equal-looking but distinct values.
"""

from __future__ import annotations

from dataclasses import dataclass

# %% values hashed by identity


@dataclass(eq=False)
class IdentityHashedPoint:
    """
    A point on a line that equals and hashes like no other point, whatever its
    coordinate.
    """

    coordinate: float
    """
    Where the point lies on the line.
    """


# %% a holder creating its points on every access


@dataclass
class PointsCreatedOnEveryAccess:
    """
    Hands out new points at the same coordinates every time its points are read.
    """

    coordinates: list[float]
    """
    Where the points it hands out lie on the line.
    """

    @property
    def points(self) -> list[IdentityHashedPoint]:
        """
        :return: A new point at each of :attr:`coordinates`.
        """
        return [IdentityHashedPoint(coordinate) for coordinate in self.coordinates]


# %% an object generated around one of the points


@dataclass
class PlacementOnAPoint:
    """
    Something put down at a point.
    """

    point: IdentityHashedPoint
    """
    The point it is put down at.
    """
