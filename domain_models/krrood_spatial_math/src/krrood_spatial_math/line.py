"""
Straight lines of an affine space.
"""

from __future__ import annotations

from typing_extensions import Generic, TypeVar

from krrood.patterns.interface import Interface
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood_spatial_math.coordinate_system import CoordinateT
from krrood_spatial_math.point import PointT
from krrood_spatial_math.vector import VectorT

# %% straight lines given by a point and a direction


class Line(Generic[PointT, VectorT], SubClassSafeGeneric, Interface):
    """
    A straight line of an affine space: the points ``point + direction * t`` for every
    number ``t``.

    A direction alone fixes only which way a line runs, and every parallel line shares
    it; :attr:`point` fixes which of those lines this one is.
    """

    point: PointT
    """
    A point the line passes through, where ``t`` is zero.
    """

    direction: VectorT
    """
    The vector the line runs along, given in the frame of :attr:`point`.
    """

    def point_at_position(self, *, position_along_line: CoordinateT) -> PointT:
        """
        :param position_along_line: How many :attr:`direction` vectors the point lies
            from :attr:`point`; negative positions lie behind it.
        :return: The point ``point + direction * position_along_line`` of this line.
        """
        return self.point + self.direction * position_along_line


LineT = TypeVar("LineT", bound=Line)
"""
The kind of straight line of an affine space.
"""


# %% straight lines between two points


class LineSegment(Generic[PointT, LineT], SubClassSafeGeneric, Interface):
    """
    The part of a straight line between two of its points: the points of :attr:`line`
    from position 0 at :attr:`start` to position 1 at :attr:`end`.
    """

    start: PointT
    """
    The point the segment runs from.
    """

    end: PointT
    """
    The point the segment runs to, given in the frame of :attr:`start`.
    """

    @property
    def line(self) -> LineT:
        """
        :return: The line through :attr:`start` along the vector to :attr:`end`, on
            which :attr:`start` lies at position 0 and :attr:`end` at position 1.
        """
        line_type = self.get_type_of_generic_parameter(parameter=LineT)
        return line_type(point=self.start, direction=self.end - self.start)
