"""
Straight lines of an affine space.
"""

from __future__ import annotations

from typing_extensions import Generic, TypeVar

from krrood.patterns.interface import Interface
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood_spatial_math.coordinate_system import GenericCoordinate
from krrood_spatial_math.point import GenericPoint
from krrood_spatial_math.vector import GenericVector

# %% straight lines given by a point and a direction


class Line(Generic[GenericPoint, GenericVector], SubClassSafeGeneric, Interface):
    """
    A straight line of an affine space: the points ``point + direction * t`` for every
    number ``t``.

    A direction alone fixes only which way a line runs, and every parallel line shares
    it; :attr:`point` fixes which of those lines this one is.
    """

    point: GenericPoint
    """
    A point the line passes through, where ``t`` is zero.
    """

    direction: GenericVector
    """
    The vector the line runs along, given in the frame of :attr:`point`.
    """

    def point_at_position(
        self, *, position_along_line: GenericCoordinate
    ) -> GenericPoint:
        """
        :param position_along_line: How many :attr:`direction` vectors the point lies
            from :attr:`point`; negative positions lie behind it.
        :return: The point ``point + direction * position_along_line`` of this line.
        """
        return self.point + self.direction * position_along_line


GenericLine = TypeVar("GenericLine", bound=Line)
"""
The kind of straight line of an affine space.
"""


# %% straight lines between two points


class LineSegment(Generic[GenericPoint, GenericLine], SubClassSafeGeneric, Interface):
    """
    The part of a straight line between two of its points: the points of :attr:`line`
    from position 0 at :attr:`start` to position 1 at :attr:`end`.
    """

    start: GenericPoint
    """
    The point the segment runs from.
    """

    end: GenericPoint
    """
    The point the segment runs to, given in the frame of :attr:`start`.
    """

    @property
    def line(self) -> GenericLine:
        """
        :return: The line through :attr:`start` along the vector to :attr:`end`, on
            which :attr:`start` lies at position 0 and :attr:`end` at position 1.
        """
        line_type = self.get_type_of_generic_parameter(parameter=GenericLine)
        return line_type(point=self.start, direction=self.end - self.start)
