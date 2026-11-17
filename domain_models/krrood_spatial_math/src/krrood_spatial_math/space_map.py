"""
Maps that send the points of one space to the points of another.
"""

from __future__ import annotations

from abc import abstractmethod

from typing_extensions import Generic, TypeVar

from krrood.patterns.interface import Interface
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood_spatial_math.coordinate_system import CoordinateT
from krrood_spatial_math.line import Line
from krrood_spatial_math.point import Point
from krrood_spatial_math.vector import Vector

# %% maps from the points of one space to the points of another

SourcePointT = TypeVar("SourcePointT", bound=Point)
"""
The kind of point a space map takes, given in a frame of its source space.
"""

TargetPointT = TypeVar("TargetPointT", bound=Point)
"""
The kind of point a space map gives, in a frame of its target space.
"""

ProjectionLineT = TypeVar("ProjectionLineT", bound=Line)
"""
The kind of line, such as a straight line or a ray, of the source points a space map
sends to one target point.
"""


class SpaceMap(
    Generic[SourcePointT, TargetPointT, ProjectionLineT, CoordinateT],
    SubClassSafeGeneric,
    Interface,
):
    """
    Sends each point of one space to a point of another space of lower dimension, such
    as 3D space onto a plane or onto a camera's image.

    Every target point is where the source points on one line, its
    :meth:`projection_line`, are sent, and :meth:`unproject` picks one of them. For
    every target point ``p`` and position ``t``, ``project(unproject(p, t)) == p``.

    .. note::
        A space map relates two spaces, so it is never a coordinate change, which stays
        within one space.
    """

    @abstractmethod
    def project(self, point: SourcePointT) -> TargetPointT:
        """
        :param point: A point of the source space.
        :return: The point of the target space ``point`` is sent to.
        """

    @abstractmethod
    def projection_line(self, point: TargetPointT) -> ProjectionLineT:
        """
        :param point: A point of the target space.
        :return: The line of every source point :meth:`project` sends to ``point``.
        """

    @abstractmethod
    def unproject(
        self, point: TargetPointT, position_along_line: CoordinateT
    ) -> SourcePointT:
        """
        :param point: A point of the target space.
        :param position_along_line: How many direction lengths the returned point lies
            from the point the :meth:`projection_line` of ``point`` is given by.
        :return: The point at ``position_along_line`` on the :meth:`projection_line` of
            ``point``.
        """


# %% projections along the normal of a plane


class OrthogonalProjection(
    SpaceMap[SourcePointT, TargetPointT, ProjectionLineT, CoordinateT], Interface
):
    """
    Sends each point of a space onto a plane in it, straight along the plane's unit
    normal, so every projection line runs along that one direction.

    The position along a projection line is the height above the plane.
    """

    plane_normal: Vector
    """
    The vector of unit length, perpendicular to the plane, that every projection line
    runs along; it points to the side whose heights are positive.
    """

    @abstractmethod
    def embed_planar_point(self, *, point: TargetPointT) -> SourcePointT:
        """
        :param point: A point of the plane.
        :return: The same point, given as a point of the space the plane lies in.
        """

    def projection_line(self, point: TargetPointT) -> ProjectionLineT:
        line_type = self.get_type_of_generic_parameter(parameter=ProjectionLineT)
        return line_type(
            point=self.embed_planar_point(point=point), direction=self.plane_normal
        )

    def unproject(
        self, point: TargetPointT, position_along_line: CoordinateT = 0
    ) -> SourcePointT:
        """
        Differs from :meth:`SpaceMap.unproject` in defaulting ``position_along_line`` to
        zero, the point on the plane itself.
        """
        return self.embed_planar_point(point=point) + self.plane_normal * position_along_line
