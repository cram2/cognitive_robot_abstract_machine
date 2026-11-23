"""
Spaces: the sets of points that coordinate systems chart.
"""

from __future__ import annotations

from typing_extensions import TYPE_CHECKING

from krrood.patterns.interface import Interface
from krrood_spatial_math.exceptions import FrameMeasuresNoLengthsError

if TYPE_CHECKING:
    from krrood_spatial_math.coordinate_system import GenericCoordinate
    from krrood_spatial_math.vector import Vector

# %% sets of points and their geometry


class Space(Interface):
    """
    A set of points with a geometry, charted by any number of coordinate systems.

    Coordinates in two coordinate systems of one space convert into each other; points
    of two spaces relate only through a map between the spaces, such as a camera.
    """


class EuclideanSpace(Space, Interface):
    """
    A space with distances and angles, such as physical space or the plane an image lies
    on.

    It has Cartesian frames, related by rigid coordinate changes, and may have other
    coordinate systems, such as affine pixel frames, related by other changes.
    """

    @classmethod
    def inner_product(cls, *, first: Vector, second: Vector) -> GenericCoordinate:
        """
        :param first: A translation of this kind of space, given in an affine frame.
        :param second: A translation given in the same frame.
        :return: The product of both lengths and the cosine of the angle between them:
            ``first`` times the Gram matrix of their frame kind times ``second``.
        :raises FrameMeasuresNoLengthsError: If their frame kind's Gram matrix is not
            known.
        """
        gram_matrix = first.frame_kind.gram_matrix
        if gram_matrix is None:
            raise FrameMeasuresNoLengthsError(frame_kind=first.frame_kind)
        return sum(
            first_coordinate * row[column] * second_coordinate
            for first_coordinate, row in zip(
                first.coordinates, gram_matrix, strict=True
            )
            for column, second_coordinate in enumerate(second.coordinates)
            if row[column] != 0
        )
