"""
Spaces: the sets of points that coordinate systems chart, with their dimension.
"""

from __future__ import annotations

from abc import abstractmethod

from krrood.patterns.classproperty import classproperty
from krrood.patterns.interface import Interface

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

    @classproperty
    @abstractmethod
    def dimension(cls) -> int:
        """
        The number of independent directions in this space, which is how many
        coordinates a Cartesian frame on it gives each point.
        """


# %% Euclidean spaces of a fixed dimension


class EuclideanSpace2Dimensional(EuclideanSpace, Interface):
    """
    A two-dimensional Euclidean space, such as the floor a robot drives on.
    """

    @classproperty
    def dimension(cls) -> int:
        return 2


class EuclideanSpace3Dimensional(EuclideanSpace, Interface):
    """
    A three-dimensional Euclidean space, such as the physical space a robot's bodies are
    in.
    """

    @classproperty
    def dimension(cls) -> int:
        return 3
