"""
Spaces: the sets of points that coordinate systems chart.
"""

from __future__ import annotations

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
