"""
Interfaces of regions in the plane, layered so that :class:`Disc` derives from
:class:`Region` only through :class:`ConvexRegion`.
"""

from __future__ import annotations

from krrood.patterns.interface import Interface

# %% the layered interfaces


class Region(Interface):
    """
    A part of the plane.
    """


class ConvexRegion(Region, Interface):
    """
    A region that contains the straight line between any two of its points.
    """


class Disc(ConvexRegion, Interface):
    """
    The points of the plane within some distance of a centre.
    """
