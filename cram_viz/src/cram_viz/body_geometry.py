"""
The measurable size of a world body's geometry.

Both the live bridge (which sizes placeholder boxes for objects the viewer has no mesh
for) and the onboarder (which records each object's height into a bundle) need the same
measurement, taken the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from semantic_digital_twin.world_description.world_entity import Body


@dataclass(frozen=True)
class BodyExtent:
    """
    A body's size along each axis, in metres.
    """

    x: float
    """
    Extent along the world x axis.
    """

    y: float
    """
    Extent along the world y axis.
    """

    z: float
    """
    Extent along the world z axis, i.e. the body's height.
    """

    @classmethod
    def of(cls, body: Body) -> Optional[BodyExtent]:
        """
        Measure a body from the first of its shape collections that has any shapes.

        Checks :attr:`Body.visual` before :attr:`Body.collision`, using
        :attr:`ShapeCollection.scale`, which measures any shape type from its bounding
        box rather than relying on a shape-specific scale attribute.

        :param body: The body to measure.
        :return: The extent, or None when both collections are empty.
        """
        for shape_collection in (body.visual, body.collision):
            if not shape_collection.shapes:
                continue
            scale = shape_collection.scale
            return cls(x=float(scale.x), y=float(scale.y), z=float(scale.z))
        return None

    def rounded(self, precision: int) -> List[float]:
        """
        The extent as ``[x, y, z]``, rounded for publication.

        :param precision: Number of decimal places to round each axis to.
        """
        return [
            round(self.x, precision),
            round(self.y, precision),
            round(self.z, precision),
        ]
