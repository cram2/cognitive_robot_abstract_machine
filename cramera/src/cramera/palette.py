"""
The colour cycle loose scene objects are drawn in.

Both the onboarder (which bakes colours into a scene bundle) and the live bridge (which
assigns them on the fly) use this one cycle, so the same object keeps its colour whether
the viewer renders the recording or the running world.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from semantic_digital_twin.world_description.geometry import Color
from typing_extensions import List

OBJECT_COLORS = (
    "#f3f0ea",
    "#cf5b3a",
    "#b8bcc4",
    "#e7c26a",
    "#7fb069",
    "#5b8cff",
    "#c98bdb",
    "#ff9d6b",
    "#6bd0c0",
    "#d0c86b",
)
"""
Distinct, muted colours that read well against the viewer's dark stage.
"""


@dataclass(frozen=True)
class ObjectPalette:
    """
    Assigns object colours by position, wrapping around when it runs out.
    """

    colors: List[Color] = field(
        default_factory=lambda: [
            # by name, not ``cls``: the factory runs after the class exists
            ObjectPalette._color_from_hex(hex_value)
            for hex_value in OBJECT_COLORS
        ]
    )
    """
    The colour cycle, in assignment order.
    """

    def color_for(self, index: int) -> str:
        """
        The colour of the object at the given position in the scene, as ``#rrggbb``.
        """
        return self._hex_from_color(self.colors[index % len(self.colors)])

    @staticmethod
    def _color_from_hex(hex_value: str) -> Color:
        """
        Parse a ``#rrggbb`` string into a :class:`Color`.

        :param hex_value: The colour, as a leading-``#`` hex triplet.
        """
        hex_value = hex_value.lstrip("#")
        red, green, blue = (
            int(hex_value[channel : channel + 2], 16) / 255 for channel in (0, 2, 4)
        )
        return Color(R=red, G=green, B=blue)

    @staticmethod
    def _hex_from_color(color: Color) -> str:
        """
        Render a :class:`Color` as a ``#rrggbb`` string.

        :param color: The colour to render.
        """
        return "#%02x%02x%02x" % (
            round(color.R * 255),
            round(color.G * 255),
            round(color.B * 255),
        )
