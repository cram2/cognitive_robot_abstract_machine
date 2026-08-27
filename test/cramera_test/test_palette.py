"""
Recorded and live object colours must come from one cycle.

The onboarder bakes colours into a scene bundle while the bridge assigns them on the
fly. When the two hold separate lists, an object beyond the shorter list's length
changes colour the moment the viewer attaches to a running demo.
"""

import inspect

from semantic_digital_twin.world_description.geometry import Color

from cramera.live import bridge
from cramera.onboard import demo
from cramera.palette import OBJECT_COLORS, ObjectPalette


class TestObjectPalette:
    def test_colors_are_assigned_in_order(self):
        palette = ObjectPalette()
        assert palette.color_for(0) == OBJECT_COLORS[0]
        assert palette.color_for(2) == OBJECT_COLORS[2]

    def test_the_cycle_wraps_around(self):
        palette = ObjectPalette()
        assert palette.color_for(len(OBJECT_COLORS)) == OBJECT_COLORS[0]

    def test_colors_are_distinct(self):
        assert len(set(OBJECT_COLORS)) == len(OBJECT_COLORS)

    def test_colors_are_stored_as_the_shared_color_type(self):
        """
        The cycle is kept internally as :class:`Color`, converting to ``#rrggbb`` only
        at :meth:`~cramera.palette.ObjectPalette.color_for`'s boundary.
        """
        assert all(isinstance(color, Color) for color in ObjectPalette().colors)


class TestOnlyOnePaletteExists:
    def test_neither_producer_defines_its_own_cycle(self):
        """
        Both colour producers must reference the shared cycle, not a private copy.
        """
        for module in (bridge, demo):
            source = inspect.getsource(module)
            assert "ObjectPalette" in source, module.__name__
            assert "#f3f0ea" not in source, module.__name__

    def test_both_producers_agree_beyond_the_shortest_former_list(self):
        """
        Index 6 is where the bridge's old ten-colour list and the onboarder's old six-
        colour list disagreed.
        """
        palette = ObjectPalette()
        assert palette.color_for(6) == OBJECT_COLORS[6]
