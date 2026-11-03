"""
A core module that imports matplotlib only where loading the module does not run it.
"""

from __future__ import annotations

import json

from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def draw(figure: Figure) -> str:
    """
    Draw on a figure.
    """
    from matplotlib import pyplot

    return json.dumps(pyplot.get_backend())


async def draw_later() -> None:
    """
    Draw once awaited.
    """
    import matplotlib.pyplot
