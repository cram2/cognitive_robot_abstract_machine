"""
A core module that loads only core modules, core dependencies and the standard library
when it is loaded.
"""

from __future__ import annotations

import json

from typing_extensions import TYPE_CHECKING

from . import sibling
from mimic_package.core.sibling import __doc__ as sibling_documentation

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def draw(figure: Figure) -> str:
    """
    Draw on a figure.
    """
    from matplotlib import pyplot

    from mimic_package.drawing import draw

    return json.dumps(pyplot.get_backend())


async def draw_later() -> None:
    """
    Draw once awaited.
    """
    import matplotlib.pyplot
