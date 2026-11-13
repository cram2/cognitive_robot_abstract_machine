"""
A core module that loads modules outside core when it is loaded, each such import marked.
"""

from __future__ import annotations

import matplotlib  # crosses
from matplotlib import pyplot  # crosses
from typing_extensions import TYPE_CHECKING

from mimic_package.drawing import draw  # crosses
from .. import drawing  # crosses

if TYPE_CHECKING:
    pass
else:
    import matplotlib.figure  # crosses


class Drawing:
    """
    A class whose body runs an import when the module is loaded.
    """

    import matplotlib.colors  # crosses
