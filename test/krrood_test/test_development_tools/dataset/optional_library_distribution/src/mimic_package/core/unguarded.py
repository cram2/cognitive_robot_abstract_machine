"""
A core module that imports matplotlib at module level, each such line marked.
"""

from __future__ import annotations

import matplotlib  # unguarded
from matplotlib import pyplot  # unguarded
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    pass
else:
    import matplotlib.figure  # unguarded


class Drawing:
    """
    A class whose body imports matplotlib when the module is loaded.
    """

    import matplotlib.colors  # unguarded
