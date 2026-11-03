from __future__ import annotations

import matplotlib  # unguarded

from krrood.optional_modules import KrroodOptionalModule


def draw() -> None:
    from matplotlib import pyplot  # unguarded

    pyplot.show()


class Painter:
    from matplotlib.patches import Patch  # unguarded

    @staticmethod
    def paint() -> None:
        import matplotlib.pyplot  # unguarded

        matplotlib.pyplot.show()

    @KrroodOptionalModule.MATPLOTLIB.ensure_installed
    def sketch(self) -> None:
        import matplotlib  # unguarded
