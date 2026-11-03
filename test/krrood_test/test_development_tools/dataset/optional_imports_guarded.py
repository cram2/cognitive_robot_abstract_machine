from __future__ import annotations

from typing_extensions import TYPE_CHECKING

from krrood.optional_modules import KrroodOptionalModule

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@KrroodOptionalModule.MATPLOTLIB.require
def draw(figure: Figure) -> None:
    from matplotlib import pyplot

    def show() -> None:
        import matplotlib.pyplot

        matplotlib.pyplot.show()

    pyplot.figure(figure)
    show()


class Painter:
    @KrroodOptionalModule.MATPLOTLIB.require
    def paint(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
