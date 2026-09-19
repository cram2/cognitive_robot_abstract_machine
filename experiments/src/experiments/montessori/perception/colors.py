"""
The colours each kind of detection is drawn in, wherever it is drawn.

Kept in one place so a piece looks the same in rviz as it does in the camera window, and
a reader comparing the two is not misled by the same thing wearing two colours.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Tuple

# %% one colour


@dataclass(frozen=True)
class DetectionColor:
    """
    A colour, held apart from the order any one drawing library wants its parts in.
    """

    red: float
    """
    How much red, from zero to one.
    """

    green: float
    """
    How much green, from zero to one.
    """

    blue: float
    """
    How much blue, from zero to one.
    """

    def to_rgba(self, alpha: float = 1.0) -> Tuple[float, float, float, float]:
        """
        :param alpha: How opaque to make it, from zero to one.
        :return: The colour as rviz takes it, red first and each part from zero to one.
        """
        return self.red, self.green, self.blue, alpha

    def to_bgr(self) -> Tuple[int, int, int]:
        """
        :return: The colour as OpenCV takes it, blue first and each part a whole number
            from zero to 255.
        """
        return tuple(
            round(part * FULL_INTENSITY) for part in (self.blue, self.green, self.red)
        )


FULL_INTENSITY = 255
"""
Value a colour's part reaches when it is at its strongest.
"""

# %% what each kind is drawn in


PIECE_COLOR = DetectionColor(0.1, 0.9, 0.3)
"""
Colour a loose piece is drawn in.
"""

HOLE_COLOR = DetectionColor(1.0, 0.6, 0.0)
"""
Colour a hole in the board is drawn in.
"""

BOARD_COLOR = DetectionColor(0.3, 0.5, 1.0)
"""
Colour the board itself is drawn in.
"""

LABEL_COLOR = DetectionColor(1.0, 1.0, 1.0)
"""
Colour a detection's name is written in.
"""
