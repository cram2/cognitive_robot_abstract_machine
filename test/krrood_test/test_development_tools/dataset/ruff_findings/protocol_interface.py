"""
A module declaring interfaces as protocols.
"""

from __future__ import annotations

import typing_extensions
from typing_extensions import Protocol  # reported


class Drawable(Protocol):
    """
    An interface whose base is imported by name.
    """

    def draw(self) -> None:
        """
        Draws the shape.
        """


class Measurable(typing_extensions.Protocol):  # reported
    """
    An interface whose base is reached through its module.
    """

    def area(self) -> float:
        """
        The shape's area.
        """
