"""
A module whose class leaves a field and a method undocumented.
"""

from __future__ import annotations

from typing import override


class Base:
    """
    A documented class.
    """

    def describe(self) -> str:
        """
        A documented method.
        """
        return ""


class Undocumented(Base):
    """
    A class with one undocumented field and one undocumented method.
    """

    documented: int = 0
    """
    A documented field.
    """

    undocumented: int = 0

    @override
    def describe(self) -> str:
        return "inherits the base docstring"

    def undocumented_method(self) -> None:
        pass
