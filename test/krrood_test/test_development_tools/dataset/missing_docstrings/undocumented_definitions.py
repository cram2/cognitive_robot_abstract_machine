"""
A module whose class leaves a field and two methods undocumented.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Base(ABC):
    """
    A documented class.
    """

    @abstractmethod
    def describe(self) -> str:
        """
        A documented abstract method.
        """

    def name(self) -> str:
        """
        A documented implemented method.
        """
        return ""


class Undocumented(Base):
    """
    A class with one undocumented field and two undocumented methods.
    """

    documented: int = 0
    """
    A documented field.
    """

    undocumented: int = 0

    def describe(self) -> str:
        return "implements the abstract method"

    def name(self) -> str:
        return "overrides the implemented method"

    def undocumented_method(self) -> None:
        pass
