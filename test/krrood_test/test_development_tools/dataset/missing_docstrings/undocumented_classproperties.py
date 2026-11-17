"""
A module whose class implements an abstract classproperty and overrides an implemented
one, both undocumented.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from krrood.patterns.classproperty import classproperty


class Base(ABC):
    """
    A documented class.
    """

    @classproperty
    @abstractmethod
    def describe(cls) -> str:
        """
        A documented abstract classproperty.
        """

    @classproperty
    def label(cls) -> str:
        """
        A documented implemented classproperty.
        """
        return ""


class Undocumented(Base):
    """
    A class with two undocumented classproperties.
    """

    @classproperty
    def describe(cls) -> str:
        return "implements the abstract classproperty"

    @classproperty
    def label(cls) -> str:
        return "overrides the implemented classproperty"
