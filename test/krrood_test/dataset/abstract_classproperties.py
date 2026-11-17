"""
Abstract base classes declaring a classproperty, and a class implementing it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from krrood.patterns.classproperty import classproperty

# %% abstract classproperties


class AbstractDescription(ABC):
    """
    A base class leaving its classproperty abstract.
    """

    @classproperty
    @abstractmethod
    def describe(cls) -> str:
        """
        A description its subclasses give.
        """


class ImplementedDescription(AbstractDescription):
    """
    A class implementing the abstract classproperty.
    """

    @classproperty
    def describe(cls) -> str:
        return "implemented"
