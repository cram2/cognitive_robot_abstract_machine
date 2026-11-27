"""
Classes that declare what their subclasses provide without providing it.
"""

from __future__ import annotations

from abc import ABC

from typing_extensions import Any, Self

from krrood.patterns.exceptions import InterfaceInstantiationError

# %% interfaces


class Interface(ABC):
    """
    A class that declares the typed attributes and methods its subclasses provide, but
    not how they are stored: each subclass decides whether an attribute is a field or a
    property. It cannot be instantiated.

    .. note::
        Never make an interface a dataclass, which turns its declarations into fields.
        Only a class naming :class:`Interface` among its own bases is an interface.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        if Interface.is_interface(class_=cls):
            raise InterfaceInstantiationError(interface=cls)
        return super().__new__(cls)

    @staticmethod
    def is_interface(*, class_: type) -> bool:
        """
        :param class_: The class to check.
        :return: Whether the class names :class:`Interface` among its own bases.
        """
        return Interface in class_.__bases__
