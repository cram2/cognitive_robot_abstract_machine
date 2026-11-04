from __future__ import annotations

from abc import ABC

from typing_extensions import Any, Self

from krrood.patterns.exceptions import InterfaceInstantiationError

# %% interfaces


class Interface(ABC):
    """
    A class that only declares the typed attributes and methods its subclasses provide,
    as fields or properties. It cannot be instantiated.

    .. note::
        Never make an interface a dataclass, which turns its declarations into fields.
        A base class with fields or behaviour of its own subclasses :class:`abc.ABC`.
        Only a class naming :class:`Interface` among its own bases is an interface.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        if Interface.is_interface(cls):
            raise InterfaceInstantiationError(cls)
        return super().__new__(cls)

    @staticmethod
    def is_interface(class_: type) -> bool:
        """
        :param class_: The class to check.
        :return: Whether the class names :class:`Interface` among its own bases.
        """
        return Interface in class_.__bases__
