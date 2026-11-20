from __future__ import annotations

import inspect
from abc import ABC, update_abstractmethods

from typing_extensions import Any, Self

from krrood.patterns.exceptions import (
    InterfaceInstantiationError,
    MissingInterfaceAttributesError,
)

# %% interfaces


class Interface(ABC):
    """
    A class that only declares the typed attributes and methods its subclasses provide,
    as fields or properties. It cannot be instantiated, and a class that implements all
    of its abstract methods cannot be defined without providing every attribute its
    interfaces declare.

    .. note::
        Never make an interface a dataclass, which turns its declarations into fields.
        A base class with fields or behaviour of its own subclasses :class:`abc.ABC`.
        Only a class naming :class:`Interface` among its own bases is an interface.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        if Interface.is_interface(class_=cls):
            raise InterfaceInstantiationError(interface=cls)
        return super().__new__(cls)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if Interface.is_interface(class_=cls) or Interface.is_abstract(class_=cls):
            return
        missing_attributes = Interface.missing_attributes(class_=cls)
        if missing_attributes:
            raise MissingInterfaceAttributesError(
                implementation=cls, attributes=missing_attributes
            )

    @staticmethod
    def is_interface(*, class_: type) -> bool:
        """
        :param class_: The class to check.
        :return: Whether the class names :class:`Interface` among its own bases.
        """
        return Interface in class_.__bases__

    @staticmethod
    def is_abstract(*, class_: type) -> bool:
        """
        :param class_: A class that is being created.
        :return: Whether the class leaves an abstract method unimplemented.
        """
        # abc finds the abstract methods only once the class is created, and
        # update_abstractmethods only updates a class that already has them
        class_.__abstractmethods__ = frozenset()
        update_abstractmethods(class_)
        return inspect.isabstract(class_)

    @staticmethod
    def missing_attributes(*, class_: type) -> tuple[str, ...]:
        """
        :param class_: A class implementing interfaces.
        :return: The attributes the interfaces of the class declare that neither the
            class nor any of its other bases declares or defines, in name order.
        """
        implementations = [
            base for base in class_.__mro__ if not Interface.is_interface(class_=base)
        ]
        declared_attributes = {
            name
            for base in class_.__mro__
            if Interface.is_interface(class_=base)
            for name in inspect.get_annotations(base)
        }
        return tuple(
            sorted(
                name
                for name in declared_attributes
                if not any(
                    name in vars(base) or name in inspect.get_annotations(base)
                    for base in implementations
                )
            )
        )
