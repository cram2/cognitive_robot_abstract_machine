from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import TypeVar, Type, List

T = TypeVar("T")


def recursive_subclasses(cls: Type[T]) -> List[Type[T]]:
    """
    :param cls: The class.
    :return: A list of the classes subclasses without the class itself.
    """
    return cls.__subclasses__() + [
        g for s in cls.__subclasses__() for g in recursive_subclasses(s)
    ]


@dataclass
class DataclassException(Exception):
    """
    A base exception class for dataclass-based exceptions.
    The way this is used is by inheriting from it and setting the `message` field in the __post_init__ method,
    then calling the super().__post_init__() method.
    """

    message: str = field(kw_only=True, default=None)

    def __post_init__(self):
        super().__init__(self.message)


def get_full_class_name(cls):
    """
    Returns the full name of a class, including the module name.

    :param cls: The class.
    :return: The full name of the class
    """
    return cls.__module__ + "." + cls.__name__
