import inspect
import sys
from enum import Enum
from typing import Callable
from uuid import UUID

import typing_extensions
from typing_extensions import List, Type
from typing_extensions import TypeVar


def classes_of_module(module) -> List[Type]:
    """
    Get all classes of a given module.

    :param module: The module to inspect.
    :return: All classes of the given module.
    """

    result = []
    for name, obj in inspect.getmembers(sys.modules[module.__name__]):
        if inspect.isclass(obj) and obj.__module__ == module.__name__:
            result.append(obj)
    return result


def behaves_like_a_built_in_class(
    clazz: Type,
) -> bool:
    return (
        is_builtin_class(clazz)
        or clazz == UUID
        or (inspect.isclass(clazz) and issubclass(clazz, Enum))
    )


def is_builtin_class(clazz: Type) -> bool:
    return clazz.__module__ == "builtins"


T = TypeVar("T")


def get_type_hint_of_keyword_argument(callable_: Callable, name: str):
    """
    :param callable_: A callable to inspect
    :param name: The name of the argument
    :return: The type hint of the argument
    """
    hints = typing_extensions.get_type_hints(
        callable_,
        globalns=getattr(callable_, "__globals__", None),
        localns=None,
        include_extras=True,  # keeps Annotated[...] / other extras if you use them
    )
    return hints.get(name)
