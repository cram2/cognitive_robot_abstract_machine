from __future__ import annotations

import inspect
import sys
from copy import copy
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Callable, Any, Dict, get_args, get_origin, Union
from uuid import UUID

import typing_extensions
from typing_extensions import List, Type, Generic, TYPE_CHECKING
from typing_extensions import TypeVar, get_origin, get_args

from krrood.class_diagrams.exceptions import CouldNotResolveType
from krrood.utils import get_scope_from_imports


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


@dataclass
class Role(Generic[T]):
    """
    Represents a role with generic typing. This is used in Role Design Pattern in OOP.

    This class serves as a container for defining roles with associated generic
    types, enabling flexibility and type safety when modeling role-specific
    behavior and data.
    """


def get_generic_type_param(cls, generic_base):
    """
    Given a subclass and its generic base, return the concrete type parameter(s).

    Example:
        get_generic_type_param(Employee, Role) -> (<class '__main__.Person'>,)
    """
    for base in getattr(cls, "__orig_bases__", []):
        base_origin = get_origin(base)
        if base_origin is None:
            continue
        if issubclass(get_origin(base), generic_base):
            return get_args(base)
    return None


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

def get_type_hints_of_object(object_: Any) -> Dict[str, Any]:
    """
    Get the type hints of an object. This is a workaround for the fact that get_type_hints() does not work with objects
     that are not defined in the same module or are imported through TYPE_CHECKING.

    :param object_: The object to get the type hints of.
    :return: The type hints of the object as a dictionary.
    """
    type_hints = {}
    local_namespace = locals()
    while True:
        try:
            type_hints = typing_extensions.get_type_hints(
                object_, include_extras=True, localns=local_namespace
            )
            break
        except NameError as e:
            module = inspect.getmodule(object_)
            if module is not None and hasattr(module, e.name):
                local_namespace[e.name] = getattr(module, e.name)
                continue
            try:
                source_path = inspect.getsourcefile(object_)
                scope = get_scope_from_imports(file_path=source_path)
                if e.name in scope:
                    local_namespace[e.name] = scope[e.name]
                    continue
            except OSError as os_error:
                raise CouldNotResolveType(e.name, os_error)
    return type_hints
