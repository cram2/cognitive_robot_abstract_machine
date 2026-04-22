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
from typing_extensions import List, Type, Generic, TYPE_CHECKING, Optional, Tuple
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


def is_external_module(module) -> bool:
    """
    Check if a module is external to the project.

    :param module: The module to check.
    :return: True if the module is external, False otherwise.
    """
    if module is None:
        return True
    if module.__name__ in ("builtins", "typing", "typing_extensions"):
        return True

    if not hasattr(module, "__file__"):
        return True

    file_path = module.__file__
    if file_path is None:
        return True

    if "site-packages" in file_path or "dist-packages" in file_path:
        return True

    # Handle standard library modules (this is a bit heuristic)
    if file_path.startswith("/usr/lib/python"):
        return True
    return False


def resolve_name_in_hierarchy(name: str, start_object: Any) -> Any:
    """
    Resolve a name by searching through the hierarchy of the start_object.

    :param name: The name to resolve.
    :param start_object: The object to start searching from.
    :return: The resolved object.
    :raises CouldNotResolveType: If the name cannot be resolved.
    """
    if not inspect.isclass(start_object):
        # Fallback to current module logic if not a class
        return get_object_by_name_from_another_object_in_same_module(name, start_object)

    for base in start_object.__mro__:
        module = inspect.getmodule(base)
        if is_builtin_class(base) or is_external_module(module):
            continue

        try:
            # Try finding it in the base class's module
            return get_object_by_name_from_another_object_in_same_module(name, base)
        except CouldNotResolveType:
            continue

    # Final fallback if hierarchy fails
    source_path = inspect.getsourcefile(start_object)
    raise CouldNotResolveType(
        name,
        extra_information=f"Could not find {name} in the hierarchy of {start_object} (starting from {source_path}).",
    )


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
    orig_bases = cls.__orig_bases__ if hasattr(cls, "__orig_bases__") else []
    for base in orig_bases:
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
    global_namespace = (
        callable_.__globals__ if hasattr(callable_, "__globals__") else None
    )
    hints = typing_extensions.get_type_hints(
        callable_,
        globalns=global_namespace,
        localns=None,
        include_extras=True,  # keeps Annotated[...] / other extras if you use them
    )
    return hints.get(name)


@lru_cache
def get_type_hints_of_object(
    object_: Any, namespace: Tuple[Tuple[str, Any], ...] = ()
) -> Dict[str, Any]:
    """
    Get the type hints of an object. This is a workaround for the fact that get_type_hints() does not work with objects
     that are not defined in the same module or are imported through TYPE_CHECKING.

    :param object_: The object to get the type hints of.
    :param namespace: A starting namespace to use for resolving type hints.
    :return: The type hints of the object as a dictionary.
    :raises CouldNotResolveType: If a type hint cannot be resolved.
    """
    if namespace:
        local_namespace = dict(namespace)
    else:
        local_namespace = {}
    while True:
        try:
            type_hints = typing_extensions.get_type_hints(
                object_, include_extras=True, localns=local_namespace
            )
            break
        except NameError as e:
            object_from_name = resolve_name_in_hierarchy(e.name, object_)
            local_namespace[e.name] = object_from_name
    return type_hints


def get_object_by_name_from_another_object_in_same_module(
    name: str, object_: Any
) -> Any:
    """
    Get the object with the given name from another object in the same module.

    :param name: The name of the type to get.
    :param object_: The object to get the type from.
    :return: The object with the given name.
    :raises CouldNotResolveType: If the type cannot be resolved.
    """
    module = inspect.getmodule(object_)
    if module is not None and hasattr(module, name):
        return getattr(module, name)
    source_path = inspect.getsourcefile(object_)
    if source_path is None:
        raise CouldNotResolveType(
            name, extra_information=f"Could not find source file for {object_}"
        )
    scope = get_scope_from_imports(file_path=source_path)
    if name in scope:
        return scope[name]
    else:
        raise CouldNotResolveType(
            name,
            extra_information=f"Could not find {name} in {source_path}, could be a deprecated import statement or "
            f"a type defined in a module that is not imported in the source file.",
        )
