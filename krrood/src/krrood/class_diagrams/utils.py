from __future__ import annotations

import inspect
import sys
from enum import Enum
from typing import Callable, Any, Dict, get_args, get_origin, Union
from uuid import UUID

import typing_extensions
from typing_extensions import List, Type, Any, Dict, TypeVar, Tuple
from typing_extensions import TypeVar

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


def get_and_resolve_generic_type_hints_of_object_using_substitutions(
    object_: Any, substitution: Dict[TypeVar, Type]
) -> Tuple[Dict[str, Type], Dict[str, Type]]:
    """
    Resolve generic type hints of an object using a substitution dictionary.

    :param object_: The object to resolve generic type hints of.
    :param substitution: The substitution dictionary to use for resolving generic type hints.
    :return: A tuple containing two dictionaries: the first maps type variable names to resolved types, and the second
     maps type variable names to original type hints.
    """
    type_hints = get_type_hints_of_object(object_)
    resolved_types = {}
    for name, hint in type_hints.items():
        resolved_types[name] = resolve_type(hint, substitution)
    return resolved_types, type_hints


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
                source = inspect.getsource(object_)
                scope = get_scope_from_imports(source=source)
                if e.name in scope:
                    local_namespace[e.name] = scope[e.name]
                    continue
            except OSError as os_error:
                raise CouldNotResolveType(e.name, os_error)
    return type_hints


def resolve_type(
    type_to_resolve: Any,
    substitution: Dict[TypeVar, Any],
) -> Any:
    """
    Resolve type variables and forward references in a type.

    :param type_to_resolve: The type to resolve.
    :param substitution: Mapping of TypeVar to concrete types.
    :return: The resolved type.
    """
    # Also map by TypeVar name to handle postponed annotations ('T')
    name_substitution = {p.__name__: a for p, a in substitution.items()}
    # Resolve string forward refs and TypeVar names
    if isinstance(type_to_resolve, str):
        if type_to_resolve in name_substitution:
            return name_substitution[type_to_resolve]
        return type_to_resolve

    if isinstance(type_to_resolve, TypeVar):
        return substitution.get(type_to_resolve, type_to_resolve)

    # Get arguments and recursively resolve them
    args = get_args(type_to_resolve)
    if not args:
        return type_to_resolve

    resolved_args = tuple(resolve_type(arg, substitution) for arg in args)

    # If the type itself can be indexed (like List[T] or Optional[T])
    params = getattr(type_to_resolve, "__parameters__", None)
    if hasattr(type_to_resolve, "__getitem__") and params:
        if len(params) < len(resolved_args):
            # Filter out NoneType if it's an Optional/Union and we have more args than parameters
            new_args = tuple(arg for arg in resolved_args if arg is not type(None))
            if len(new_args) == len(params):
                if len(params) == 1:
                    return type_to_resolve[new_args[0]]
                return type_to_resolve[new_args]

        if len(params) == 1 and len(resolved_args) == 1:
            return type_to_resolve[resolved_args[0]]
        return type_to_resolve[resolved_args]

    # Fallback: re-construct from origin (e.g. for Union/Optional or built-in generics)
    origin = get_origin(type_to_resolve)
    if origin is not None:
        # Special case for Union which might be represented as typing.Union
        # and needs to be indexed.
        if origin is Union:
            return origin[resolved_args]
        try:
            return origin[resolved_args]
        except TypeError:
            # Some origins might not be indexable directly or might need single arg
            if len(resolved_args) == 1:
                return origin[resolved_args[0]]
            raise

    return type_to_resolve
