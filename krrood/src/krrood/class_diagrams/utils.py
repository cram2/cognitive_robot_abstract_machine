from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
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


@dataclass
class TypeHintResolutionResult:
    """
    Represents the result of resolving generic type hints of an object using a substitution dictionary.
    """

    resolved_type: TypeVar | Type | str
    """
    The resolved type or the original type hint if no substitution was made.
    """
    resolved: bool
    """
    Whether any substitutions have been made.
    """
    type_hint: TypeVar | Type | str
    """
    The original type hint.
    """


def get_and_resolve_generic_type_hints_of_object_using_substitutions(
    object_: Any, substitution: Dict[TypeVar, Type]
) -> Dict[str, TypeHintResolutionResult]:
    """
    Resolve generic type hints of an object using a substitution dictionary.

    :param object_: The object to resolve generic type hints of.
    :param substitution: The substitution dictionary to use for resolving generic type hints.
    :return: A dictionary mapping type variable names to TypeHintResolutionResult objects.
    """
    type_hints = get_type_hints_of_object(object_)
    return {name: resolve_type(hint, substitution) for name, hint in type_hints.items()}


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
) -> TypeHintResolutionResult:
    """
    Resolve type variables in a type.

    :param type_to_resolve: The type to resolve.
    :param substitution: Mapping of TypeVars to other types that will substitute the TypeVars.
    :return: A TypeHintResolutionResult object containing the resolved type and a boolean indicating whether any
    substitutions were made.
    """
    if isinstance(type_to_resolve, TypeVar):
        if type_to_resolve not in substitution:
            return TypeHintResolutionResult(type_to_resolve, False, type_to_resolve)
        return TypeHintResolutionResult(
            substitution[type_to_resolve], True, type_to_resolve
        )

    # If the type itself can be indexed (like List[T] or Optional[T])
    params = getattr(type_to_resolve, "__parameters__", None)
    if hasattr(type_to_resolve, "__getitem__") and params:
        new_params = []
        resolved: bool = False  # whether any substitutions were made
        for param in params:
            if param in substitution:
                new_params.append(substitution[param])
                resolved = True
            else:
                new_params.append(param)
        return TypeHintResolutionResult(
            type_to_resolve[*new_params], resolved, type_to_resolve
        )

    return TypeHintResolutionResult(type_to_resolve, False, type_to_resolve)
