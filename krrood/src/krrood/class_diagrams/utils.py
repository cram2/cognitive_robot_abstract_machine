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
from typing_extensions import List, Type, Any, Dict, TypeVar, Tuple, Iterable, Iterator
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


def get_most_specific_types(types: Iterable[type]) -> List[type]:
    ts = list(dict.fromkeys(types))  # stable unique
    keep = []
    for t in ts:
        # drop t if there exists u that is a strict subtype of t
        if not any(u is not t and issubclass_or_role(u, t) for u in ts):
            keep.append(t)
    return keep


@lru_cache
def issubclass_or_role(child: Type, parent: Type | Tuple[Type, ...]) -> bool:
    """
    Check if `child` is a subclass of `parent` or if `child` is a Role whose role taker is a subclass of `parent`.

    :param child: The child class.
    :param parent: The parent class.
    :return: True if `child` is a subclass of `parent` or if `child` is a Role for `parent`, False otherwise.
    """
    from krrood.patterns.role.role import Role

    if issubclass(child, parent):
        return True
    if issubclass(child, Role) and child is not Role:
        role_taker_type = child.get_role_taker_type()
        if issubclass_or_role(role_taker_type, parent):
            return True
    return False


@lru_cache
def nearest_common_ancestor(classes):
    return next(all_nearest_common_ancestors(classes), None)


def all_nearest_common_ancestors(classes) -> Iterator[Type]:
    if not classes:
        return
    method_resolution_orders = {cls: copy(cls.mro()) for cls in classes}
    yield from _all_nearest_common_ancestors_from_classes_method_resolution_order(method_resolution_orders)



@lru_cache
def role_aware_nearest_common_ancestor(classes):
    return next(role_aware_all_nearest_common_ancestors(classes), None)


def role_aware_all_nearest_common_ancestors(classes) -> Iterator[Type]:
    if not classes:
        return

    from krrood.patterns.role.role import Role

    # Get MROs as lists
    method_resolution_orders = {cls: copy(cls.mro()) for cls in classes}
    for cls, method_resolution_order in method_resolution_orders.items():
        if Role not in method_resolution_order:
            continue
        rol_idx = method_resolution_order.index(Role)
        method_resolution_order[rol_idx] = cls.get_role_taker_type()

    yield from _all_nearest_common_ancestors_from_classes_method_resolution_order(method_resolution_orders)


def _all_nearest_common_ancestors_from_classes_method_resolution_order(method_resolution_orders: Dict[Type, List[Type]]) -> Iterator[Type]:
    # Iterate in MRO order of the first class
    method_resolution_orders_values = list(method_resolution_orders.values())
    seen_candidates = set()
    for candidate in method_resolution_orders_values[0]:
        if any(issubclass(seen_candidate, candidate) for seen_candidate in seen_candidates):
            continue
        if all(candidate in mro for mro in method_resolution_orders_values[1:]):
            seen_candidates.add(candidate)
            yield candidate
