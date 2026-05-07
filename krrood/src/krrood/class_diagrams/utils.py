from __future__ import annotations

import builtins
import inspect
import sys
from copy import copy
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import get_args, get_origin
from uuid import UUID

import typing_extensions
from typing_extensions import (
    Iterable,
    Iterator,
    List,
    Type,
    Any,
    Dict,
    Tuple,
    TypeVar,
    Callable,
)

from krrood import logger
from krrood.class_diagrams.exceptions import CouldNotResolveType
from krrood.utils import get_scope_from_imports, is_builtin_type


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


def behaves_like_a_built_in_type(
    clazz: Type,
) -> bool:
    return (
        is_builtin_type(clazz)
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


def _trace_generic_params(cls: type, generic_base: type):
    """Trace how cls parameterizes generic_base by walking its __orig_bases__.

    Handles both subscripted bases (e.g. ``B[T]``) and plain bases (e.g. ``B``),
    recursively resolving TypeVar substitutions at each level of the hierarchy.

    :param cls: The class whose hierarchy is searched.
    :param generic_base: The target generic base whose parameters should be resolved.
    :return: A tuple of type arguments for generic_base as seen from cls, or None.
    """
    for base in getattr(cls, "__orig_bases__", []):
        origin = get_origin(base)
        if origin is None:
            if isinstance(base, type) and issubclass(base, generic_base):
                result = _trace_generic_params(base, generic_base)
                if result is not None:
                    return result
        elif issubclass(origin, generic_base):
            args = get_args(base)
            if origin is generic_base:
                return args
            inner = _trace_generic_params(origin, generic_base)
            if inner is None:
                return args
            params = getattr(origin, "__parameters__", ())
            if not params:
                return inner
            sub = {p: a for p, a in zip(params, args) if isinstance(p, TypeVar)}
            return tuple(sub.get(p, p) if isinstance(p, TypeVar) else p for p in inner)
    return None


def get_generic_type_param(cls, generic_base):
    """
    Given a subclass and its generic base, return the concrete type parameter(s).

    Correctly traces TypeVar substitutions through transitive subclass relationships.
    When a direct base is a subscripted transitive subclass of generic_base (e.g.
    ``B[T]`` where ``B`` is a subclass of ``generic_base`` but not ``generic_base``
    itself), the substitution is resolved through the full inheritance chain rather
    than returning ``B``'s type arguments directly.

    Example:
        get_generic_type_param(Employee, Role) -> (<class '__main__.Person'>,)
    """
    orig_bases = cls.__orig_bases__ if hasattr(cls, "__orig_bases__") else []
    for base in orig_bases:
        origin = get_origin(base)
        if origin is None:
            continue
        if not issubclass(origin, generic_base):
            continue
        args = get_args(base)
        if origin is generic_base:
            return args
        inner = _trace_generic_params(origin, generic_base)
        if inner is None:
            return args
        params = getattr(origin, "__parameters__", ())
        if not params:
            return inner
        sub = {p: a for p, a in zip(params, args) if isinstance(p, TypeVar)}
        return tuple(sub.get(p, p) if isinstance(p, TypeVar) else p for p in inner)
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


def _resolve_annotation_typevar(param: TypeVar, generic_base: type) -> TypeVar:
    """
    Return the annotation-level TypeVar that param corresponds to in generic_base.

    When a class inherits from a generic like ``SubClassSafeGeneric[TSpecific]``,
    Python's ``__parameters__`` may expose the *defining* TypeVar (e.g. ``T`` from
    ``SubClassSafeGeneric``) rather than ``TSpecific``.  This function walks
    ``generic_base.__orig_bases__`` to find the actual TypeVar used in annotations.

    :param param: A TypeVar from ``generic_base.__parameters__``.
    :param generic_base: The class whose bases are searched.
    :return: The annotation-level TypeVar, or param if no aliasing is found.
    """
    for base in getattr(generic_base, "__orig_bases__", []):
        base_origin = get_origin(base)
        if base_origin is None:
            continue
        for bp, ba in zip(getattr(base_origin, "__parameters__", ()), get_args(base)):
            if bp is param and isinstance(ba, TypeVar):
                return ba
    return param


@dataclass(frozen=True)
class GenericTypeSubstitution:
    """
    Represents the TypeVar-to-type mappings produced by specializing a generic class.

    :param substitution: A mapping of TypeVars to their substitutions.
    """

    substitution: Dict[TypeVar, Any]

    @classmethod
    def from_specialization(
        cls, concrete_class: type, generic_base: type
    ) -> GenericTypeSubstitution:
        """
        Build a substitution from how concrete_class specializes generic_base.

        Resolves TypeVar aliasing that arises when ``generic_base`` inherits from
        another generic (e.g. ``SubClassSafeGeneric[TSpecific]``): in that case
        ``__parameters__`` may expose the *defining* TypeVar rather than the
        annotation-level one, so the mapping is adjusted via the base's
        ``__orig_bases__``.

        :param concrete_class: The class that specializes generic_base.
        :param generic_base: The generic base class being specialized.
        :return: A GenericTypeSubstitution representing the TypeVar mappings.
        """
        params = getattr(generic_base, "__parameters__", ())
        args = get_generic_type_param(concrete_class, generic_base) or ()
        substitution = {
            _resolve_annotation_typevar(p, generic_base): a
            for p, a in zip(params, args)
        }
        return cls(substitution)

    def apply(self, type_hint: Any) -> TypeHintResolutionResult:
        """
        Apply the substitution to a type hint.

        :param type_hint: The type hint to resolve.
        :return: A TypeHintResolutionResult with the resolved type and a flag indicating if substitution occurred.
        """
        return resolve_type(type_hint, self.substitution)

    @property
    def has_substitutions(self) -> bool:
        """
        Return True if any TypeVar is actually mapped to a different type.
        """
        return any(key is not value for key, value in self.substitution.items())


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
        subscript_param = new_params[0] if len(new_params) == 1 else tuple(new_params)
        return TypeHintResolutionResult(
            type_to_resolve[subscript_param], resolved, type_to_resolve
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
    yield from _all_nearest_common_ancestors_from_classes_method_resolution_order(
        method_resolution_orders
    )


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

    yield from _all_nearest_common_ancestors_from_classes_method_resolution_order(
        method_resolution_orders
    )


def _all_nearest_common_ancestors_from_classes_method_resolution_order(
    method_resolution_orders: Dict[Type, List[Type]],
) -> Iterator[Type]:
    # Iterate in MRO order of the first class
    method_resolution_orders_values = list(method_resolution_orders.values())
    seen_candidates = set()
    for candidate in method_resolution_orders_values[0]:
        if any(
            issubclass(seen_candidate, candidate) for seen_candidate in seen_candidates
        ):
            continue
        if all(candidate in mro for mro in method_resolution_orders_values[1:]):
            seen_candidates.add(candidate)
            yield candidate


def get_property_return_type(property_value: property) -> Any:
    """Return the return-type annotation of a property.

    :param property_value: The property descriptor.
    :return: The resolved return type, or None if unavailable.
    """
    try:
        hints = get_type_hints_of_object(property_value.fget)
        return hints.get("return")
    except Exception:
        raw = property_value.fget.__annotations__.get("return")
        return raw if raw else None


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
        except NameError as name_error:
            object_from_name = resolve_name_in_hierarchy(name_error.name, object_)
            local_namespace[name_error.name] = object_from_name
        except TypeError as type_error:
            logger.warning(
                f"Could not get type hints for {object_} due to TypeError: {type_error}. This may be caused by a type"
                f" hint that cannot be resolved. Returning empty type hints."
            )
            raise
    return type_hints


def same_package(module_a: str, module_b: str) -> bool:
    """Return True when both module names belong to the same top-level package."""
    top_a = module_a.split(".")[0]
    top_b = module_b.split(".")[0]
    return bool(top_a) and top_a == top_b


def topological_sort_by_inheritance(classes: list[type]) -> list[type]:
    """Return classes sorted so that ancestors come before their descendants."""
    result: list[type] = []
    remaining = list(classes)
    while remaining:
        for cls in remaining:
            if not any(
                issubclass(cls, other) and other is not cls for other in remaining
            ):
                result.append(cls)
                remaining.remove(cls)
                break
        else:
            result.extend(remaining)
            break
    return result


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
    elif name in builtins.__dict__:
        return builtins.__dict__[name]
    else:
        raise CouldNotResolveType(
            name,
            extra_information=f"Could not find {name} in {source_path}, could be a deprecated import statement or "
            f"a type defined in a module that is not imported in the source file.",
        )
