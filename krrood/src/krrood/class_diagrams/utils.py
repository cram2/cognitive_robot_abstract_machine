from __future__ import annotations

import builtins
import inspect
import os
import sys
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from uuid import UUID

import typing_extensions
from typing_extensions import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    TypeVarTuple,
)

from krrood import logger
from krrood.class_diagrams.exceptions import CouldNotResolveType
from krrood.utils import ensure_hashable, get_scope_from_imports, is_builtin_type


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
    """
    Return whether *clazz* should be treated as a built-in for class-diagram purposes.

    :param clazz: The class to check.
    :return: True for built-in types, :class:`uuid.UUID`, and enums.
    """
    return (
        is_builtin_type(clazz)
        or clazz == UUID
        or (inspect.isclass(clazz) and issubclass(clazz, Enum))
    )


def common_base_class(types: List[Type]) -> Optional[Type]:
    """
    Return the lowest common ancestor of *types*, or ``None`` if the only
    common ancestor is :class:`object`.

    Non-class entries (e.g. unresolved forward references) are silently
    skipped.  If no classes remain after filtering, ``None`` is returned.
    """
    classes = [t for t in types if inspect.isclass(t)]
    if not classes:
        return None
    if len(classes) == 1:
        return classes[0]
    common = set(classes[0].__mro__)
    for t in classes[1:]:
        common &= set(t.__mro__)
    for cls in classes[0].__mro__:
        if cls in common and cls is not object:
            return cls
    return None


def is_builtin_class(clazz: Type) -> bool:
    """
    :param clazz: The class to check.
    :return: Whether *clazz* is defined in the ``builtins`` module.
    """
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
    if isinstance(type_to_resolve, (TypeVar, TypeVarTuple)):
        type_to_resolve_key = ensure_hashable(type_to_resolve)
        if type_to_resolve_key not in substitution:
            return TypeHintResolutionResult(type_to_resolve, False, type_to_resolve)
        return TypeHintResolutionResult(
            substitution[type_to_resolve_key], True, type_to_resolve
        )

    # If the type itself can be indexed (like List[T] or Optional[T])
    parameters = getattr(type_to_resolve, "__parameters__", None)
    if not (hasattr(type_to_resolve, "__getitem__") and parameters):
        return TypeHintResolutionResult(type_to_resolve, False, type_to_resolve)

    new_parameters = []
    resolved: bool = False
    for parameter in parameters:
        if parameter not in substitution:
            new_parameters.append(parameter)
            continue

        value = substitution[parameter]
        if isinstance(parameter, TypeVarTuple) and isinstance(value, tuple):
            new_parameters.extend(value)
        else:
            new_parameters.append(value)
        resolved = True

    subscript_parameter = (
        new_parameters[0] if len(new_parameters) == 1 else tuple(new_parameters)
    )
    return TypeHintResolutionResult(
        type_to_resolve[subscript_parameter], resolved, type_to_resolve
    )


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
                f" hint that cannot be resolved."
            )
            raise
    return type_hints


@lru_cache(maxsize=None)
def _scope_from_imports_by_mtime(source_path: str, mtime: float) -> Dict[str, Any]:
    """
    Import scope for *source_path*, cached per ``(source_path, mtime)``.

    See :func:`_cached_scope_from_imports`; the *mtime* component of the key is what makes the
    cache self-invalidate when the file changes on disk.

    :param source_path: Path to the source file whose import scope is built.
    :param mtime: Last-modification time of *source_path*, used only as part of the cache key.
    :return: The import scope dictionary for *source_path*.
    """
    return get_scope_from_imports(file_path=source_path)


def _cached_scope_from_imports(source_path: str) -> Dict[str, Any]:
    """
    Return the import scope of *source_path*, reusing a per-file cache.

    The class-diagram type-hint fallback re-parses the same handful of library source files
    thousands of times; caching by ``(path, mtime)`` collapses that to one parse per file while
    self-invalidating if the file changes on disk. The returned dictionary is shared between callers
    and must be treated as read-only, consistent with the :func:`lru_cache`-d
    :func:`get_type_hints_of_object`.

    This cache is intentionally local to the class-diagram resolution path and not applied to
    :func:`~krrood.utils.get_scope_from_imports` itself, because other callers (ripple-down-rules
    code generation) re-read regenerated files within a single process and rely on the uncached
    behaviour.

    :param source_path: Path to the source file whose import scope is needed.
    :return: The import scope dictionary for *source_path*.
    """
    try:
        mtime = os.path.getmtime(source_path)
    except OSError:
        return get_scope_from_imports(file_path=source_path)
    return _scope_from_imports_by_mtime(source_path, mtime)


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
    scope = _cached_scope_from_imports(source_path)
    if name in scope:
        return scope[name]
    if name in builtins.__dict__:
        return builtins.__dict__[name]
    # The cached scope can be incomplete if it was first built while a dependency was only partially
    # initialized (a circular import during module load), which makes a TYPE_CHECKING import such as
    # ``World`` silently unresolvable and that absence gets cached. Recompute the scope without the
    # cache now that imports may have completed before giving up.
    fresh_scope = get_scope_from_imports(file_path=source_path)
    if name in fresh_scope:
        return fresh_scope[name]
    raise CouldNotResolveType(
        name,
        extra_information=f"Could not find {name} in {source_path}, could be a deprecated import statement or "
        f"a type defined in a module that is not imported in the source file.",
    )
