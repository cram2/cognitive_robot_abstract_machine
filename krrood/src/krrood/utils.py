"""
Standard-library helpers for naming and resolving classes, modules and import paths, and
reading dataclass fields.
"""

from __future__ import annotations

import builtins
import importlib
import os
import sys
import types
from dataclasses import Field
from dataclasses import fields, MISSING
from functools import lru_cache
from inspect import isclass
from os import PathLike
from os.path import dirname
from pathlib import Path
from typing_extensions import (
    get_origin,
    get_args,
    Generic,
    Hashable,
    Any,
)

from typing_extensions import TypeVar, Callable, TypeVarTuple, _SpecialForm

from krrood.exceptions import (
    NoModuleSourceProvided,
    NoDefaultValueFound,
    PackageNameNotFoundError,
    PathMissingRequiredPartsError,
)

T = TypeVar("T")


def recursive_subclasses(cls: type[T]) -> list[type[T]]:
    """
    :param cls: The class.
    :return: A list of the classes subclasses without the class itself.
    """
    return cls.__subclasses__() + [
        g for s in cls.__subclasses__() for g in recursive_subclasses(s)
    ]


def get_full_class_name(cls):
    """
    Return the full name of a class, including the module name.

    :param cls: The class.
    :return: The full name of the class
    """
    return cls.__module__ + "." + cls.__name__


def module_and_class_name(type_: type | _SpecialForm) -> str:
    """
    :param type_: A class or special form.
    :return: Its fully qualified ``"{module}.{name}"`` identifier.
    """
    return f"{get_module_of_type(type_)}.{type_.__name__}"


def get_module_of_type(type_: type | _SpecialForm) -> str:
    """
    :param type_: The type of which the module is obtained.
    :return: The module name of the given type_.
    """
    if type_ is types.NoneType:
        return "types"
    return type_.__module__


def resolve_class_from_full_name(fully_qualified_class_name: str) -> type:
    """
    Import and return the class named by a fully qualified name of the form
    ``"module.submodule.ClassName"``, as written by :func:`get_full_class_name` or
    :func:`module_and_class_name`.

    :param fully_qualified_class_name: The fully qualified class name.
    :return: The resolved class.
    """
    module_name, class_name = fully_qualified_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def get_default_value(dataclass_type, field_name):
    """
    Return the default value for a given field in a dataclass.

    :param dataclass_type: The dataclass type to get the default value for.
    :param field_name: The name of the field to get the default value for.
    :return: The default value for the field.
    """
    for f in fields(dataclass_type):
        if f.name != field_name:
            continue
        if f.default is not MISSING:
            return f.default
        elif f.default_factory is not MISSING:  # handles mutable defaults
            return f.default_factory()
        else:
            raise NoDefaultValueFound(clazz=dataclass_type, field_name=field_name)
    return None


def get_default_values_for_dataclass(dataclass_type):
    """
    Return a dict mapping field names to their default values.

    Only includes fields that actually define a default.

    :param dataclass_type: The dataclass type to get the default values for.
    :return: A dict mapping field names to their default values.
    """
    defaults = {}

    for f in fields(dataclass_type):
        if f.default is not MISSING:
            defaults[f.name] = f.default
        elif f.default_factory is not MISSING:
            defaults[f.name] = f.default_factory()

    return defaults


@lru_cache
def own_dataclass_fields(cls) -> list[Field]:
    """
    :return: The fields of the dataclass that are not inherited from a base class.
    """
    base_fields = set()
    for base in cls.__mro__[1:]:
        if hasattr(base, "__dataclass_fields__"):
            base_fields.update(base.__dataclass_fields__.keys())

    return [f for f in fields(cls) if f.name not in base_fields]


def is_typing_type(type_object: Any):
    """
    :param type_object: A type object to check.
    :return: True if the type is a type from the typing module, False otherwise.
    """
    return hasattr(type_object, "__module__") and type_object.__module__ == "typing"


def is_builtin_type(type_object: Any):
    """
    :param type_object: A type object to check.
    :return: True if the type is a built-in type, False otherwise.
    """
    return (
        isinstance(type_object, type)
        and type_object.__module__ == "builtins"
        and hasattr(builtins, type_object.__name__)
    )


def get_import_root_from_path(path: Path) -> Path:
    """
    Find the directory an import of a path is resolved against.

    :param path: The file system path to find the import root of.
    :return: The nearest ancestor of the path that is not itself a package.
    """
    root = Path(path).resolve()
    while (root / "__init__.py").exists():
        parent = root.parent
        if parent == root:
            break
        root = parent
    return root


def get_import_path_from_path(path: str) -> str | None:
    """
    Convert a file system path to a Python import path.

    :param path: The file system path to convert.
    :return: The Python import path, or None if the path is not inside a package.
    """
    absolute_path = Path(path).resolve()
    root = get_import_root_from_path(absolute_path)
    if root == absolute_path:
        return None
    return str(absolute_path.relative_to(root)).replace(os.path.sep, ".")


def make_path_importable(path: Path) -> None:
    """
    Put the directory an import of a path is resolved against on the search path.

    ..note:: Generated code is written wherever its caller chose, which need not be a
        directory Python already searches, so importing it back requires saying where
        to look for it.

    :param path: The file system path that is about to be imported.
    """
    root = str(get_import_root_from_path(path))
    if root not in sys.path:
        sys.path.insert(0, root)


def get_function_import_data(func: Callable) -> tuple[str, str]:
    """
    Get the import path of a function.

    :param func: The function to get the import path for.
    :return: The import path of the function.
    """
    func_name = get_method_name(func)
    func_class_name = get_method_class_name_if_exists(func)
    func_file_path = get_method_file_name(func)
    func_file_name = func_file_path.split("/")[-1].split(".")[
        0
    ]  # Get the file name without extension
    func_import_path = get_import_path_from_path(dirname(func_file_path))
    func_import_path = (
        f"{func_import_path}.{func_file_name}" if func_import_path else func_file_name
    )
    if func_class_name and func_class_name != func_name:
        func_import_name = func_class_name
    else:
        func_import_name = func_name
    return func_import_path, func_import_name


def get_method_name(method: Callable) -> str:
    """
    Get the name of a method.

    :param method: The method to get the name of.
    :return: The name of the method.
    """
    return method.__name__ if hasattr(method, "__name__") else str(method)


def get_class_and_attribute_name(class_name: str, attribute_name: str) -> str:
    """
    Return the dot-qualified name ``"{class_name}.{attribute_name}"``.

    :param class_name: The owner class name, typically ``SomeClass.__name__``.
    :param attribute_name: The attribute or variable name to qualify.
    :return: The qualified name string.
    """
    return f"{class_name}.{attribute_name}"


def get_method_class_name_if_exists(method: Callable) -> str | None:
    """
    Get the class name of a method if it has one.

    :param method: The method to get the class name of.
    :return: The class name of the method.
    """
    if hasattr(method, "__self__"):
        if hasattr(method.__self__, "__name__"):
            return method.__self__.__name__
        elif hasattr(method.__self__, "__class__"):
            return method.__self__.__class__.__name__
    return (
        method.__qualname__.split(".")[0]
        if hasattr(method, "__qualname__") and "." in method.__qualname__
        else None
    )


def get_method_file_name(method: Callable) -> str:
    """
    Get the file name of a method.

    :param method: The method to get the file name of.
    :return: The file name of the method.
    """
    return method.__code__.co_filename


def get_relative_import(
    target_file_path: str | PathLike[str],
    imported_module_path: str | None = None,
    module_name: str | None = None,
    package_name: str | None = None,
) -> str:
    """
    Get a relative import path from the target file to the imported module.

    :param target_file_path: The file path of the target file.
    :param imported_module_path: The file path of the module being imported.
    :param module_name: The module name, if available.
    :param package_name: The name of the root package where the module is located.
    :return: A relative import path as a string.
    """
    # Convert to absolute paths
    if module_name is not None:
        imported_module_path = sys.modules[module_name].__file__
    if imported_module_path is None:
        raise NoModuleSourceProvided(
            imported_module_path=imported_module_path, module_name=module_name
        )
    target_path = Path(target_file_path).resolve()
    imported_file_name = Path(imported_module_path).name
    target_file_name = Path(target_file_path).name
    if package_name is not None:
        target_path = Path(
            get_path_starting_from_latest_encounter_of(
                str(target_path), package_name, [target_file_name]
            )
        )
    imported_path = Path(imported_module_path).resolve()
    if package_name is not None:
        imported_path = Path(
            get_path_starting_from_latest_encounter_of(
                str(imported_path), package_name, [imported_file_name]
            )
        )

    # Compute relative path from target to imported module
    rel_path = os.path.relpath(imported_path.parent, target_path.parent)

    # Convert path to Python import format
    rel_parts = [part.replace("..", ".") for part in Path(rel_path).parts]
    rel_parts = rel_parts if rel_parts else [""]
    dot_parts = [part for part in rel_parts if part == "."]
    non_dot_parts = [part for part in rel_parts if part != "."] + [imported_path.stem]

    # Join the parts
    joined_parts = "." + "".join(dot_parts) + ".".join(non_dot_parts)

    return joined_parts


def get_path_starting_from_latest_encounter_of(
    path: str, package_name: str, should_contain: list[str]
) -> str:
    """
    Get the path starting from the package name.

    :param path: The full path to the file.
    :param package_name: The name of the package to start from.
    :param should_contain: The names of the files or directories to look for.
    :return: The path starting from the package name that contains all the names in
        should_contain, otherwise raise an error.
    :raise PackageNameNotFoundError: If the package name could not be found in the path.
    :raise PathMissingRequiredComponentsError: If the path does not contain all the
        names in should_contain.
    """
    path_parts = path.split(os.path.sep)
    if package_name not in path_parts:
        raise PackageNameNotFoundError(package_name=package_name, path=path)
    idx = path_parts.index(package_name)
    prev_idx = idx
    while all(sc in path_parts[idx:] for sc in should_contain):
        prev_idx = idx
        try:
            idx = path_parts.index(package_name, idx + 1)
        except ValueError:
            break
    if all(sc in path_parts[idx:] for sc in should_contain):
        path_parts = path_parts[prev_idx:]
        return os.path.join(*path_parts)
    else:
        raise PathMissingRequiredPartsError(required_parts=should_contain, path=path)


def get_generic_type_parameters(
    cls,
    generic_base: type,
    include_root_generic_base: bool = True,
    include_specialized_generic_base: bool = True,
) -> list[type[T]]:
    """
    Given a subclass and its generic base, return the concrete type parameter(s).

    Example:
        get_generic_type_parameters(Employee, Role) -> [<class '__main__.Person'>]

    Direct parameterizations (e.g. ``class C(B, Generic[U])``) take priority over
    an inherited binding discovered by recursing into an unparameterized base.

    :param cls: The subclass to check.
    :param generic_base: The generic base class to check against.
    :param include_root_generic_base: Whether to include type parameters the class gets from its own typing.Generic directly.
    :param include_specialized_generic_base: Whether to include type parameters from superclasses that are generic, which are not typing.Generic.
    :return: A list of concrete type parameters
    """
    parameters = []
    if include_root_generic_base:
        # Use __parameters__ to get the class's own unbound TypeVars.
        parameters.extend(list(getattr(cls, "__parameters__", [])))

    if include_specialized_generic_base:
        for base in getattr(cls, "__orig_bases__", []):
            base_origin = get_origin(base)
            if (
                not base_origin
                or base_origin is Generic
                or not issubclass(base_origin, generic_base)
            ):
                continue
            for argument in get_args(base):
                if not isinstance(argument, (TypeVar, TypeVarTuple)):
                    parameters.append(argument)
                elif not include_root_generic_base:
                    # If we specifically excluded root generic parameters, we might still want
                    # TypeVars that are being passed to this specialized base
                    # Example: For `class Child(Generic[T, U], Parent[T])`:
                    # - `include_root_generic_base=True` returns `[T, U]` (captures all definitions, avoids duplicates).
                    # - `include_root_generic_base=False` returns `[T]` (captures only what is specifically passed to `Parent`).
                    parameters.append(argument)

    return parameters


def get_existing_field_by_name(cls, name: str) -> Field | None:
    """
    Find the existing field in the MRO if it exists.

    :param name: The name of the field.
    :return: The existing field if found, otherwise None.
    """
    for base in cls.__mro__:
        fields = getattr(base, "__dataclass_fields__", None)
        if fields and name in fields:
            return fields[name]
    return None


def is_hashable(obj) -> bool:
    """
    Checks if an object is hashable by attempting to compute its hash.

    :param obj: The object to check.
    :return: True if the object is hashable, False otherwise.
    """
    try:
        hash(obj)
        return True
    except TypeError:
        return False


def ensure_hashable(obj) -> Hashable:
    """
    :return: The object itself if it is hashable, otherwise its id.
    """
    if not is_hashable(obj):
        return id(obj)
    return obj


def is_dynamic_class(cls: type) -> bool:
    """
    Check if a class is dynamically created.

    This is done by checking if the class is actually registered in that module under
    its own name Normal classes will be found; classes created with  for instance
    make_dataclass  usually won't be unless manually assigned.
    :param cls: The class to check.
    :return: True if the class is dynamically created, False otherwise.
    """
    # Ensure it is a class first
    if not isclass(cls):
        return False

    # Get the module where the class claims to be defined
    module = sys.modules.get(cls.__module__)
    if module is None:
        return True  # If module doesn't exist, it's likely dynamic

    return getattr(module, cls.__name__, None) is not cls
