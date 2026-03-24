from __future__ import annotations

import ast
import importlib
import inspect
import os
import subprocess
import sys
import types
from ast import Module
from collections import defaultdict
from dataclasses import dataclass, field, fields, Field
from functools import lru_cache
from importlib.util import resolve_name
from inspect import isclass
from os.path import dirname
from pathlib import Path
from typing import Union, Type, Optional, Callable, Tuple, List, Iterable, Dict, Any

from typing_extensions import (
    TypeVar,
    Type,
    List,
    Optional,
    _SpecialForm,
    Iterable,
    Dict,
    get_origin,
    get_args,
)

from krrood import logger

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


@lru_cache
def inheritance_path_length(child_class: Type, parent_class: Type) -> Optional[int]:
    """
    Calculate the inheritance path length between two classes.
    Every inheritance level that lies between `child_class` and `parent_class` increases the length by one.
    In case of multiple inheritance, the path length is calculated for each branch and the minimum is returned.

    :param child_class: The child class.
    :param parent_class: The parent class.
    :return: The minimum path length between `child_class` and `parent_class` or None if no path exists.
    """
    if not (
        isclass(child_class)
        and isclass(parent_class)
        and issubclass(child_class, parent_class)
    ):
        return None

    return _inheritance_path_length(child_class, parent_class, 0)


def _inheritance_path_length(
    child_class: Type, parent_class: Type, current_length: int = 0
) -> int:
    """
    Helper function for :func:`inheritance_path_length`.

    :param child_class: The child class.
    :param parent_class: The parent class.
    :param current_length: The current length of the inheritance path.
    :return: The minimum path length between `child_class` and `parent_class`.
    """

    if child_class == parent_class:
        return current_length
    else:
        return min(
            _inheritance_path_length(base, parent_class, current_length + 1)
            for base in child_class.__bases__
            if issubclass(base, parent_class)
        )


def module_and_class_name(t: Union[Type, _SpecialForm]) -> str:
    return f"{t.__module__}.{t.__name__}"


def extract_imports_from(
    module: Optional[types.ModuleType] = None,
    file_path: Optional[str] = None,
    source: Optional[str] = None,
    ast_tree: Optional[ast.AST] = None,
    exclude_libraries: Optional[List[str]] = None,
    convert_relative_to_absolute: bool = False,
) -> List[str]:
    """
    Extract imports from a module or source code or a file path or an ast and returns them as a list of strings.

    :param module: The module to extract imports from.
    :param file_path: The file path to extract imports from.
    :param source: The source code to extract imports from.
    :param ast_tree: The ast tree to extract imports from.
    :param exclude_libraries: A list of libraries to exclude from the imports.
    :param convert_relative_to_absolute: Whether to convert relative imports to absolute imports.
    """
    exclude_libraries = exclude_libraries or []
    if module is None and source is None and file_path is None and ast_tree is None:
        raise ValueError(
            "Either module, source, file_path or ast_tree must be provided"
        )
    if module:
        source = inspect.getsource(module)
    elif file_path:
        with open(file_path, "r") as f:
            source = f.read()

    tree = ast_tree or ast.parse(source)

    import_modules = set()
    from_imports = defaultdict(set)

    current_module = module.__name__

    for node in ast.walk(tree):

        # import x
        if isinstance(node, ast.Import):

            for alias in node.names:
                name = alias.name

                if name in exclude_libraries:
                    continue

                if alias.asname:
                    import_modules.add(f"{name} as {alias.asname}")
                else:
                    import_modules.add(name)

        # from x import y
        elif isinstance(node, ast.ImportFrom):

            prefix = "." * node.level
            module_name = node.module or ""
            full_module = f"{prefix}{module_name}"

            if convert_relative_to_absolute and node.level > 0:
                full_module = resolve_name(full_module, current_module)

            if node.module and node.module in exclude_libraries:
                continue

            for alias in node.names:
                if alias.asname:
                    from_imports[full_module].add(f"{alias.name} as {alias.asname}")
                else:
                    from_imports[full_module].add(alias.name)

    result = set()

    for mod in import_modules:
        result.add(f"import {mod}")

    for mod, names in from_imports.items():
        joined = ", ".join(sorted(names))
        result.add(f"from {mod} import {joined}")

    return sorted(result)


def generate_relative_import(
    from_module: str, target_module: str, symbol: str | None = None
) -> str:
    """
    Generate a relative import statement using Python's own resolver.

    :param from_module: The module where the import is being made.
    :param target_module: The module to import.
    :param symbol: The symbol (e.g., a class, a method, ..., etc.) to import (optional).
    """

    # Compute absolute module name as Python would resolve it
    absolute = resolve_name(target_module, from_module)

    from_pkg = from_module.rsplit(".", 1)[0]
    from_parts = from_pkg.split(".")
    target_parts = absolute.split(".")

    # find common prefix
    i = 0
    while (
        i < min(len(from_parts), len(target_parts)) and from_parts[i] == target_parts[i]
    ):
        i += 1

    up = len(from_parts) - i
    prefix = "." * (up + 1)

    remainder = ".".join(target_parts[i:])

    if symbol:
        if remainder:
            return f"from {prefix}{remainder} import {symbol}"
        return f"from {prefix} import {symbol}"
    else:
        return f"from {prefix} import {remainder}"


@lru_cache
def own_dataclass_fields(cls) -> List[Field]:
    """
    :return: The fields of the dataclass that are not inherited from a base class.
    """
    base_fields = set()
    for base in cls.__mro__[1:]:
        if hasattr(base, "__dataclass_fields__"):
            base_fields.update(base.__dataclass_fields__.keys())

    return [f for f in fields(cls) if f.name not in base_fields]


def get_type_names_per_module_from_types(
    type_objects: Iterable[Type],
    excluded_names: Optional[List[str]] = None,
    excluded_modules: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """
    Get a dictionary of type names grouped by module.

    :param type_objects: A list of type objects to format.
    :param excluded_names: A list of names to exclude from the imports.
    :param excluded_modules: A list of modules to exclude from the imports.
    :return: A dictionary of type names grouped by module.
    """
    excluded_modules = [] if excluded_modules is None else excluded_modules
    excluded_names = [] if excluded_names is None else excluded_names
    module_to_types = defaultdict(list)
    for type_object in type_objects:
        try:
            if isinstance(type_object, type) or is_typing_type(type_object):
                module = type_object.__module__
                name = type_object.__qualname__
            elif callable(type_object):
                module, name = get_function_import_data(type_object)
            elif hasattr(type(type_object), "__module__"):
                module = type(type_object).__module__
                name = type(type_object).__qualname__
            else:
                continue
            if name == "NoneType":
                module = "types"
            if (
                module is None
                or module == "builtins"
                or module.startswith("_")
                or module in sys.builtin_module_names
                or module in excluded_modules
                or "<" in module
                or name in excluded_names
                or "site-packages" in module.split(".")
            ):
                continue
            if module == "typing":
                module = "typing_extensions"
            module_to_types[module].append(name)
        except AttributeError:
            continue
    return module_to_types


def is_typing_type(type_object: Type):
    """
    :param type_object: A type object to check.
    :return: True if the type is a type from the typing module, False otherwise.
    """
    return type_object.__module__ == "typing"


def is_builtin_type(type_object: Type):
    """
    :param type_object: A type object to check.
    :return: True if the type is a built-in type, False otherwise.
    """
    return isinstance(type_object, type) and type_object.__module__ == "builtins"


def get_import_path_from_path(path: str) -> Optional[str]:
    """
    Convert a file system path to a Python import path.

    :param path: The file system path to convert.
    :return: The Python import path.
    """
    package_name = os.path.abspath(path)
    packages = package_name.split(os.path.sep)
    parent_package_idx = 0
    for i in range(len(packages)):
        if i == 0:
            current_path = package_name
        else:
            current_path = "/" + "/".join(packages[:-i])
        if os.path.exists(os.path.join(current_path, "__init__.py")):
            parent_package_idx -= 1
        else:
            break
    package_name = (
        ".".join(packages[parent_package_idx:]) if parent_package_idx < 0 else None
    )
    return package_name


def get_function_import_data(func: Callable) -> Tuple[str, str]:
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


def get_method_class_name_if_exists(method: Callable) -> Optional[str]:
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
    target_file_path,
    imported_module_path: Optional[str] = None,
    module: Optional[str] = None,
    package_name: Optional[str] = None,
) -> str:
    """
    Get a relative import path from the target file to the imported module.

    :param target_file_path: The file path of the target file.
    :param imported_module_path: The file path of the module being imported.
    :param module: The module name, if available.
    :param package_name: The name of the root package where the module is located.
    :return: A relative import path as a string.
    """
    # Convert to absolute paths
    if module is not None:
        imported_module_path = sys.modules[module].__file__
    if imported_module_path is None:
        raise ValueError("Either imported_module_path or module must be provided")
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
    path: str, package_name: str, should_contain: List[str]
) -> str:
    """
    Get the path starting from the package name.

    :param path: The full path to the file.
    :param package_name: The name of the package to start from.
    :param should_contain: The names of the files or directorys to look for.
    :return: The path starting from the package name that contains all the names in should_contain, otherwise raise an error.
    :raise ValueError: If the path does not contain all the names in should_contain.
    """
    path_parts = path.split(os.path.sep)
    if package_name not in path_parts:
        raise ValueError(f"Could not find {package_name} in {path}")
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
        raise ValueError(f"Could not find {should_contain} in {path}")


def get_imports_from_types(
    type_objects: Iterable[Type],
    target_file_path: Optional[str] = None,
    package_name: Optional[str] = None,
    excluded_names: Optional[List[str]] = None,
    excluded_modules: Optional[List[str]] = None,
) -> List[str]:
    """
    Format import lines from type objects.

    :param type_objects: A list of type objects to format.
    :param target_file_path: The file path to which the imports should be relative.
    :param package_name: The name of the package to use for relative imports.
    :param excluded_names: A list of names to exclude from the imports.
    :param excluded_modules: A list of modules to exclude from the imports.
    :return: A list of formatted import lines.
    """
    module_to_types = get_type_names_per_module_from_types(
        type_objects, excluded_names, excluded_modules
    )

    lines = []
    stem_imports = []
    for module, names in module_to_types.items():
        filtered_names = set()
        for name in set(names):
            if "." in name:
                stem = ".".join(name.split(".")[1:])
                name_to_import = name.split(".")[0]
                filtered_names.add(name_to_import)
                stem_imports.append(f"{stem} = {name_to_import}.{stem}")
            else:
                filtered_names.add(name)
        joined = ", ".join(sorted(set(filtered_names)))
        import_path = module
        if (
            (target_file_path is not None)
            and (package_name is not None)
            and (package_name in module)
        ):
            import_path = get_relative_import(
                target_file_path, module=module, package_name=package_name
            )
        lines.append(f"from {import_path} import {joined}")
    lines.extend(stem_imports)
    return lines


def run_black_on_file(filename: str):
    """
    Format the file with black
    """
    command = [sys.executable, "-m", "black", filename]
    run_subprocess_on_file(command)


def run_ruff_on_file(filename: str):
    """
    Format the file with ruff
    """
    command = ["ruff", "check", "--fix", filename]
    run_subprocess_on_file(command)


def run_subprocess_on_file(command: List[str]):
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Ruff failed with code {e.returncode}\n"
            f"STDOUT:\n{e.stdout}\n"
            f"STDERR:\n{e.stderr}"
        ) from e


def get_generic_type_param(cls, generic_base: Type[T]) -> Optional[List[Type[T]]]:
    """
    Given a subclass and its generic base, return the concrete type parameter(s).

    Example:
        get_generic_type_param(Employee, Role) -> (<class '__main__.Person'>,)
    """
    for base in getattr(cls, "__orig_bases__", []):
        base_origin = get_origin(base)
        if base_origin is None:
            if isclass(base) and issubclass(base, generic_base):
                res = get_generic_type_param(base, generic_base)
                if res:
                    return res
            continue
        if issubclass(base_origin, generic_base):
            args = get_args(base)
            return list(args) if args else None
    return None


import libcst as cst


class TypeCheckingImportCollector(cst.CSTVisitor):
    def __init__(self):
        self.imports = []

    def visit_If(self, node: cst.If):
        if self.is_type_checking(node.test):
            for stmt in node.body.body:
                if isinstance(stmt, cst.SimpleStatementLine):
                    for small in stmt.body:
                        if isinstance(small, (cst.Import, cst.ImportFrom)):
                            self.imports.append(small)

    def is_type_checking(self, test):
        # if TYPE_CHECKING
        if isinstance(test, cst.Name):
            return test.value == "TYPE_CHECKING"

        # if typing.TYPE_CHECKING
        if isinstance(test, cst.Attribute):
            if isinstance(test.value, cst.Name):
                return (
                    test.value.value in ["typing", "typing_extensions"]
                    and test.attr.value == "TYPE_CHECKING"
                )

        return False


def get_type_checking_imports(file_path):
    with open(file_path) as f:
        module = cst.parse_module(f.read())

    visitor = TypeCheckingImportCollector()
    module.visit(visitor)

    return visitor.imports


def get_scope_from_imports(
    file_path: Optional[str] = None,
    tree: Optional[ast.AST] = None,
    package_name: Optional[str] = None,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Creates a scope dictionary from imports in a Python file or an AST tree.

    :param file_path: The path to the Python file to extract imports from.
    :param tree: An AST tree to extract imports from. If provided, file_path is ignored.
    :param package_name: The name of the package to use for relative imports.
    :param source: The source code to extract imports from. If provided, file_path and tree are ignored.
    """
    if tree is None and file_path is None and source is None:
        raise ValueError("Either file_path, tree, or source must be provided")
    if file_path and source is None:
        with open(file_path, "r") as f:
            source = f.read()
    tree = tree or ast.parse(source, filename=file_path)

    scope = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name
                asname = alias.asname or alias.name
                try:
                    scope[asname] = importlib.import_module(
                        module_name, package=package_name
                    )
                except ImportError as e:
                    logger.warning(f"Could not import {module_name}: {e}")
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module
            for alias in node.names:
                name = alias.name
                asname = alias.asname or name
                try:
                    if node.level > 0:  # Handle relative imports
                        package_name = get_import_path_from_path(
                            Path(
                                os.path.join(file_path, *[".."] * node.level)
                            ).resolve()
                        )
                    if (
                        package_name is not None and node.level > 0
                    ):  # Handle relative imports
                        module_rel_path = Path(
                            os.path.join(file_path, *[".."] * node.level, module_name)
                        ).resolve()
                        idx = str(module_rel_path).rfind(package_name)
                        if idx != -1:
                            module_name = str(module_rel_path)[idx:].replace(
                                os.path.sep, "."
                            )
                    try:
                        module = importlib.import_module(
                            module_name, package=package_name
                        )
                    except ModuleNotFoundError:
                        module = importlib.import_module(
                            f"{package_name}.{module_name}"
                        )
                    if name == "*":
                        scope.update(module.__dict__)
                    else:
                        scope[asname] = getattr(module, name)
                except (ImportError, AttributeError) as e:
                    logger.warning(
                        f"Could not import {module_name}: {e} while extracting imports from {file_path}"
                    )

    return scope
