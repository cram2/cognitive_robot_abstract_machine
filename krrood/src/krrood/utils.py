from __future__ import annotations

import ast
import builtins
import importlib
import inspect
import os
import subprocess
import sys
import types
from collections import defaultdict
from copy import deepcopy
from dataclasses import Field
from dataclasses import fields, MISSING
from functools import lru_cache, wraps
from importlib.util import resolve_name
from inspect import isclass
from os import PathLike
from os.path import dirname
from pathlib import Path
from typing import Tuple
from typing import Union, Any

from typing_extensions import TypeVar, Type, List, Optional, Callable
from typing_extensions import (
    _SpecialForm,
    Iterable,
    Dict,
    get_origin,
    get_args,
)

from krrood import logger
from krrood.exceptions import (
    NoSourceDataToParseImportsFrom,
    NoModuleSourceProvided,
    NoDefaultValueFound,
    PackageNameNotFoundError,
    PathMissingRequiredPartsError,
    SubprocessExecutionError,
    SourceDataNotProvided,
    ModuleNotFoundForConvertingImportsToAbsolute,
)

T = TypeVar("T")


def recursive_subclasses(cls: Type[T]) -> List[Type[T]]:
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
            raise NoDefaultValueFound(dataclass_type, field_name)
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
        raise NoSourceDataToParseImportsFrom(
            module=module, file_path=file_path, ast_tree=ast_tree
        )
    if module:
        source = inspect.getsource(module)
        current_module_name = module.__name__
    elif file_path:
        with open(file_path, "r") as f:
            source = f.read()
        current_module_name = os.path.splitext(os.path.basename(file_path))[0]
    elif convert_relative_to_absolute:
        raise ModuleNotFoundForConvertingImportsToAbsolute(
            path=file_path, source_code=source
        )

    tree = ast_tree or ast.parse(source)

    import_modules = set()
    from_imports = defaultdict(set)

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
                full_module = resolve_name(full_module, current_module_name)

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
    target_file_path: str | PathLike[str],
    imported_module_path: Optional[str] = None,
    module_name: Optional[str] = None,
    package_name: Optional[str] = None,
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
    path: str, package_name: str, should_contain: List[str]
) -> str:
    """
    Get the path starting from the package name.

    :param path: The full path to the file.
    :param package_name: The name of the package to start from.
    :param should_contain: The names of the files or directories to look for.
    :return: The path starting from the package name that contains all the names in should_contain, otherwise raise an error.
    :raise PackageNameNotFoundError: If the package name could not be found in the path.
    :raise PathMissingRequiredComponentsError: If the path does not contain all the names in should_contain.
    """
    path_parts = path.split(os.path.sep)
    if package_name not in path_parts:
        raise PackageNameNotFoundError(package_name, path)
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
        raise PathMissingRequiredPartsError(should_contain, path)


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
                target_file_path, module_name=module, package_name=package_name
            )
        lines.append(f"from {import_path} import {joined}")
    lines.extend(stem_imports)
    return lines


def run_black_on_file(filename: str):
    """
    Format the file with black

    :param filename: The name of the file to format.
    """
    command = [sys.executable, "-m", "black", filename]
    run_subprocess_on_file(command)


def run_ruff_on_file(filename: str):
    """
    Format the file with ruff

    :param filename: The name of the file to format.
    """
    command = ["ruff", "check", "--fix", filename]
    run_subprocess_on_file(command)


def run_subprocess_on_file(command: List[str]):
    """
    Run a subprocess command and handle errors.

    :param command: The command to run as a list of arguments.
    :raises SubprocessExecutionError: If the subprocess command fails.
    """
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise SubprocessExecutionError(command, e.returncode, e.stdout, e.stderr) from e


def get_generic_type_param(cls, generic_base: Type[T]) -> Optional[List[Type[T]]]:
    """
    Given a subclass and its generic base, return the concrete type parameter(s).

    Example:
        get_generic_type_param(Employee, Role) -> (<class '__main__.Person'>,)

    :param cls: The subclass to check.
    :param generic_base: The generic base class to check against.
    :return: A list of concrete type parameters, or None if not found.
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


def get_scope_from_imports(
    file_path: Optional[str] = None,
    tree: Optional[ast.AST] = None,
    package_name: Optional[str] = None,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a scope dictionary from imports in a Python file or an AST tree.

    :param file_path: The path to the Python file to extract imports from.
    :param tree: An AST tree to extract imports from. If provided, file_path is ignored.
    :param package_name: The name of the package to use for relative imports.
    :param source: The source code to extract imports from. If provided, file_path and tree are ignored.
    :return: A dictionary representing the scope with imported modules and their attributes.
    """
    if tree is None and file_path is None and source is None:
        raise SourceDataNotProvided(file_path, tree, source)

    # Ensure we have source and a parsed AST
    if file_path and source is None:
        with open(file_path, "r") as f:
            source = f.read()
    parsed_tree = tree or (
        ast.parse(source)
        if file_path is None
        else ast.parse(source, filename=file_path)
    )

    scope: Dict[str, Any] = {}

    for node in ast.walk(parsed_tree):
        if isinstance(node, ast.Import):
            _handle_import_node(node, scope, package_name)
        elif isinstance(node, ast.ImportFrom):
            package_name = _handle_import_from_node(
                node=node,
                scope=scope,
                file_path=file_path,
                package_name=package_name,
            )

    return scope


def _import_module_safely(
    module_name: str, package_name: Optional[str]
) -> Optional[types.ModuleType]:
    """
    Attempt to import a module with an optional package context and return the module or None on failure.

    :param module_name: The name of the module to import.
    :param package_name: The package name to use for relative imports, or None for absolute imports.
    :return: The imported module or None if import fails.
    """
    module = get_module_object(module_name, package_name)
    if module is not None:
        return module

    try:
        return importlib.import_module(module_name, package=package_name)
    except ModuleNotFoundError:
        if not package_name:
            return None
        try:
            if module_name.startswith(".") and package_name:
                full_name = resolve_name(module_name, package_name)
            else:
                full_name = f"{package_name}.{module_name}"
            if full_name in sys.modules:
                return sys.modules[full_name]
            return importlib.import_module(full_name)
        except Exception:
            return None
    except ImportError:
        return None


def get_module_object(
    module_name: str, package_name: Optional[str] = None
) -> Optional[types.ModuleType]:
    """
    :param module_name: The name of the module to import.
    :param package_name: The name of the package containing the module.
    :return: The imported module object.
    """
    try:
        full_name = module_name
        if module_name.startswith(".") and package_name:
            full_name = resolve_name(module_name, package_name)

        if full_name in sys.modules:
            return sys.modules[full_name]
    except Exception:
        return None


def _resolve_relative_import(
    file_path: Optional[str],
    node: ast.ImportFrom,
    module_name: Optional[str],
    package_name: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """
    Resolve relative import context and possibly adjust module and package names based on file location.

    :param file_path: The path to the file containing the import statement.
    :param node: The import from node to process.
    :param module_name: The module name to resolve.
    :param package_name: The package name to use for relative imports.
    :return: a tuple of (resolved_module_name, resolved_package_name).
    """
    resolved_module = module_name
    resolved_package = package_name

    if node.level > 0 and file_path is not None:
        try:
            resolved_package = get_import_path_from_path(
                Path(os.path.join(file_path, *[".."] * node.level)).resolve()
            )
        except Exception:
            # Keep original package if resolution fails
            pass

        if resolved_package is not None and module_name is not None:
            try:
                module_rel_path = Path(
                    os.path.join(file_path, *[".."] * node.level, module_name)
                ).resolve()
                idx = str(module_rel_path).rfind(resolved_package)
                if idx != -1:
                    resolved_module = str(module_rel_path)[idx:].replace(
                        os.path.sep, "."
                    )
            except Exception:
                # Fall back to original module name
                pass

    return resolved_module, resolved_package


def _handle_import_node(
    node: ast.Import, scope: Dict[str, Any], package_name: Optional[str]
) -> None:
    """
    Process a standard import node and update the provided scope mapping.

    :param node: The import node to process.
    :param scope: The scope mapping to update.
    :param package_name: The package name to use for relative imports.
    """
    for alias in node.names:
        module_name = alias.name
        asname = alias.asname or alias.name
        module = _import_module_safely(module_name, package_name)
        if module is not None:
            scope[asname] = module
        else:
            logger.warning(f"Could not import {module_name}")


def _handle_import_from_node(
    node: ast.ImportFrom,
    scope: Dict[str, Any],
    file_path: Optional[str],
    package_name: Optional[str],
) -> Optional[str]:
    """
    Process a from-import node and update the provided scope mapping.

    :param node: The from-import node to process.
    :param scope: The scope mapping to update.
    :param file_path: The path to the file containing the import statement.
    :param package_name: The package name to use for relative imports.
    :return: the (possibly) updated package_name to mimic original behavior.
    """
    module_name = node.module

    # Resolve relative imports (may update package_name and module_name)
    resolved_module_name, resolved_package_name = _resolve_relative_import(
        file_path=file_path,
        node=node,
        module_name=module_name,
        package_name=package_name,
    )

    # Mimic original behavior: allow package_name to be overwritten for subsequent iterations
    package_name = resolved_package_name

    module = None
    if resolved_module_name is not None:
        module = _import_module_safely(resolved_module_name, package_name)

    if module is None and resolved_package_name and resolved_module_name:
        # Fallback already attempted in _import_module_safely; keep for parity
        module = _import_module_safely(
            f"{resolved_package_name}.{resolved_module_name}", None
        )

    if module is None:
        logger.warning(
            f"Could not import {resolved_module_name} while extracting imports from {file_path}"
        )
        return package_name

    for alias in node.names:
        name = alias.name
        asname = alias.asname or name
        try:
            if name == "*":
                scope.update(module.__dict__)
            else:
                scope[asname] = getattr(module, name)
        except AttributeError as e:
            logger.warning(
                f"Could not import {resolved_module_name}: {e} while extracting imports from {file_path}"
            )

    return package_name


T = TypeVar("T", bound=Callable[..., Any])


def memoize(function: T) -> T:
    """
    Caches the return value of a function call at the instance level.
    """

    @wraps(function)
    def wrapper(self, *args: Any, **kwargs: Any) -> T:
        if not hasattr(self, "__memo__"):
            self.__memo__ = {}
        memo = self.__memo__

        key = (function, self, args, frozenset(kwargs.items()))
        try:
            return memo[key]
        except KeyError:
            rv = function(self, *args, **kwargs)
            memo[key] = rv
            return rv

    return wrapper  # type: ignore


def copy_memoize(function: T) -> T:
    """
    Caches the return value of a function call at the instance level but returns a deepcopy of the value.
    """

    @wraps(function)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "__memo__"):
            self.__memo__ = {}
        memo = self.__memo__

        key = (function, self, args, frozenset(kwargs.items()))
        try:
            return deepcopy(memo[key])
        except KeyError:
            rv = function(self, *args, **kwargs)
            memo[key] = rv
            return deepcopy(rv)

    return wrapper


def clear_memoization_cache(instance):
    """
    Clears the memoization cache of an instance.
    """
    if hasattr(instance, "__memo__"):
        instance.__memo__.clear()
