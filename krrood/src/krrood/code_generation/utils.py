"""
General utilities for generating Python source code.

These functions are infrastructure used by multiple krrood subsystems
(ORMatic, RDR, EQL-RDR, class_diagrams).  Domain-specific logic lives in
the respective packages.
"""

from __future__ import annotations

import ast
import enum
import inspect
import logging
import os
import re
import subprocess
import sys
import textwrap
import typing
from collections import defaultdict
from importlib.util import resolve_name
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import typing_extensions
from typing_extensions import TYPE_CHECKING

from krrood.exceptions import (
    ModuleNotFoundForConvertingImportsToAbsolute,
    NoModuleSourceProvided,
    NoSourceDataToParseImportsFrom,
    SubprocessExecutionError,
)
from krrood.utils import (
    get_function_import_data,
    get_import_path_from_path,
    get_method_class_name_if_exists,
    get_method_file_name,
    get_method_name,
    get_path_starting_from_latest_encounter_of,
    is_builtin_type,
    is_typing_type,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class FunctionMissingAnnotationsError(TypeError):
    """Raised at decoration time when a function lacks required type annotations."""


# ---------------------------------------------------------------------------
# Naming utilities
# ---------------------------------------------------------------------------


def str_to_snake_case(snake_str: str) -> str:
    """Convert any string to snake_case.

    :param snake_str: The string to convert.
    :return: The snake_case string.
    """
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", snake_str)
    s1 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
    s1 = re.sub(r"_{2,}", "_", s1)
    s1 = re.sub(r"^_|_$", "", s1)
    return s1


def to_camel_case(name: str) -> str:
    """Convert snake_case to CamelCase. E.g. ``'my_func'`` → ``'MyFunc'``."""
    return "".join(part.capitalize() for part in name.split("_"))


def to_variable_name(class_name: str) -> str:
    """Convert a CamelCase class name to a lowerCamelCase variable name.

    E.g. ``"Distance"`` → ``"distance"``, ``"MyDistance"`` → ``"myDistance"``.
    """
    return class_name[0].lower() + class_name[1:] if class_name else class_name


# ---------------------------------------------------------------------------
# Callable inspection
# ---------------------------------------------------------------------------


def generate_callable_import(func: Callable) -> Tuple[str, str]:
    """Return ``(import_line, access_expression)`` for *func*.

    :param func: The callable to generate an import for.
    :returns: A 2-tuple: the ``from … import …`` line and the name expression
        used to reference the callable after that import.

    Module-level function ``distance`` in ``my.module``::

        ("from my.module import distance", "distance")

    Method ``MyClass.distance`` in ``my.module``::

        ("from my.module import MyClass", "MyClass.distance")
    """
    module_name = func.__module__
    qualname = func.__qualname__
    qualname_parts = qualname.split(".")

    parent_segment = qualname_parts[-2] if len(qualname_parts) >= 2 else None
    is_method = (
        parent_segment is not None
        and parent_segment.isidentifier()
        and "<" not in parent_segment
    )

    if is_method:
        class_name = parent_segment
        import_line = f"from {module_name} import {class_name}"
        access_expr = f"{class_name}.{func.__name__}"
    else:
        import_line = f"from {module_name} import {func.__name__}"
        access_expr = func.__name__

    return import_line, access_expr


def validate_annotations(func: Callable) -> None:
    """Raise :exc:`FunctionMissingAnnotationsError` if any required annotation is absent.

    Unannotated ``self`` and ``cls`` parameters are silently excluded.
    """
    sig = inspect.signature(func)
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        if param.annotation is inspect.Parameter.empty:
            raise FunctionMissingAnnotationsError(
                f"Parameter '{param_name}' of '{func.__qualname__}' "
                f"lacks a type annotation."
            )
    if sig.return_annotation is inspect.Parameter.empty:
        raise FunctionMissingAnnotationsError(
            f"Function '{func.__qualname__}' lacks a return type annotation."
        )


# ---------------------------------------------------------------------------
# Import extraction / generation
# ---------------------------------------------------------------------------


def extract_imports_from(
    module: Optional[types.ModuleType] = None,
    file_path: Optional[str] = None,
    source: Optional[str] = None,
    ast_tree: Optional[ast.AST] = None,
    exclude_libraries: Optional[List[str]] = None,
    convert_relative_to_absolute: bool = False,
) -> List[str]:
    """Extract import statements from a module, source, file path, or AST.

    :param module: The module to extract imports from.
    :param file_path: The file path to extract imports from.
    :param source: The source code to extract imports from.
    :param ast_tree: The ast tree to extract imports from.
    :param exclude_libraries: A list of libraries to exclude from the imports.
    :param convert_relative_to_absolute: Whether to convert relative imports to absolute.
    :returns: A sorted list of import-line strings.
    """
    exclude_libraries = exclude_libraries or []
    if module is None and source is None and file_path is None and ast_tree is None:
        raise NoSourceDataToParseImportsFrom(
            module=module, file_path=file_path, ast_tree=ast_tree
        )
    current_module_name = None
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

    import_modules: Set[str] = set()
    from_imports: Dict[str, Set[str]] = defaultdict(set)

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

    result: Set[str] = set()

    for mod in sorted(import_modules):
        result.add(f"import {mod}")

    for mod, names in sorted(from_imports.items()):
        joined = ", ".join(sorted(names))
        result.add(f"from {mod} import {joined}")

    return sorted(result)


def generate_relative_import(
    from_module: str, target_module: str, symbol: Optional[str] = None
) -> str:
    """Generate a relative import statement using Python's own resolver.

    :param from_module: The module where the import is being made.
    :param target_module: The module to import.
    :param symbol: The symbol (e.g., a class, a method, etc.) to import (optional).
    :returns: A relative import statement string.
    """
    absolute = resolve_name(target_module, from_module)

    from_pkg = from_module.rsplit(".", 1)[0]
    from_parts = from_pkg.split(".")
    target_parts = absolute.split(".")

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


def get_type_names_per_module_from_types(
    type_objects: Iterable[Type],
    excluded_names: Optional[List[str]] = None,
    excluded_modules: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Get a dictionary of type names grouped by module.

    :param type_objects: A list of type objects to format.
    :param excluded_names: A list of names to exclude from the imports.
    :param excluded_modules: A list of modules to exclude from the imports.
    :return: A dictionary of type names grouped by module.
    """
    excluded_modules = [] if excluded_modules is None else excluded_modules
    excluded_names = [] if excluded_names is None else excluded_names
    module_to_types: Dict[str, List[str]] = defaultdict(list)
    for type_object in type_objects:
        try:
            if isinstance(type_object, type) or is_typing_type(type_object):
                mod = type_object.__module__
                name = type_object.__qualname__
            elif callable(type_object):
                mod, name = get_function_import_data(type_object)
            elif hasattr(type(type_object), "__module__"):
                mod = type(type_object).__module__
                name = type(type_object).__qualname__
            else:
                continue
            if name == "NoneType":
                mod = "types"
            if (
                mod is None
                or mod == "builtins"
                or mod.startswith("_")
                or mod in sys.builtin_module_names
                or mod in excluded_modules
                or "<" in mod
                or name in excluded_names
                or "site-packages" in mod.split(".")
            ):
                continue
            if mod == "typing":
                mod = "typing_extensions"
            module_to_types[mod].append(name)
        except AttributeError:
            continue
    return module_to_types


def get_imports_from_types(
    type_objects: Iterable[Type],
    target_file_path: Optional[str] = None,
    package_name: Optional[str] = None,
    excluded_names: Optional[List[str]] = None,
    excluded_modules: Optional[List[str]] = None,
) -> List[str]:
    """Format import lines from type objects.

    :param type_objects: A list of type objects to format.
    :param target_file_path: The file path to which the imports should be relative.
    :param package_name: The name of the package to use for relative imports.
    :param excluded_names: A list of names to exclude from the imports.
    :param excluded_modules: A list of modules to exclude from the imports.
    :return: A list of formatted import lines.
    """
    from krrood.utils import get_relative_import

    module_to_types = get_type_names_per_module_from_types(
        type_objects, excluded_names, excluded_modules
    )

    lines: List[str] = []
    stem_imports: List[str] = []
    for module, names in module_to_types.items():
        filtered_names: Set[str] = set()
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


# ---------------------------------------------------------------------------
# Type-hint serialisation
# ---------------------------------------------------------------------------

# Mapping from origin types to their typing hint equivalents
_ORIGIN_TYPE_TO_HINT: Dict[type, type] = {
    list: List,
    set: Set,
    dict: Dict,
    tuple: Tuple,
}


def _extract_types(tp: Any, seen: Optional[Set[type]] = None) -> Set[type]:
    """Recursively extract all base types from a type hint."""
    from typing import ForwardRef, get_args, get_origin

    if seen is None:
        seen = set()

    if tp in seen or isinstance(tp, str):
        return seen

    if isinstance(tp, ForwardRef):
        return seen

    origin = get_origin(tp)
    args = get_args(tp)

    if origin:
        if origin in _ORIGIN_TYPE_TO_HINT:
            seen.add(_ORIGIN_TYPE_TO_HINT[origin])
        else:
            seen.add(origin)
        for arg in args:
            _extract_types(arg, seen)
    elif isinstance(tp, type):
        seen.add(tp)

    return seen


def stringify_type_hint(tp: Any) -> str:
    """Recursively convert a type hint to its string representation.

    Handles :class:`~typing.ForwardRef`, generic aliases (e.g. ``List[int]``),
    builtins, and qualified names.  This is the **single canonical** function
    for converting type hints to source-code strings — use it everywhere
    instead of manual ``t.__name__`` concatenation.

    :param tp: The type hint to convert.
    :returns: A Python source-code string for the type.
    """
    from typing import ForwardRef, get_args, get_origin

    if isinstance(tp, str):
        return tp

    if isinstance(tp, ForwardRef):
        return tp.__forward_arg__

    origin = get_origin(tp)
    args = get_args(tp)

    if origin is not None:
        origin_str = getattr(origin, "__name__", str(origin)).capitalize()
        args_str = ", ".join(stringify_type_hint(arg) for arg in args)
        return f"{origin_str}[{args_str}]"

    if isinstance(tp, type):
        if tp.__module__ == "builtins":
            return tp.__name__
        return f"{tp.__qualname__}"

    return str(tp)


# Backward-compatible alias for the old name
stringify_hint = stringify_type_hint


def value_to_source(value: object) -> str:
    """Convert a Python value to its source-code representation.

    Handles: ``None``, booleans, integers, floats, strings, enum members,
    and type objects.  Falls back to ``repr(value)`` for unrecognised types.

    :param value: The Python value to convert.
    :returns: A source-code string.
    """
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, enum.Enum):
        return f"{type(value).__name__}.{value.name}"
    if isinstance(value, type):
        return value.__name__
    return repr(value)


def get_types_to_import_from_type_hints(hints: List[Type]) -> Set[Type]:
    """Extract importable types from a list of type hints.

    :param hints: A list of type hints to extract types from.
    :return: A set of types that need to be imported.
    """
    seen_types = _extract_types(None, set())
    for hint in hints:
        _extract_types(hint, seen_types)

    to_import: Set[Type] = set()
    for tp in seen_types:
        from typing import ForwardRef

        if isinstance(tp, ForwardRef) or isinstance(tp, str):
            continue
        if not is_builtin_type(tp):
            to_import.add(tp)

    return to_import


def get_types_to_import_from_func_type_hints(func: Callable) -> Set[Type]:
    """Extract importable types from a function's annotations.

    :param func: The function to extract type hints from.
    :returns: A set of types that need to be imported.
    """
    hints = typing.get_type_hints(func)

    sig = inspect.signature(func)
    all_hints = list(hints.values())
    if sig.return_annotation != inspect.Signature.empty:
        all_hints.append(sig.return_annotation)

    for param in sig.parameters.values():
        if param.annotation != inspect.Parameter.empty:
            all_hints.append(param.annotation)

    return get_types_to_import_from_type_hints(all_hints)


# ---------------------------------------------------------------------------
# Source extraction
# ---------------------------------------------------------------------------


def extract_function_or_class_file(
    file_path: str,
    function_names: List[str],
    join_lines: bool = True,
    return_line_numbers: bool = False,
    include_signature: bool = True,
    as_list: bool = False,
    is_class: bool = False,
) -> Union[
    Dict[str, Union[str, List[str]]],
    Tuple[Dict[str, Union[str, List[str]]], Dict[str, Tuple[int, int]]],
]:
    """Extract the source code of a function/class from a file.

    :param file_path: The path to the file.
    :param function_names: The names of the functions/classes to extract.
    :param join_lines: Whether to join the lines of the function.
    :param return_line_numbers: Whether to return the line numbers.
    :param include_signature: Whether to include the function/class signature.
    :param as_list: Whether to return a list of sources instead of a dict.
    :param is_class: Whether to also look for class definitions.
    :return: A dictionary mapping names to source code, or a tuple with line numbers.
    """
    with open(file_path, "r") as f:
        source = f.read()

    return extract_function_or_class_from_source(
        source,
        function_names,
        join_lines=join_lines,
        return_line_numbers=return_line_numbers,
        include_signature=include_signature,
        as_list=as_list,
        is_class=is_class,
    )


def extract_function_or_class_from_source(
    source: str,
    function_names: List[str],
    join_lines: bool = True,
    return_line_numbers: bool = False,
    include_signature: bool = True,
    as_list: bool = False,
    is_class: bool = False,
) -> Union[
    Dict[str, Union[str, List[str]]],
    Tuple[Dict[str, Union[str, List[str]]], Dict[str, Tuple[int, int]]],
]:
    """Extract the source code of a function/class from source text.

    :param source: The string containing the source code.
    :param function_names: The names of the functions/classes to extract.
    :param join_lines: Whether to join the lines of the function.
    :param return_line_numbers: Whether to return the line numbers.
    :param include_signature: Whether to include the function/class signature.
    :param as_list: Whether to return a list of sources instead of a dict.
    :param is_class: Whether to also look for class definitions.
    :return: A dictionary mapping names to source code, or a tuple with line numbers.
    """
    # Ensure function_names is a list (avoid circular import from rdr.utils).
    # Keep the same semantics as the original ``make_list``: strings are
    # treated as a single element, not iterated over character-by-character.
    if isinstance(function_names, str) or not isinstance(function_names, list):
        try:
            iter(function_names)
            function_names = (
                list(function_names)
                if not isinstance(function_names, str)
                else [function_names]
            )
        except TypeError:
            function_names = [function_names]

    tree = ast.parse(source)
    functions_source: Dict[str, Union[str, List[str]]] = {}
    functions_source_list: List[Union[str, List[str]]] = []
    line_numbers: Dict[str, Tuple[int, int]] = {}
    line_numbers_list: List[Tuple[int, int]] = []
    look_for_type = ast.ClassDef if is_class else ast.FunctionDef

    for node in tree.body:
        if isinstance(node, look_for_type) and (
            node.name in function_names or len(function_names) == 0
        ):
            lines = source.splitlines()
            func_lines = lines[node.lineno - 1 : node.end_lineno]
            if not include_signature:
                func_lines = func_lines[1:]
            if as_list:
                line_numbers_list.append((node.lineno, node.end_lineno))
            else:
                line_numbers[node.name] = (node.lineno, node.end_lineno)
            parsed_function = (
                textwrap.dedent("\n".join(func_lines)) if join_lines else func_lines
            )
            if as_list:
                functions_source_list.append(parsed_function)
            else:
                functions_source[node.name] = parsed_function
            if len(function_names) > 0:
                if len(functions_source) >= len(function_names) or len(
                    functions_source_list
                ) >= len(function_names):
                    break
    if len(functions_source) < len(function_names) and len(functions_source_list) < len(
        function_names
    ):
        logger.warning(
            f"Could not find all functions: {function_names} not found, "
            f"functions not found: {set(function_names) - set(functions_source.keys())}"
        )
    if return_line_numbers:
        return functions_source if not as_list else functions_source_list, (
            line_numbers if not as_list else line_numbers_list
        )
    return functions_source if not as_list else functions_source_list


# ---------------------------------------------------------------------------
# FunctionCase dataclass generation
# ---------------------------------------------------------------------------


def function_to_dataclass_source(
    func: Callable,
    base_class_fqn: str = (
        "krrood.entity_query_language.rdr.function_case.FunctionCase"
    ),
    class_name: Optional[str] = None,
) -> str:
    """Emit Python source for a ``@dataclass`` subclass of ``FunctionCase``.

    The emitted class has:

    - ``function: ClassVar[Callable] = <access_expr>`` — bound to the decorated
      callable via a module-level import (wrapped in try/except so the source
      can also be exec'd in isolated test namespaces).
    - One field per annotated parameter (``self`` / ``cls`` excluded).
    - ``_output: <return_annotation>`` — the attribute the RDR will predict.

    :param func: The callable to generate a case type for.
    :param base_class_fqn: Fully-qualified name of the base class to inherit from.
    :param class_name: Override for the generated class name.  When ``None`` the
        name is derived from ``func.__name__`` via :func:`to_camel_case`.
    :raises FunctionMissingAnnotationsError: If any required annotation is absent.
    :returns: A Python source string that can be written to a ``.py`` file.
    """
    validate_annotations(func)

    if class_name is None:
        class_name = to_camel_case(func.__name__)
    import_line, access_expr = generate_callable_import(func)

    base_module, base_class_name = base_class_fqn.rsplit(".", 1)

    # Resolve string annotations (produced by `from __future__ import annotations`
    # in the caller's module) to actual type objects before formatting.
    try:
        type_hints: Dict[str, object] = typing.get_type_hints(func)
    except NameError:
        type_hints = {}

    # Collect custom types referenced by annotations.
    sig = inspect.signature(func)
    referenced_types: Dict[str, type] = {}
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        t = type_hints.get(param_name, param.annotation)
        if isinstance(t, type) and t.__module__ not in ("builtins",):
            referenced_types[t.__name__] = t
    t_ret = type_hints.get("return", sig.return_annotation)
    if isinstance(t_ret, type) and t_ret.__module__ not in ("builtins",):
        referenced_types[t_ret.__name__] = t_ret

    # Generate type-import lines using the centralized import generator.
    type_import_lines = "\n".join(
        get_imports_from_types(list(referenced_types.values()))
    )
    type_imports_str = type_import_lines + "\n" if type_import_lines else ""

    # Use centralized stringify_type_hint instead of inline _type_name helper.
    field_lines = [
        f"    {param_name}: {stringify_type_hint(type_hints.get(param_name, param.annotation))}"
        for param_name, param in sig.parameters.items()
        if param_name not in ("self", "cls")
    ]
    return_ann_str = stringify_type_hint(
        type_hints.get("return", sig.return_annotation)
    )

    lines = [
        "from __future__ import annotations",
        "from dataclasses import dataclass",
        "from typing_extensions import ClassVar, Callable",
        f"from {base_module} import {base_class_name}",
        type_imports_str,
        "try:",
        f"    {import_line}",
        "except ImportError:",
        "    pass",
        "",
        "",
        "@dataclass",
        f"class {class_name}({base_class_name}):",
        f'    """FunctionCase for the `{func.__name__}` function."""',
        *field_lines,
        f"    _output: {return_ann_str}",
        "",
        "",
        # Assign function ClassVar outside the class body so that Python's
        # @dataclass machinery (which sees string annotations under PEP 563)
        # never confuses it for an instance field with a default value.
        "try:",
        f"    {class_name}.function = {access_expr}",
        "except NameError:",
        "    pass",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def run_subprocess_on_file(command: List[str]) -> None:
    """Run a subprocess command and handle errors.

    :param command: The command to run as a list of arguments.
    :raises SubprocessExecutionError: If the subprocess command fails.
    """
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise SubprocessExecutionError(command, e.returncode, e.stdout, e.stderr) from e


def run_black_on_file(filename: str) -> None:
    """Format *filename* with Black.

    :param filename: The name of the file to format.
    """
    command = [sys.executable, "-m", "black", filename]
    run_subprocess_on_file(command)


def run_ruff_on_file(filename: str) -> None:
    """Lint and fix *filename* with Ruff.

    :param filename: The name of the file to format.
    """
    command = ["ruff", "check", "--fix", filename]
    run_subprocess_on_file(command)
