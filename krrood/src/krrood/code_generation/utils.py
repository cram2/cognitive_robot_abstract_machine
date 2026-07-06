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
    ForwardRef,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
)

import typing_extensions
from typing_extensions import TYPE_CHECKING

from krrood.exceptions import (
    ModuleNotFoundForConvertingImportsToAbsolute,
    NoModuleSourceProvided,
    NoSourceDataToParseImportsFrom,
    SubprocessExecutionError,
)
from krrood.class_diagrams.utils import get_type_hints_of_object
from krrood.code_generation.generator import CodeGenerator
from krrood.code_generation.source_extraction_utils import (
    extract_function_or_class_file,
    extract_function_or_class_from_source,
    extract_imports_from,
)
from krrood.utils import (
    get_function_import_data,
    get_import_path_from_path,
    get_method_class_name_if_exists,
    get_method_file_name,
    get_method_name,
    get_path_starting_from_latest_encounter_of,
    get_relative_import,
    is_builtin_type,
    is_typing_type,
)

logger = logging.getLogger(__name__)

_generator = CodeGenerator(
    template_dir=os.path.join(os.path.dirname(__file__), "templates")
)

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


def _extract_types(type_hint: Any, seen: Optional[Set[type]] = None) -> Set[type]:
    """Recursively extract all base types from a type hint."""
    if seen is None:
        seen = set()

    if type_hint in seen or isinstance(type_hint, str):
        return seen

    if isinstance(type_hint, ForwardRef):
        return seen

    origin = get_origin(type_hint)
    args = get_args(type_hint)

    if origin:
        if origin in _ORIGIN_TYPE_TO_HINT:
            seen.add(_ORIGIN_TYPE_TO_HINT[origin])
        else:
            seen.add(origin)
        for arg in args:
            _extract_types(arg, seen)
    elif isinstance(type_hint, type):
        seen.add(type_hint)

    return seen


def stringify_type_hint(type_hint: Any) -> str:
    """Recursively convert a type hint to its string representation.

    Handles :class:`~typing.ForwardRef`, generic aliases (e.g. ``List[int]``),
    builtins, and qualified names.  This is the **single canonical** function
    for converting type hints to source-code strings — use it everywhere
    instead of manual ``t.__name__`` concatenation.

    :param type_hint: The type hint to convert.
    :returns: A Python source-code string for the type.
    """
    if isinstance(type_hint, str):
        return type_hint

    if isinstance(type_hint, ForwardRef):
        return type_hint.__forward_arg__

    origin = get_origin(type_hint)
    args = get_args(type_hint)

    if origin is not None:
        origin_str = getattr(origin, "__name__", str(origin)).capitalize()
        args_str = ", ".join(stringify_type_hint(arg) for arg in args)
        return f"{origin_str}[{args_str}]"

    if isinstance(type_hint, type):
        if type_hint.__module__ == "builtins":
            return type_hint.__name__
        return f"{type_hint.__qualname__}"

    return str(type_hint)


# Backward-compatible alias for the old name
stringify_hint = stringify_type_hint


def value_to_source(value: object) -> str:
    """Convert a Python value to its source-code representation.

    Handles: ``None``, booleans, integers, floats, strings, enum members,
    and type objects.  Falls back to ``repr(value)`` for unrecognized types.

    :param value: The Python value to convert.
    :returns: A source-code string.
    """
    if value is None:
        return "None"
    if isinstance(value, enum.Enum):
        return f"{type(value).__name__}.{value.name}"
    if isinstance(value, type) or is_typing_type(value):
        return stringify_type_hint(value)
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
    hints = get_type_hints_of_object(func)

    sig = inspect.signature(func)
    all_hints = list(hints.values())
    if sig.return_annotation != inspect.Signature.empty:
        all_hints.append(sig.return_annotation)

    for param in sig.parameters.values():
        if param.annotation != inspect.Parameter.empty:
            all_hints.append(param.annotation)

    return get_types_to_import_from_type_hints(all_hints)


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
        type_hints: Dict[str, object] = get_type_hints_of_object(func)
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

    # Build field data for the Jinja2 template.
    fields = [
        {
            "name": param_name,
            "type_str": stringify_type_hint(
                type_hints.get(param_name, param.annotation)
            ),
        }
        for param_name, param in sig.parameters.items()
        if param_name not in ("self", "cls")
    ]
    return_ann_str = stringify_type_hint(
        type_hints.get("return", sig.return_annotation)
    )

    return _generator.render(
        "function_case.py.jinja",
        base_module=base_module,
        base_class_name=base_class_name,
        class_name=class_name,
        func_name=func.__name__,
        import_line=import_line,
        access_expr=access_expr,
        type_imports_str=type_imports_str,
        fields=fields,
        return_type_str=return_ann_str,
    )


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
